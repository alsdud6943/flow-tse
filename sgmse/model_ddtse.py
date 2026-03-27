from math import ceil
import warnings

import matplotlib.pyplot as plt
from inspect import isfunction
import math
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import time
import os
import numpy as np
from torch.nn import MultiheadAttention
import torch.nn as nn
from torch.nn import functional as F
from torch import einsum
from einops import rearrange, repeat


import hydra
from omegaconf import DictConfig

from sgmse.util.inference import evaluate_model
# from sgmse.util.graphics import visualize_example, visualize_one
from sgmse.util.other import pad_spec, si_sdr_torch
VIS_EPOCHS = 1 

#torch.autograd.set_detect_anomaly(True)

class StochasticRegenerationModelDDTSE(pl.LightningModule):
	def __init__(self,
		backbone_score: nn.Module,
		data_module: nn.Module,
		lr: float = 1e-4, ema_decay: float = 0.999,
		t_eps: float = 3e-2, 
		num_eval_files: int = 100,
		loss_type_score: str = 'mse',
		**kwargs 
	):
		"""
		Create a new ScoreModel for DDTSE with waveform reference support.
		Args:
			backbone: The underlying backbone DNN that serves as a score-based model.
				Must have an output dimensionality equal to the input dimensionality.
			lr: The learning rate of the optimizer. (1e-4 by default).
			ema_decay: The decay constant of the parameter EMA (0.999 by default).
			t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
			reduce_mean: If `True`, average the loss across data dimensions.
				Otherwise sum the loss across data dimensions.
		"""
		super().__init__()
		print("backbone_score type:", type(backbone_score))
		self.backbone_score = backbone_score # Changed assignment
		self.data_module = data_module

		self.t_eps = t_eps
		self.lr = lr
		self.ema_decay = ema_decay
		self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
		self._error_loading_ema = False

		self.loss_type_score = loss_type_score

		self.configure_losses()

		self.num_eval_files = num_eval_files
		# self.save_hyperparameters()

		self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

		self.sigma_min=1e-4
		

	def configure_losses(self):
		# Score Loss
		if self.loss_type_score == "mse":
			self.loss_fn_score = lambda err, vec: self._reduce_op(torch.square(torch.abs(err - vec)))
		else:
			raise NotImplementedError
		
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

		return optimizer

	
	# on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
	def on_load_checkpoint(self, checkpoint):
		ema = checkpoint.get('ema', None)
		if ema is not None:
			self.ema.load_state_dict(checkpoint['ema'])
		else:
			self._error_loading_ema = True
			warnings.warn("EMA state_dict not found in checkpoint!")

	def on_save_checkpoint(self, checkpoint):
		checkpoint['ema'] = self.ema.state_dict()

	def train(self, mode=True, no_ema=False):
		res = super().train(mode)  # call the standard `train` method with the given mode
		if not self._error_loading_ema:
			if mode == False and not no_ema:
				# eval
				self.ema.store(self.parameters())        # store current params in EMA
				self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
			else:
				# train
				if self.ema.collected_params is not None:
					self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
		return res

	def eval(self, no_ema=False):
		return self.train(False, no_ema=no_ema)

	def _loss(self, err, vec):

		loss_score = self.loss_fn_score(err,vec) if self.loss_type_score != "none" else None #mse

		return loss_score

	def _weighted_mean(self, x, w):
		return torch.mean(x * w)
	
	def forward_score(self, x, ref, t, mix_condition, **kwargs):
		"""
		x: (B, F, T) = (batch_size, 256, T)
		ref: (B, 1, emb_dim) = (batch_size, 1, emb_dim) - precomputed speaker embedding
		t: (B)
		mix_condition: (B, F, T) = (batch_size, 256, T)
		"""

		# x.unsqueeze_(1) # (B, 1, F, T)
		# mix_condition = mix_condition.unsqueeze(1) # (B, 1, F, T)
		# dnn_input = torch.cat([x, mix_condition], dim=1) # (B, 2, F, T), 2 here is the number of input sources

		score = self.backbone_score(x = x, ref=ref, mix_condition=mix_condition, time_cond=t)

		if len(score)==2:
			score = -score[0]
		return score

	def sample_x(self, mean, sigma):
		eps = torch.randn_like(mean)
		return mean+eps*sigma


	def _step(self, batch, batch_idx):
		# Updated to handle speaker embedding from dataset
		# Expected batch format: (mix_stft, clean_stft, ref, *paths)
		mix_stft, clean_stft, ref = batch[0], batch[1], batch[2]
		# print(f"ref shape _step: {ref.shape}")  # Debugging line to check ref: torch.Size([2, 1, 256])

		# CFM part
		t1 = torch.rand([mix_stft.shape[0],1,1], device=mix_stft.device)* (1- self.t_eps) + self.t_eps #original code version self.t_eps/2
		
		z = torch.randn_like(mix_stft) 
		y1 = (1 - (1 - self.sigma_min) * t1) * z + t1 * clean_stft 		# Gaussian to x
		vec = clean_stft - (1 - self.sigma_min) * z

		# shape of y1: (B, F, T)
		# (B,D,F,T) in original FlowAVSE code, D=1 if single channel, D=2 if stereo. In my code, assume always single channel

		# Score estimation
		# shape of t1: (B, 1, 1) -> (B)
		t1 = t1.squeeze()
		if clean_stft.shape[0]==1:
			t1 = t1.unsqueeze(0)

		# Squeeze the speaker embedding to remove the extra dimension
		# ref = ref.squeeze(1) # (B, emb_dim)
		score_1 = self.forward_score(x = y1, ref=ref, t=t1, mix_condition=mix_stft)
		
		# Debug: Check for NaN in score
		if torch.isnan(score_1).any() or torch.isinf(score_1).any():
			print(f"WARNING: NaN or inf detected in score_1")
			print(f"score_1 shape: {score_1.shape}, min: {score_1.min()}, max: {score_1.max()}")
			score_1 = torch.nan_to_num(score_1, nan=0.0, posinf=0.0, neginf=0.0)
		
		loss = self._loss(score_1, vec)
		
		return loss

	def training_step(self, batch, batch_idx):
		loss = self._step(batch, batch_idx)

		self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
		return loss

	def optimizer_step(self, *args, **kwargs):
		# Method overridden so that the EMA params are updated after each optimizer step
		super().optimizer_step(*args, **kwargs)
		self.ema.update(self.parameters())

	def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
		loss = self._step(batch, batch_idx)

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files
			if self.current_epoch %2 ==0 and self.current_epoch !=0:
				num_eval_files =100
			# pesq_est, si_sdr_est, estoi_est, ovrl_est, sig_est, sim_est, spec, audio, y_den = evaluate_model(self, num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
			print("")
			print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.2f}")
			print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
			# print(f"OVRL at epoch {self.current_epoch} : {ovrl_est:.2f}")
			# print(f"SIG at epoch {self.current_epoch} : {sig_est:.2f}")
			# print(f"SIM at epoch {self.current_epoch} : {sim_est:.2f}")
			
			print('__________________________________________________________________')

			self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			# self.log('ValidationOVRL', ovrl_est, on_step=False, on_epoch=True, sync_dist=True)
			# self.log('ValidationSIG', sig_est, on_step=False, on_epoch=True, sync_dist=True)
			# self.log('ValidationSIM', sim_est, on_step=False, on_epoch=True, sync_dist=True)

			if audio is not None:
				y_list, x_hat_list, x_list = audio
				for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
					if self.current_epoch == 0:
						self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(-1),self.current_epoch, sample_rate=sr)
						self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(-1),self.current_epoch, sample_rate=sr)
					self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(-1),self.current_epoch, sample_rate=sr)
			'''
			if spec is not None:
				figures = []
				y_stft_list, x_hat_stft_list, x_stft_list = spec
				for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
					figures.append(
						visualize_example(
						torch.abs(y_stft), 
						torch.abs(x_hat_stft), 
						torch.abs(x_stft), return_fig=True))
				self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures) #, sync_dist=True)
			'''
		return loss

	def on_validation_epoch_end(self):
		loss = self.trainer.callback_metrics.get('valid_loss')
		if loss is not None:
			print(f"valid_loss at epoch {self.current_epoch} : {loss:.2f}")

	def to(self, *args, **kwargs):
		self.ema.to(*args, **kwargs)
		return super().to(*args, **kwargs)

	def train_dataloader(self):
		return self.data_module.train_dataloader()

	def val_dataloader(self):
		return self.data_module.val_dataloader()

	def test_dataloader(self):
		return self.data_module.test_dataloader()

	def setup(self, stage=None):
		return self.data_module.setup(stage=stage)

	def to_audio(self, spec, length=None):
		return self._istft(self._backward_transform(spec), length)

	def _forward_transform(self, spec):
		return self.data_module.spec_fwd(spec)

	def _backward_transform(self, spec):
		return self.data_module.spec_back(spec)

	def _stft(self, sig):
		return self.data_module.stft(sig)

	def _istft(self, spec, length=None):
		return self.data_module.istft(spec, length)

	def extract(self, mix, ref, timeit=False,
		return_stft=False,
		num_solver_steps: int = 1, # Added num_solver_steps parameter
		**kwargs
	):

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.

		mix: noisy speech (T)
		ref: ref: (B, 1, emb_dim) = (batch_size, 1, emb_dim) 

		normalization -> padding -> STFT -> forward_transform -> 
		"""

		T_orig = mix.size(0)

		norm_factor = mix.abs().max().item()
		mix = mix / norm_factor
		# ref = ref / norm_factor

		MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0) # MIX: (B, F, T) = (1, F, T)		
		# REF = ref.cuda().unsqueeze(0) # REF: (B, T') = (1, T') - keep as raw waveform for wespeaker
		ref = ref.cuda()

		with torch.no_grad():
			t_span = torch.linspace(self.t_eps, 1, num_solver_steps + 1, device=MIX.device) # Use num_solver_steps and MIX.device
			t, _, dt = t_span[0].item(), t_span[-1].item(), (t_span[1] - t_span[0]).item() # Ensure t and dt are scalars initially
			sol = []
			vector=[]
			current_step_idx = 0 # Use 0-indexed steps for t_span

			x = torch.randn_like(MIX)
			
			while current_step_idx < num_solver_steps:
				t_current = t_span[current_step_idx].item()
				# Ensure t_tensor is a tensor of shape (batch_size,) on the correct device
				t_tensor = torch.full((x.shape[0],), t_current, device=MIX.device)
				
				dphi_dt = self.forward_score(x=x, ref=ref, t=t_tensor, mix_condition=MIX)
				
				# Calculate dt for the current step
				dt_current = (t_span[current_step_idx + 1] - t_span[current_step_idx]).item()
				
				x = x + dt_current * dphi_dt
				# t = t_span[current_step_idx + 1].item() # t for the next iteration's start
				
				sol.append(x)
				vector.append(dphi_dt)
				current_step_idx += 1

			sample = sol[-1] if sol else x # Handle case where num_solver_steps might be 0

			if return_stft:
				pass

		x_hat = self.to_audio(sample.squeeze(), T_orig)
		x_hat = x_hat * norm_factor
		x_hat = x_hat.squeeze().cpu()

		# Debug: Check for NaN in final output
		if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
			print(f"WARNING: NaN or inf detected in final enhanced audio")
			print(f"x_hat shape: {x_hat.shape}, min: {x_hat.min()}, max: {x_hat.max()}")
			x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)

		return x_hat

	# grad norm logging
	def on_before_optimizer_step(self, optimizer):
		norms = [p.grad.detach().data.norm(2) for p in self.parameters() if p.grad is not None]
		total_norm = torch.stack(norms).norm(2)
		self.log("grad_norm", total_norm, batch_size=self.data_module.batch_size)
