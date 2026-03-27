from math import ceil
import warnings
import copy

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

#from .diffusion_utils import checkpoint

import hydra
from omegaconf import DictConfig

import logging

from sgmse.util.inference import evaluate_model
# from sgmse.util.graphics import visualize_example, visualize_one
from sgmse.util.other import pad_spec, si_sdr_torch

VIS_EPOCHS = 1 

log = logging.getLogger(__name__) 

#torch.autograd.set_detect_anomaly(True)

class StochasticRegenerationModel(pl.LightningModule):
	def __init__(self,
		backbone_score: nn.Module = None,
		data_module: nn.Module = None,
		lr: float = 1e-4, ema_decay: float = 0.999,
		t_eps: float = 3e-2, 
		num_eval_files: int = 100,
		loss_type_score: str = 'mse',
		num_solver_steps: int = 1, # should be 1 by default for SingleStepRegenerationModel, adjust in yaml configuration
		**kwargs 
	):
		"""
		Create a new ScoreModel.
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
		
		# Manually instantiate backbone_score and data_module if they are configs
		if isinstance(backbone_score, DictConfig):
			print("Instantiating backbone_score from config")
			self.backbone_score = hydra.utils.instantiate(backbone_score)
		else:
			print("backbone_score type:", type(backbone_score))
			self.backbone_score = backbone_score
			
		if isinstance(data_module, DictConfig):
			print("Instantiating data_module from config")
			self.data_module = hydra.utils.instantiate(data_module)
		else:
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
		self.num_solver_steps = num_solver_steps

	def configure_losses(self):
		# Score Loss
		if self.loss_type_score == "mse":
			self.loss_fn_score = lambda err, vec: torch.mean(torch.square(torch.abs(err - vec)))  # Mean over B, F, T dimensions
			# self.loss_fn_score = lambda err, vec: torch.mean(torch.sum(torch.square(torch.abs(err - vec)), dim=(-2, -1)))  # Sum over F,T dimensions, Mean over batch dimension
			# self.loss_fn_score = lambda err, vec: self._reduce_op(torch.square(torch.abs(err - vec)))
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
		ref: (B, F, T') = (batch_size, 256, T')
		t: (B,)
		mix_condition: (B, F, T) = (batch_size, 256, T)
		"""

		# print("x shape:", x.shape)
		# print("ref shape:", ref.shape)
		# print("mix_condition shape:", mix_condition.shape)

		score = self.backbone_score(x = x, ref=ref, mix_condition=mix_condition, time_cond=t)
		if len(score)==2:
			score = -score[0]
		return score

	def sample_x(self, mean, sigma):
		eps = torch.randn_like(mean)
		return mean+eps*sigma

	def _step(self, batch, batch_idx):
		# mix_stft, clean_stft, regi_stft: complex spectrograms with shape (B, F, T) or (B, F, T')
		mix_stft, clean_stft, regi_stft = batch[0], batch[1], batch[2]

		# print("mix_stft shape:", mix_stft.shape)
		# print("clean_stft shape:", clean_stft.shape)
		# print("regi_stft shape:", regi_stft.shape)

		# CFM part
		t1 = torch.rand([mix_stft.shape[0],1,1], device=mix_stft.device)* (1- self.t_eps) + self.t_eps #original code version self.t_eps/2

		# print("t1 shape:", t1.shape)

		z = torch.randn_like(mix_stft) 
		y1 = (1 - (1 - self.sigma_min) * t1) * z + t1 * clean_stft 		# Gaussian to x
		vec = clean_stft - (1 - self.sigma_min) * z

		# shape of y1: (B, F, T)
		# (B,D,F,T) in original FlowAVSE code, D=1 if single channel, D=2 if stereo. In my code, assume always single channel

		# shape of t1: (B, 1, 1) -> (B,)
		t1 = t1.squeeze(-1).squeeze(-1)  # Remove only the last two dimensions, keeping batch dimension

		score_1 = self.forward_score(x = y1, ref = regi_stft, t=t1, mix_condition=mix_stft)
	
		loss = self._loss(score_1, vec)
		# print(f"score_1 shape: {score_1.shape}, vec shape: {vec.shape}")
		return loss

	def training_step(self, batch, batch_idx):
		loss = self._step(batch, batch_idx)
		# self.ema.update(self.parameters())

		self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
		return loss

	def optimizer_step(self, *args, **kwargs):
		# Method overridden so that the EMA params are updated after each optimizer step
		super().optimizer_step(*args, **kwargs)
		self.ema.update(self.parameters())

	def validation_step(self, batch, batch_idx, sr=16000):
		loss = self._step(batch, batch_idx)

		# print(f"Validation step {batch_idx}, loss: {loss:.2f}\n")
		# Debug info
		# val_dataloader_len = len(self.val_dataloader()) if hasattr(self, 'val_dataloader') else "Unknown"
		# print(f"[GPU {self.global_rank}] Validation step {batch_idx}/{val_dataloader_len}, loss: {loss:.2f}, batch_size: {batch[0].shape[0]}")

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files

			# pesq_est, si_sdr_est, estoi_est, ovrl_est, sig_est, sim_est, spec, audio, y_den = evaluate_model(self, num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, num_solver_steps=self.num_solver_steps, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			pesq_est_1step, si_sdr_est_1step, estoi_est_1step, spec_1step, audio_1step, y_den_1step = evaluate_model(self, num_eval_files, num_solver_steps=1, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			print("")
			print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.2f}")
			print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
			print(f"PESQ 1-step at epoch {self.current_epoch} : {pesq_est_1step:.2f}")
			print(f"SISDR 1-step at epoch {self.current_epoch} : {si_sdr_est_1step:.2f}")
			print(f"ESTOI 1-step at epoch {self.current_epoch} : {estoi_est_1step:.2f}")

			# print(f"OVRL at epoch {self.current_epoch} : {ovrl_est:.2f}")
			# print(f"SIG at epoch {self.current_epoch} : {sig_est:.2f}")
			# print(f"SIM at epoch {self.current_epoch} : {sim_est:.2f}")
			
			print('__________________________________________________________________')

			self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

			self.log('ValidationPESQ_1step', pesq_est_1step, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR_1step', si_sdr_est_1step, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI_1step', estoi_est_1step, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

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
		num_solver_steps: int = 30, # Added num_solver_steps parameter
		**kwargs
	):

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.

		mix: noisy speech (T)
		ref: reference speech (T) 

		normalization -> padding -> STFT -> forward_transform -> 
		"""

		T_orig = mix.size(0)

		norm_factor = mix.abs().max().item()
		mix = mix / norm_factor
		ref = ref / norm_factor

		# mix = F.pad(mix, (0, self.data_module.n_fft - self.data_module.hop_length), mode='constant', value=0)
		# ref = F.pad(ref, (0, self.data_module.n_fft - self.data_module.hop_length), mode='constant', value=0)
		# manual padding seems unnecessary because of center=True in stft

		MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0) # MIX: (B, F, T) = (1, F, T)		
		REF = torch.unsqueeze(self._forward_transform(self._stft(ref.cuda())), 0) # REF: (B, F, T') = (1, F, T')

		with torch.no_grad():
			t_span = torch.linspace(self.t_eps, 1, num_solver_steps + 1, device=MIX.device) # Use num_solver_steps and MIX.device
			t, _, dt = t_span[0].item(), t_span[-1].item(), (t_span[1] - t_span[0]).item() # Ensure t and dt are scalars initially
			sol = []
			vector=[]
			current_step_idx = 0 # Use 0-indexed steps for t_span

			x = torch.randn_like(MIX)
			# x = MIX

			while current_step_idx < num_solver_steps:
				t_current = t_span[current_step_idx].item()
				# Ensure t_tensor is a tensor of shape (batch_size,) on the correct device
				t_tensor = torch.full((x.shape[0],), t_current, device=MIX.device)
				
				dphi_dt = self.forward_score(x=x, ref=REF, t=t_tensor, mix_condition=MIX)
				
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

		return x_hat

	# grad norm logging
	def on_before_optimizer_step(self, optimizer):
		norms = [p.grad.detach().data.norm(2) for p in self.parameters() if p.grad is not None]
		total_norm = torch.stack(norms).norm(2)
		self.log("grad_norm", total_norm)


# You need to set conditional=False in config file
class SingleStepRegenerationModel(StochasticRegenerationModel):
	def _step(self, batch, batch_idx):
		# mix_stft, clean_stft, regi_stft: complex spectrograms with shape (B, F, T) or (B, F, T')
		mix_stft, clean_stft, regi_stft = batch[0], batch[1], batch[2]

		# CFM part
		# t1 = torch.full([mix_stft.shape[0]], self.t_eps, device=mix_stft.device)
		t1 = torch.zeros([mix_stft.shape[0]], device=mix_stft.device)

		z = torch.randn_like(mix_stft) 
		vec = clean_stft - z

		score_1 = self.forward_score(x=z, ref=regi_stft, t=t1, mix_condition=mix_stft)

		loss = self._loss(score_1, vec)
		return loss

	def validation_step(self, batch, batch_idx, sr=16000):
		loss = self._step(batch, batch_idx)

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files

			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, num_solver_steps=1, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)

			print("")
			print(f"PESQ 1-step at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR 1-step at epoch {self.current_epoch} : {si_sdr_est:.2f}")
			print(f"ESTOI 1-step at epoch {self.current_epoch} : {estoi_est:.2f}")

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

	def extract(self, mix, ref, timeit=False,
		return_stft=False,
		num_solver_steps: int = 1, # Added num_solver_steps parameter. need to set to 1 for SingleStepRegenerationModel
		**kwargs
	):

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.

		mix: noisy speech (T)
		ref: reference speech (T) 

		normalization -> padding -> STFT -> forward_transform -> 
		"""

		T_orig = mix.size(0)

		norm_factor = mix.abs().max().item()
		mix = mix / norm_factor
		ref = ref / norm_factor

		MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0) # MIX: (B, F, T) = (1, F, T)		
		REF = torch.unsqueeze(self._forward_transform(self._stft(ref.cuda())), 0) # REF: (B, F, T') = (1, F, T')

		with torch.no_grad():
			x = torch.randn_like(MIX)
			# t1 = torch.full([x.shape[0],], self.t_eps, device=MIX.device)  # t1: (B,)
			t1 = torch.zeros([x.shape[0]], device=MIX.device) # t1: (B,)
			dphi_dt = self.forward_score(x=x, ref=REF, t=t1, mix_condition=MIX)

			x_hat = x + dphi_dt # Single step update
			
		x_hat = self.to_audio(x_hat.squeeze(), T_orig)
		x_hat = x_hat * norm_factor
		x_hat = x_hat.squeeze().cpu()

		return x_hat


class DisrcriminativeRegenerationModel(StochasticRegenerationModel):
	def __init__(self, backbone_score=None, data_module=None, **kwargs):
		super().__init__(backbone_score=backbone_score, data_module=data_module, **kwargs)
	
	def forward_dsc(self, x, ref, **kwargs):
		"""
		Forward pass for discriminative model.
		x: (B, F, T) = (batch_size, 256, T) - speech mixture
		ref: (B, F, T') = (batch_size, 256, T') - reference speech
		"""
		output = self.backbone_score(x=x, ref=ref, time_cond=None)
		return output
	
	def _step(self, batch, batch_idx):
		# mix_stft, clean_stft, regi_stft: complex spectrograms with shape (B, F, T) or (B, F, T')
		mix_stft, clean_stft, regi_stft = batch[0], batch[1], batch[2]
		
		# Forward pass with mixture and reference
		output = self.forward_dsc(x=mix_stft, ref=regi_stft)
		
		# MSE loss between output and clean speech
		loss = self._loss(output, clean_stft)
		return loss

	def validation_step(self, batch, batch_idx, sr=16000):
		loss = self._step(batch, batch_idx)

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files

			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, num_solver_steps=1, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)

			print("")
			print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.2f}")
			print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
			
			print('__________________________________________________________________')

			self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

			if audio is not None:
				y_list, x_hat_list, x_list = audio
				for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
					if self.current_epoch == 0:
						self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(-1),self.current_epoch, sample_rate=sr)
						self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(-1),self.current_epoch, sample_rate=sr)
					self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(-1),self.current_epoch, sample_rate=sr)
		return loss
	
	def extract(self, mix, ref, timeit=False, return_stft=False, **kwargs):
		"""
		One-call speech enhancement for discriminative model.
		
		mix: noisy speech (T)
		ref: reference speech (T)
		"""
		T_orig = mix.size(0)

		norm_factor = mix.abs().max().item()
		mix = mix / norm_factor
		ref = ref / norm_factor

		MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0)  # MIX: (B, F, T) = (1, F, T)		
		REF = torch.unsqueeze(self._forward_transform(self._stft(ref.cuda())), 0)  # REF: (B, F, T') = (1, F, T')

		with torch.no_grad():
			# Direct forward pass without stochastic process
			enhanced = self.forward_dsc(x=MIX, ref=REF)

		x_hat = self.to_audio(enhanced.squeeze(), T_orig)
		x_hat = x_hat * norm_factor
		x_hat = x_hat.squeeze().cpu()

		return x_hat


class DeterministicFlowModel(StochasticRegenerationModel):
	def forward_score(self, x, ref, t, **kwargs):
		"""
		x: (B, F, T) = (batch_size, 256, T)
		ref: (B, F, T') = (batch_size, 256, T')
		t: (B,)
		"""
		score = self.backbone_score(x = x, ref=ref, time_cond=t)
		if len(score)==2:
			score = -score[0]
		return score

	def _step(self, batch, batch_idx):
		# mix_stft, clean_stft, regi_stft: complex spectrograms with shape (B, F, T) or (B, F, T')
		mix_stft, clean_stft, regi_stft = batch[0], batch[1], batch[2]
		
		t1 = torch.rand(mix_stft.shape[0], device=mix_stft.device)

		vec = clean_stft - mix_stft

		# Forward pass with mixture and reference
		score_1 = self.forward_score(x=mix_stft, ref=regi_stft, t=t1)

		# MSE loss between output and clean speech
		loss = self._loss(score_1, vec)
		return loss

	def validation_step(self, batch, batch_idx, sr=16000):
		loss = self._step(batch, batch_idx)

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files

			# pesq_est, si_sdr_est, estoi_est, ovrl_est, sig_est, sim_est, spec, audio, y_den = evaluate_model(self, num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, num_solver_steps=self.num_solver_steps, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			print("")
			print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.2f}")
			print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
			
			print('__________________________________________________________________')

			self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

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

	def extract(self, mix, ref, timeit=False,
		return_stft=False,
		num_solver_steps: int = 30, # Added num_solver_steps parameter
		**kwargs
	):

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.

		mix: noisy speech (T)
		ref: reference speech (T) 

		normalization -> padding -> STFT -> forward_transform -> 
		"""

		T_orig = mix.size(0)

		norm_factor = mix.abs().max().item()
		mix = mix / norm_factor
		ref = ref / norm_factor

		MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0) # MIX: (B, F, T) = (1, F, T)		
		REF = torch.unsqueeze(self._forward_transform(self._stft(ref.cuda())), 0) # REF: (B, F, T') = (1, F, T')

		with torch.no_grad():
			t_span = torch.linspace(0, 1, num_solver_steps + 1, device=MIX.device) # Use num_solver_steps and MIX.device
			t, _, dt = t_span[0].item(), t_span[-1].item(), (t_span[1] - t_span[0]).item() # Ensure t and dt are scalars initially
			sol = []
			vector=[]
			current_step_idx = 0 # Use 0-indexed steps for t_span

			x = MIX

			while current_step_idx < num_solver_steps:
				t_current = t_span[current_step_idx].item()
				# Ensure t_tensor is a tensor of shape (batch_size,) on the correct device
				t_tensor = torch.full((x.shape[0],), t_current, device=MIX.device)
				
				dphi_dt = self.forward_score(x=x, ref=REF, t=t_tensor)
				
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

		return x_hat

# 전부 다 (m, e)로 condtion된거
# mixture+noise -> clean
# noise -> clean

# time embedding 없이 하는법: n번동안 계속 x1으로 난사
# training할 때 t = 1 - (1-t_init)^m		(m>1, t_init in U[0,1])로 해도 됨

# time embedding 있게 하는법: 
# training, sampling 할 때 t = 1 - (1-t_init)^m	(m>1, t_init in U[0,1])으로 뽑기

class StochasticRegenerationModel_saturatetime(StochasticRegenerationModel):
	def __init__(self, backbone_score=None, data_module=None, sat_power=10, prior_sigma=0.1, prior_mean='zero', **kwargs):
		super().__init__(backbone_score=backbone_score, data_module=data_module, **kwargs)
		self.sat_power = sat_power
		self.prior_sigma = prior_sigma
		self.prior_mean = prior_mean

	def _step(self, batch, batch_idx):
		# mix_stft, clean_stft, regi_stft: complex spectrograms with shape (B, F, T) or (B, F, T')
		mix_stft, clean_stft, regi_stft = batch[0], batch[1], batch[2]

		# CFM part
		t1 = torch.rand([mix_stft.shape[0]], device=mix_stft.device) # *(1-self.t_eps) + self.t_eps
		# t1 = 1 - (1 - t1) ** self.sat_power
		t1 = t1 * (1-self.t_eps) + self.t_eps

		# print("t1 shape:", t1.shape)
		t1 = t1.unsqueeze(-1).unsqueeze(-1)  # Add two

		if self.prior_mean == 'zero':
			z = torch.randn_like(mix_stft) * self.prior_sigma 
		elif self.prior_mean == 'mix':
			z = mix_stft + torch.randn_like(mix_stft) * self.prior_sigma

		y1 = (1 - (1 - self.sigma_min) * t1) * z + t1 * clean_stft 		# Gaussian to x
		vec = clean_stft - (1 - self.sigma_min) * z

		# shape of y1: (B, F, T)
		# (B,D,F,T) in original FlowAVSE code, D=1 if single channel, D=2 if stereo. In my code, assume always single channel

		# shape of t1: (B, 1, 1) -> (B,)
		t1 = t1.squeeze(-1).squeeze(-1)  # Remove only the last two dimensions, keeping batch dimension

		score_1 = self.forward_score(x = y1, ref = regi_stft, t=t1, mix_condition=mix_stft)
	
		loss = self._loss(score_1, vec)
		# print(f"score_1 shape: {score_1.shape}, vec shape: {vec.shape}")
		return loss

	def validation_step(self, batch, batch_idx, sr=16000):
		loss = self._step(batch, batch_idx)

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files

			# pesq_est, si_sdr_est, estoi_est, ovrl_est, sig_est, sim_est, spec, audio, y_den = evaluate_model(self, num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, num_solver_steps=self.num_solver_steps, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			pesq_est_1step, si_sdr_est_1step, estoi_est_1step, spec_1step, audio_1step, y_den_1step = evaluate_model(self, num_eval_files, num_solver_steps=1, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			pesq_timesat_est, si_sdr_timesat_est, estoi_timesat_est, spec, audio, y_den = evaluate_model(self, num_eval_files, num_solver_steps=self.num_solver_steps, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, mode='timesat')
			print("")
			print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.2f}")
			print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")

			print(f"PESQ 1-step at epoch {self.current_epoch} : {pesq_est_1step:.2f}")
			print(f"SISDR 1-step at epoch {self.current_epoch} : {si_sdr_est_1step:.2f}")
			print(f"ESTOI 1-step at epoch {self.current_epoch} : {estoi_est_1step:.2f}")

			print(f"PESQ timesat at epoch {self.current_epoch} : {pesq_timesat_est:.2f}")
			print(f"SISDR timesat at epoch {self.current_epoch} : {si_sdr_timesat_est:.2f}")
			print(f"ESTOI timesat at epoch {self.current_epoch} : {estoi_timesat_est:.2f}")

			print('__________________________________________________________________')

			self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

			self.log('ValidationPESQ_1step', pesq_est_1step, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR_1step', si_sdr_est_1step, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI_1step', estoi_est_1step, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

			self.log('ValidationPESQ_timesat', pesq_timesat_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR_timesat', si_sdr_timesat_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI_timesat', estoi_timesat_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

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
	
	def extract(self, mix, ref, timeit=False,
		return_stft=False,
		num_solver_steps: int = 30, # Added num_solver_steps parameter
		**kwargs
	):

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.

		mix: noisy speech (T)
		ref: reference speech (T) 

		normalization -> padding -> STFT -> forward_transform -> 
		"""

		T_orig = mix.size(0)

		norm_factor = mix.abs().max().item()
		mix = mix / norm_factor
		ref = ref / norm_factor

		MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0) # MIX: (B, F, T) = (1, F, T)		
		REF = torch.unsqueeze(self._forward_transform(self._stft(ref.cuda())), 0) # REF: (B, F, T') = (1, F, T')

		with torch.no_grad():
			t_span = torch.linspace(0, 1, num_solver_steps + 1, device=MIX.device)*(1-self.t_eps) + self.t_eps # Use num_solver_steps and MIX.device
			t_span = 1 - (1 - t_span) ** self.sat_power
			t_span = t_span * (1-self.t_eps) + self.t_eps

			sol = []
			vector=[]
			current_step_idx = 0 # Use 0-indexed steps for t_span

			if self.prior_mean == 'zero':
				x = torch.randn_like(MIX) * self.prior_sigma 
			elif self.prior_mean == 'mix':
				x = MIX + torch.randn_like(MIX) * self.prior_sigma

			while current_step_idx < num_solver_steps:
				t_current = t_span[current_step_idx].item()
				# Ensure t_tensor is a tensor of shape (batch_size,) on the correct device
				t_tensor = torch.full((x.shape[0],), t_current, device=MIX.device)
				
				dphi_dt = self.forward_score(x=x, ref=REF, mix_condition=MIX, t=t_tensor)
				
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

		return x_hat

	# def extract(self, mix, ref, timeit=False,
	# 	return_stft=False,
	# 	num_solver_steps: int = 30, # Added num_solver_steps parameter
	# 	**kwargs
	# ):

	# 	"""
	# 	One-call speech enhancement of noisy speech `y`, for convenience.

	# 	mix: noisy speech (T)
	# 	ref: reference speech (T) 

	# 	normalization -> padding -> STFT -> forward_transform -> 
	# 	"""

	# 	T_orig = mix.size(0)

	# 	norm_factor = mix.abs().max().item()
	# 	mix = mix / norm_factor
	# 	ref = ref / norm_factor

	# 	MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0) # MIX: (B, F, T) = (1, F, T)		
	# 	REF = torch.unsqueeze(self._forward_transform(self._stft(ref.cuda())), 0) # REF: (B, F, T') = (1, F, T')

	# 	with torch.no_grad():
	# 		# t_span = torch.linspace(0, 1, num_solver_steps + 1, device=MIX.device) # Use num_solver_steps and MIX.device
	# 		# t_span = 1 - (1 - t_span) ** self.sat_power

	# 		sol = []
	# 		vector=[]
	# 		current_step_idx = 0 # Use 0-indexed steps for t_span

	# 		# x = torch.randn_like(MIX)
	# 		x = MIX + torch.randn_like(MIX) * self.prior_sigma
	# 		t_current = self.t_eps

	# 		while current_step_idx < num_solver_steps:
	# 			if current_step_idx != 0:
	# 				t_current = 1
	# 			# Ensure t_tensor is a tensor of shape (batch_size,) on the correct device
	# 			t_tensor = torch.full((x.shape[0],), t_current, device=MIX.device)
				
	# 			dphi_dt = self.forward_score(x=x, ref=REF, mix_condition=MIX, t=t_tensor)
				
	# 			x = x + dphi_dt
	# 			# t = t_span[current_step_idx + 1].item() # t for the next iteration's start

	# 			sol.append(x)
	# 			vector.append(dphi_dt)
	# 			current_step_idx += 1

	# 		sample = sol[-1] if sol else x # Handle case where num_solver_steps might be 0

	# 		if return_stft:
	# 			pass

	# 	x_hat = self.to_audio(sample.squeeze(), T_orig)
	# 	x_hat = x_hat * norm_factor
	# 	x_hat = x_hat.squeeze().cpu()

	# 	return x_hat

class StochasticRegenerationModel_frommix(StochasticRegenerationModel):
	def _step(self, batch, batch_idx):
		# mix_stft, clean_stft, regi_stft: complex spectrograms with shape (B, F, T) or (B, F, T')
		mix_stft, clean_stft, regi_stft = batch[0], batch[1], batch[2]

		# CFM part
		t1 = torch.rand([mix_stft.shape[0]], device=mix_stft.device) # *(1-self.t_eps) + self.t_eps
		t1 = t1 * (1-self.t_eps) + self.t_eps

		# print("t1 shape:", t1.shape)
		t1 = t1.unsqueeze(-1).unsqueeze(-1)  # Add two

		z = mix_stft

		y1 = (1 - (1 - self.sigma_min) * t1) * z + t1 * clean_stft 		# mix to x
		vec = clean_stft - (1 - self.sigma_min) * z

		# shape of y1: (B, F, T)
		# (B,D,F,T) in original FlowAVSE code, D=1 if single channel, D=2 if stereo. In my code, assume always single channel

		# shape of t1: (B, 1, 1) -> (B,)
		t1 = t1.squeeze(-1).squeeze(-1)  # Remove only the last two dimensions, keeping batch dimension

		score_1 = self.forward_score(x = y1, ref = regi_stft, t=t1, mix_condition=mix_stft)
	
		loss = self._loss(score_1, vec)
		# print(f"score_1 shape: {score_1.shape}, vec shape: {vec.shape}")
		return loss

	def validation_step(self, batch, batch_idx, sr=16000):
		loss = self._step(batch, batch_idx)

		self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

		# Evaluate speech enhancement performance
		if batch_idx == 0 and self.num_eval_files != 0:
			num_eval_files = self.num_eval_files

			# pesq_est, si_sdr_est, estoi_est, ovrl_est, sig_est, sim_est, spec, audio, y_den = evaluate_model(self, num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)
			pesq_est, si_sdr_est, estoi_est, spec, audio, y_den = evaluate_model(self, num_eval_files, num_solver_steps=self.num_solver_steps, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS)

			print("")
			print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
			print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.2f}")
			print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")

			print('__________________________________________________________________')

			self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)
			self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size, sync_dist=True)

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
	
	def extract(self, mix, ref, timeit=False,
		return_stft=False,
		num_solver_steps: int = 30, # Added num_solver_steps parameter
		**kwargs
	):

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.

		mix: noisy speech (T)
		ref: reference speech (T) 

		normalization -> padding -> STFT -> forward_transform -> 
		"""

		T_orig = mix.size(0)

		norm_factor = mix.abs().max().item()
		mix = mix / norm_factor
		ref = ref / norm_factor

		MIX = torch.unsqueeze(self._forward_transform(self._stft(mix.cuda())), 0) # MIX: (B, F, T) = (1, F, T)		
		REF = torch.unsqueeze(self._forward_transform(self._stft(ref.cuda())), 0) # REF: (B, F, T') = (1, F, T')

		with torch.no_grad():
			t_span = torch.linspace(self.t_eps, 1, num_solver_steps + 1, device=MIX.device) # Use num_solver_steps and MIX.device
			t, _, dt = t_span[0].item(), t_span[-1].item(), (t_span[1] - t_span[0]).item() # Ensure t and dt are scalars initially
			sol = []
			vector=[]
			current_step_idx = 0 # Use 0-indexed steps for t_span

			# x = torch.randn_like(MIX)
			x = MIX

			while current_step_idx < num_solver_steps:
				t_current = t_span[current_step_idx].item()
				# Ensure t_tensor is a tensor of shape (batch_size,) on the correct device
				t_tensor = torch.full((x.shape[0],), t_current, device=MIX.device)
				
				dphi_dt = self.forward_score(x=x, ref=REF, t=t_tensor, mix_condition=MIX)
				
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

		return x_hat