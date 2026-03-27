import os
import tqdm
import torch
import numpy as np
import soundfile as sf
# from torchaudio import load
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from sgmse.util.other import si_sdr
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
import wespeakerruntime as wespeaker

# Add project root to Python path to allow direct imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from sgmse.model import StochasticRegenerationModel
from sgmse.util.other import si_sdr_torch
import matplotlib.pyplot as plt
# from FLOWTSE.datasets.librimix import TargetDataset, TargetSpecsDataset, get_window
import tempfile


def spec_fwd(self, spec):
	if self.spec_abs_exponent != 1:
		e = self.spec_abs_exponent
		spec = spec.abs()**e * torch.exp(1j * spec.angle())
	return spec * self.spec_factor


@hydra.main(config_path="config", config_name="test", version_base=None)
def main(cfg: DictConfig):
	# --- Resolve absolute paths ---
	cfg.ckpt = to_absolute_path(cfg.ckpt)
	cfg.model.data_module.test_mix_scp = to_absolute_path(cfg.model.data_module.test_mix_scp)
	cfg.model.data_module.test_clean_scp = to_absolute_path(cfg.model.data_module.test_clean_scp)
	cfg.model.data_module.test_regi_scp = to_absolute_path(cfg.model.data_module.test_regi_scp)

	# --- Create output directories ---
	hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
	testresult_dir = hydra_cfg.runtime.output_dir
	
	# print(f"Hydra output directory: {hydra_output_dir}") # Hydra output directory: /home/minyeong.jeong/flowSS/FLOWTSE/test_runs/2025-07-30_14-28-01__10steps
	# print(f"Current working directory: {os.getcwd()}") # Current working directory: /home/minyeong.jeong/flowSS/FLOWTSE
	
	# # Since cfg.save_dir is ".", we want to save directly in the Hydra output directory
	# if cfg.save_dir == ".":
	# 	testresult_dir = hydra_output_dir
	# else:
	# 	testresult_dir = os.path.join(hydra_output_dir, cfg.save_dir)
	
	print(f"testresult_dir: {testresult_dir}")
	os.makedirs(testresult_dir, exist_ok=True)
	log_path = os.path.join(testresult_dir, "test_metrics.csv")

	# instantiate the model
	disc_model = hydra.utils.instantiate(cfg.discriminativemodel)
	gen_model = hydra.utils.instantiate(cfg.generativemodel)

	for i, model in enumerate([disc_model, gen_model]):
		ckpt_path = getattr(cfg, "disc_ckpt", None) if i == 0 else getattr(cfg, "gen_ckpt", None)
		if ckpt_path is not None:
			ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
			print(f"Loading pretrained weights from: {ckpt_path}")
			checkpoint = torch.load(ckpt_path, map_location=model.device, weights_only=False)
			model.load_state_dict(checkpoint['state_dict'])
			
			# Load EMA state if available
			if 'ema' in checkpoint:
				model.ema.load_state_dict(checkpoint['ema'])
				print("EMA state loaded from checkpoint")
			else:
				print("Warning: No EMA state found in checkpoint")
			
			ckpt_path = None

	disc_model.cuda()
	disc_model.eval(no_ema=False)  # This activates EMA weights for evaluation
	gen_model.cuda()
	gen_model.eval(no_ema=False)  # This activates EMA weights for evaluation
	print("Model.device", gen_model.device)

	# --- Setup Dataset ---
	print("Setting up dataset...")

	# # Initialize DNSMOS and speaker model
	# dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
	speaker_model = wespeaker.Speaker(lang='en')

	# --- Prepare Log File ---
	with open(log_path, 'w') as f:
		f.write("filename,pesq,si_sdr,estoi\n")

	all_metrics = {'pesq': [], 'si_sdr': [], 'estoi': [], 'sig': [], 'ovrl': [], 'sim': []}

	# Manually call setup to initialize the test_set attribute
	model.data_module.setup('test')
	# model.data_module.setup('fit')

	# --- Main Evaluation Loop ---
	for i in tqdm.tqdm(range(len(model.data_module.test_set)), desc="Evaluating"):
		# Get raw audio data, replicating the validation loop's data loading
		# mix_audio, clean_audio, ref_audio, mix_path, clean_path, regi_path = test_dataset.__getitem__(i, return_time=True)
		mix_audio, clean_audio, ref_audio, mix_path, clean_path, regi_path = model.data_module.test_set.__getitem__(i, return_time=True)

		# Precompute filename for logging and temp files
		filename = os.path.basename(mix_path)

		# Move tensors to the model's device
		mix_audio = mix_audio.to(model.device)
		clean_audio = clean_audio.to(model.device)
		ref_audio = ref_audio.to(model.device)

		original_length = mix_audio.shape[-1]

		# --- Padding Logic for U-Net ---
		# The U-Net backbone in this model has 6 downsampling stages (len(ch_mult)-1).
		# This means the spectrogram's time dimension must be divisible by 2**6 = 64
		# to ensure that skip connection tensor sizes match during upsampling.
		# We pad the audio to a length that produces a valid number of frames.
		hop_length = cfg.model.data_module.hop_length
		divisor = 64

		# Calculate the number of frames produced by the STFT.
		# The model uses a centered STFT, so num_frames = audio_length // hop_length + 1
		num_frames = original_length // hop_length + 1

		if num_frames % divisor != 0:
			# Calculate the next multiple of the divisor
			target_frames = (num_frames + divisor - 1) // divisor * divisor # smallest multiple of divisor that is ≥ num_frames
			# Calculate the audio length that will produce `target_frames`
			padded_length = (target_frames - 1) * hop_length
			padding = padded_length - original_length
			# Pad the right side of the audio tensors
			mix_audio = torch.nn.functional.pad(mix_audio, (0, padding))
			ref_audio = torch.nn.functional.pad(ref_audio, (0, padding))
		else:
			padded_length = original_length

		"""
		One-call speech enhancement of noisy speech `y`, for convenience.

		mix: noisy speech (T)
		ref: reference speech (T) 

		normalization -> padding -> STFT -> forward_transform -> 
		"""

		# mix랑 clean이랑 길이 다른거 체크, 수정해야함. 근데 libri2mix min split은 길이 같지 않나?

		with torch.no_grad():
			ensemble_config = cfg.get('ensemble', False)
			if ensemble_config:
				# If ensemble is an integer, use it as the number of ensembles
				# If ensemble is True (boolean), default to 10 ensembles
				num_ensembles = ensemble_config if isinstance(ensemble_config, int) else 10
				print(f"Running ensemble evaluation with {num_ensembles} models...")
				clean_hats = []
				for _ in range(num_ensembles):
					clean_hat = model.extract(mix_audio, ref_audio, num_solver_steps=cfg.num_solver_steps)
					clean_hats.append(clean_hat)
				clean_hat = torch.mean(torch.stack(clean_hats), dim=0)
			else:
				clean_hat = model.extract(mix_audio, ref_audio, num_solver_steps=cfg.num_solver_steps)

		# Crop the enhanced audio back to the original length
		if padded_length != original_length:
			clean_hat = clean_hat[..., :original_length]

		clean_audio_np = clean_audio.cpu().numpy()
		clean_hat_np = clean_hat.cpu().numpy()
		# print(f"shape of clean_audio_np: {clean_audio_np.shape}, clean_hat_np: {clean_hat_np.shape}")
		# shape of clean_audio_np: (166960,), clean_hat_np: (166960,)

		try:
			# --- PESQ, SI-SDR, ESTOI ---
			pesq_score = pesq(16000, clean_audio_np, clean_hat_np, 'wb')
			si_sdr_score = si_sdr_torch(torch.from_numpy(clean_audio_np), torch.from_numpy(clean_hat_np)).item()
			estoi_score = stoi(clean_audio_np, clean_hat_np, 16000, extended=True)

			# --- DNSMOS ---
			if clean_hat.ndim == 1:
				clean_hat = clean_hat.unsqueeze(0)

			dnsmos_scores = dnsmos(clean_hat)
			sig_score = dnsmos_scores[..., 1].item()  # Signal quality
			ovrl_score = dnsmos_scores[..., 3].item()  # Overall quality

			# --- Speaker Similarity ---
			try:
				tmp_hat_path = os.path.join(testresult_dir, f"temp_{filename}_hat.wav")
				sf.write(tmp_hat_path, clean_hat_np, 16000)
				emb1 = speaker_model.extract_embedding(tmp_hat_path).squeeze()
				os.remove(tmp_hat_path)

				tmp_clean_path = os.path.join(testresult_dir, f"temp_{filename}_clean.wav")
				sf.write(tmp_clean_path, clean_audio_np, 16000)
				emb2 = speaker_model.extract_embedding(tmp_clean_path).squeeze()
				os.remove(tmp_clean_path)
		
				sim_score = speaker_model.compute_cosine_score(emb1, emb2)
			except Exception as e:
				print(f"Error computing speaker similarity for {filename}: {e}")
				sim_score = 0.0

			# sig_score = 0.0
			# ovrl_score = 0.0
			# sim_score = 0.0

			filename = os.path.basename(mix_path)
			with open(log_path, 'a') as f:
				f.write(f"{filename},{pesq_score:.3f},{si_sdr_score:.3f},{estoi_score:.3f},{sig_score:.3f},{ovrl_score:.3f},{sim_score:.3f}\n")
				# f.write(f"{filename},{pesq_score:.3f},{si_sdr_score:.3f},{estoi_score:.3f}\n")

			print(f"Processed {filename}: PESQ={pesq_score:.3f}, SI-SDR={si_sdr_score:.3f}, ESTOI={estoi_score:.3f}, DNSMOS Signal Quality={sig_score:.3f}, Overall Quality={ovrl_score:.3f}, Speaker Similarity={sim_score:.3f}")
			all_metrics['pesq'].append(pesq_score)
			all_metrics['si_sdr'].append(si_sdr_score)
			all_metrics['estoi'].append(estoi_score)
			all_metrics['sig'].append(sig_score)
			all_metrics['ovrl'].append(ovrl_score)
			all_metrics['sim'].append(sim_score)
		except Exception as e:
			print(f"Error calculating metrics for {filename}: {e}")

		if i < 50: # Limit to first 50 samples for saving
			# Save the enhanced audio
			sf.write(os.path.join(testresult_dir, filename), clean_hat_np, 16000)


	# --- Print Average Scores ---
	print("\n--- Average Test Results ---")
	avg_pesq = np.mean(all_metrics['pesq'])
	avg_si_sdr = np.mean(all_metrics['si_sdr'])
	avg_estoi = np.mean(all_metrics['estoi'])
	avg_sig = np.mean(all_metrics['sig'])
	avg_ovrl = np.mean(all_metrics['ovrl'])
	avg_sim = np.mean(all_metrics['sim'])

	print(f"PESQ: {avg_pesq:.2f}")
	print(f"SI-SDR: {avg_si_sdr:.2f}")
	print(f"ESTOI: {avg_estoi:.2f}")
	print(f"Signal Quality (DNSMOS): {avg_sig:.2f}")
	print(f"Overall Quality (DNSMOS): {avg_ovrl:.2f}")
	print(f"Speaker Similarity: {avg_sim:.2f}")
	print('____________________________')

	with open(log_path, 'a') as f:
		f.write("\nAverage Scores\n")
		f.write(f"PESQ,{avg_pesq:.3f}\n")
		f.write(f"SI-SDR,{avg_si_sdr:.3f}\n")
		f.write(f"ESTOI,{avg_estoi:.3f}\n")
		f.write(f"Signal Quality (DNSMOS),{avg_sig:.3f}\n")
		f.write(f"Overall Quality (DNSMOS),{avg_ovrl:.3f}\n")
		f.write(f"Speaker Similarity,{avg_sim:.3f}\n")
		f.write('____________________________\n')

	print(f"Testing finished. Enhanced audio saved in '{testresult_dir}'.")
	print(f"Metrics saved in '{log_path}'.")

if __name__ == '__main__':
	main()

# python test.py --config-name=test