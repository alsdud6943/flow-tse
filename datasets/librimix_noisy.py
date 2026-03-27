import torch
from torch.utils.data import Dataset
import random
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F

import glob
import os

def _activelev(*args):
	"""
	need to update like matlab
	"""
	res = torch.concat(list(args))
	return torch.max(torch.abs(res))


def unify_energy(*args):
	max_amp = _activelev(*args)
	if max_amp == 0:
		max_amp = 1.0
	mix_scale = 1.0 / max_amp
	return [x * mix_scale for x in args], max_amp


def generate_target_audio(spk1, spk2, regi, noise, snr=5):
	"""
	spk 1: T1
	spk 2: T2
	regi: T3
	noise: T4
	"""
	(spk1, spk2), _ = unify_energy(spk1, spk2)

	# 1) spk1:spk2 SIR setting
	sir_db = (random.random() * 2 - 1) * snr   # U[-snr, snr]
	spk2 = spk2 * (10 ** (-sir_db / 20.0))

	# 2) spk1:noise SNR setting
	snr_db = (random.random() * 2 - 1) * snr   # U[-snr, snr]
	noise = noise * (10 ** (-snr_db / 20.0))

	mix = spk1 + spk2 + noise

	(mix, clean, regi), _ = unify_energy(mix, spk1, regi)
	return (mix, clean, regi)


def truc_wav(*audio: torch.Tensor, length, deterministic=False):
	"""
	Given a list of audio with the same length as arguments, chunk the audio into a given length.
	Note that all the audios will be chunked using the same offset

	Args:
		audio: the list of audios to be chunked, should have the same length with shape [T] (1D)
		length: the length to be chunked into, if length is None, return the original audio
	Returns:
		A list of chuncked audios
	"""
	audio_len = audio[0].size(0)  # [T]
	res = []
	if length == None:
		for a in audio:
			res.append(a)
		return res[0] if len(res) == 1 else res
	if audio_len > length:
		if deterministic == True:
			offset = 0
		else:
			offset = random.randint(0, audio_len - length - 1)
		for a in audio:
			res.append(a[offset : offset + length])
	else:
		for a in audio:
			res.append(F.pad(a, (0, length - a.size(0)), "constant"))
	return res[0] if len(res) == 1 else res

def get_source_list(file_path: str, ret_name=False):
	files = []
	names = []
	with open(file_path, "r") as f:
		for line in f.readlines():
			l = line.replace("\n", "").split(" ")
			name = l[0]
			path = l[-1]
			files.append(path)
			names.append(name)
	if ret_name:
		return names, files
	return files


class TargetDMDataset(Dataset):
	def __init__(
		self,
		scp_path,
		noise_path,
		epoch_num=100000,
		mix_length=32640,  #48080 in TSELM code, should set to 32640 to match num_frames=256. Target_len = (num_frames - 1) * hop_length #(256-1)*128=32640
		# can be set to 48960 for about 3 seconds, 32640 for about 2 seconds
		regi_length=64080,
	):
		"""
		Initialize the Target DM Dataset.
		This class is used for dynamic mixing of target speech extraction dataset

		Args:
			scp_path: the .pt file which saves a dictionary of speker_name -> list of path to source files
			epoch_num: specifcy how many data to be considered as one epoch
			mix_length: the length of the mixing speech and clean speech
			regi_length: the length of the register speech
		"""
		# self.speaker_dict = torch.load(scp_path)
		self.speaker_dict = torch.load(scp_path, weights_only=True)
		self.length = epoch_num
		self.mix_length = mix_length
		self.regi_length = regi_length

		self.noise_path = noise_path
		self.noise_list = glob.glob(os.path.join(noise_path, "*.wav"))

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		keys_list = list(self.speaker_dict.keys())
		speaker_1 = random.choice(keys_list)
		speaker_2 = random.choice(keys_list)
		while speaker_2 == speaker_1:
			speaker_2 = random.choice(keys_list)
		spk1 = random.choice(self.speaker_dict[speaker_1])
		regi = random.choice(self.speaker_dict[speaker_1])
		while regi == spk1:
			regi = random.choice(self.speaker_dict[speaker_1])
		spk2 = random.choice(self.speaker_dict[speaker_2])        
		spk1_audio = torchaudio.load(spk1)[0].squeeze(0)  # [T]
		spk2_audio = torchaudio.load(spk2)[0].squeeze(0)
		regi_audio = torchaudio.load(regi)[0].squeeze(0)

		# randomly select one noise file from the WHAM noise list
		noise_file = random.choice(self.noise_list)
		noise_audio, sr = torchaudio.load(noise_file)
		# Handle both mono and stereo noise files (following LibriMix approach)
		if len(noise_audio.shape) > 1 and noise_audio.shape[0] > 1:
			# If stereo, take the first channel
			noise_audio = noise_audio[0]
		else:
			# If mono, remove channel dimension
			noise_audio = noise_audio.squeeze(0)

		if self.regi_length is not None:
			regi_audio = truc_wav(regi_audio, length=self.regi_length)
		else:
			regi_audio = truc_wav(regi_audio, length=self.mix_length)
		spk1_audio = truc_wav(spk1_audio, length=self.mix_length)
		spk2_audio = truc_wav(spk2_audio, length=self.mix_length)
		noise_audio = truc_wav(noise_audio, length=self.mix_length)

		mix, clean, regi = generate_target_audio(spk1_audio, spk2_audio, regi_audio, noise_audio)
		return mix, clean, regi

class TargetDataset(Dataset):
	def __init__(
		self,
		mix_path: str,
		regi_path: str,
		clean_path: str,
		mix_length=32640,  #48080 in TSELM code, should set to 32640 to match num_frames=256
		regi_length=64080,
	):
		"""
		The regular dataset for target speaker extraction.
		Has to provide three .scp files that have mix_path, regi_path, clean_path aligned
		"""
		self.mix_list = get_source_list(mix_path)
		self.regi_list = get_source_list(regi_path)
		self.clean_list = get_source_list(clean_path)
		self.mix_length = mix_length
		self.regi_length = regi_length
		pass

	def __len__(self):
		return len(self.mix_list)

	def __getitem__(self, idx):
		mix_path = self.mix_list[idx]
		regi_path = self.regi_list[idx]
		clean_path = self.clean_list[idx]
		mix_audio = torchaudio.load(mix_path)[0].squeeze(0)  # [T]
		regi_audio = torchaudio.load(regi_path)[0].squeeze(0)
		clean_audio = torchaudio.load(clean_path)[0].squeeze(0)
		mix_audio, clean_audio = truc_wav(
			mix_audio, clean_audio, length=self.mix_length, deterministic=True
		)
		regi_audio = truc_wav(regi_audio, length=self.regi_length, deterministic=True)
		return mix_audio, clean_audio, regi_audio, mix_path, clean_path, regi_path


############################################################################################


# Helper function for STFT window, similar to data_module_vi.py
def get_window(window_type, window_length):
	if window_type == 'sqrthann':
		return torch.sqrt(torch.hann_window(window_length, periodic=True))
	elif window_type == 'hann':
		return torch.hann_window(window_length, periodic=True)
	else:
		raise NotImplementedError(f"Window type {window_type} not implemented!")


class TargetSpecsDataset(Dataset):
	def __init__(self, underlying_dataset, n_fft, hop_length, window_tensor, spec_transform_func):
		self.underlying_dataset = underlying_dataset
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.window_tensor = window_tensor
		self.spec_transform_func = spec_transform_func

		self.stft_kwargs_template = {
			"n_fft": self.n_fft,
			"hop_length": self.hop_length,
			"center": True,
			"return_complex": True,
		}

	def __len__(self):
		return len(self.underlying_dataset)

	def __getitem__(self, idx: int, return_time: bool = False) -> tuple:
		"""
		Args:
			idx (int): Index of the item to fetch.
			return_time (bool, optional): If True, returns time-domain audio.
												 Defaults to False (returns STFT).
		Returns:
			tuple: If return_time is True, returns time-domain audio signals and passthrough data:
				   (mix_time, clean_time, regi_time, *passthrough_data).
				   Otherwise, returns STFT processed signals and passthrough data:
				   (mix_stft, clean_stft, regi_stft, *passthrough_data).
				   The passthrough_data typically includes file paths if provided by the underlying dataset.
		"""
		raw_data = self.underlying_dataset[idx]

		audio_signals_to_process = raw_data[:3]  # mix, clean, regi audios
		passthrough_data = raw_data[3:]          # paths or other metadata

		if return_time: # Directly use return_time
			# Return only time-domain audio signals and any passthrough data
			output_list = list(audio_signals_to_process)
			output_list.extend(passthrough_data)
			return tuple(output_list)
		else:
			# Process to STFT
			processed_audios_stft = []
			for audio_signal in audio_signals_to_process:
				# Prepare STFT kwargs with window on the correct device
				current_stft_kwargs = {
					**self.stft_kwargs_template,
					"window": self.window_tensor.to(audio_signal.device)
				}
				# manual padding seems unnecessary because of center=True in stft
				# audio_signal = F.pad(audio_signal, (0, self.n_fft - self.hop_length), "constant")
				audio_stft_complex = torch.stft(audio_signal, **current_stft_kwargs)

				audio_stft_processed = self.spec_transform_func(audio_stft_complex)
				processed_audios_stft.append(audio_stft_processed)

			mix_stft, clean_stft, regi_stft = processed_audios_stft

			output_list = [mix_stft, clean_stft, regi_stft]
			output_list.extend(passthrough_data)  # Add paths back
			return tuple(output_list)


class SpecsDataModuleTSE(pl.LightningDataModule):
	def __init__(
		self,
		val_mix_scp=None, val_regi_scp=None, val_clean_scp=None,
		test_mix_scp=None, test_regi_scp=None, test_clean_scp=None,

		noise_path=None,  # Path to the noise source list file

		use_dynamic_mixing_train=True,  # This should be True
		train_speaker_scp_for_dm=None,  # Path to your train_100_360.pt file
		epoch_num_for_dm=100000,

		batch_size=8,
		num_workers=8,
		mix_length=32640, # should set to 32640 for num_frames=256, 49024 for num_frames=384
		regi_length=64080,

		n_fft=510,
		hop_length=128,
		window_type="hann",  # Changed from 'window' to 'window_type' to avoid conflict
		spec_factor=0.15,
		spec_abs_exponent=0.5,

		pin_memory=True,
		return_time=False,
		**kwargs
	):
		super().__init__()
		self.val_mix_scp = val_mix_scp
		self.val_regi_scp = val_regi_scp
		self.val_clean_scp = val_clean_scp
		self.test_mix_scp = test_mix_scp
		self.test_regi_scp = test_regi_scp
		self.test_clean_scp = test_clean_scp

		self.noise_path = noise_path  # Path to the noise source list file

		self.use_dynamic_mixing_train = use_dynamic_mixing_train
		self.train_speaker_scp_for_dm = train_speaker_scp_for_dm
		self.epoch_num_for_dm = epoch_num_for_dm

		self.batch_size = batch_size
		self.num_workers = num_workers
		self.mix_length = mix_length  # Passed to underlying datasets
		self.regi_length = regi_length  # Passed to underlying datasets

		self.n_fft = n_fft
		self.hop_length = hop_length
		self.window_type = window_type
		
		# Generate a base window tensor (typically on CPU). TargetSpecsDataset will move to device if needed.
		self.generated_window_tensor = get_window(self.window_type, self.n_fft)

		self.spec_factor = spec_factor
		self.spec_abs_exponent = spec_abs_exponent

		self.pin_memory = pin_memory
		self.return_time = return_time
		self.kwargs = kwargs

		self.windows_cache = {}  # For device-specific window tensors for utility stft/istft

	def _get_window_for_device(self, device):
		if device not in self.windows_cache:
			self.windows_cache[device] = get_window(self.window_type, self.n_fft).to(device)
		return self.windows_cache[device]

	@property
	def stft_utility_kwargs(self):  # For general utility stft method
		return {"n_fft": self.n_fft, "hop_length": self.hop_length, "center": True, "return_complex": True}

	@property
	def istft_utility_kwargs(self):  # For general utility istft method
		return {"n_fft": self.n_fft, "hop_length": self.hop_length, "center": True}

	def spec_fwd(self, spec):
		if self.spec_abs_exponent != 1:
			e = self.spec_abs_exponent
			spec = spec.abs()**e * torch.exp(1j * spec.angle())
		return spec * self.spec_factor

	def spec_back(self, spec):
		spec = spec / self.spec_factor
		if self.spec_abs_exponent != 1:
			e = self.spec_abs_exponent
			spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
		return spec

	def stft(self, sig):  # Utility STFT
		window = self._get_window_for_device(sig.device)
		return torch.stft(sig, **{**self.stft_utility_kwargs, "window": window})

	def istft(self, spec_complex, length=None):  # Utility iSTFT
		window = self._get_window_for_device(spec_complex.device)
		return torch.istft(spec_complex, **{**self.istft_utility_kwargs, "window": window, "length": length})

	def setup(self, stage=None):
		target_specs_dataset_args = {
			"n_fft": self.n_fft,
			"hop_length": self.hop_length,
			"window_tensor": self.generated_window_tensor,  # Pass the CPU tensor
			"spec_transform_func": self.spec_fwd,
			# "return_time": self.return_time # Removed
		}

		if stage == 'fit' or stage is None:
			if self.use_dynamic_mixing_train:
				if not self.train_speaker_scp_for_dm:
					raise ValueError("train_speaker_scp_for_dm (path to speaker .pt file) must be provided when use_dynamic_mixing_train is enabled.")
				train_audio_set = TargetDMDataset(
					scp_path=self.train_speaker_scp_for_dm,
					noise_path=self.noise_path,
					epoch_num=self.epoch_num_for_dm,
					mix_length=self.mix_length,
					regi_length=self.regi_length
				)
				self.train_set = TargetSpecsDataset(underlying_dataset=train_audio_set, **target_specs_dataset_args)
			else:
				raise ValueError("Training configuration error: --use_dynamic_mixing_train must be specified, and --train_speaker_scp_for_dm must be provided with the path to the training .pt file. The old method of providing separate train_mix_scp, train_regi_scp, train_clean_scp is no longer supported.")

			if all([self.val_mix_scp, self.val_regi_scp, self.val_clean_scp]):
				val_audio_set = TargetDataset(
					mix_path=self.val_mix_scp,
					regi_path=self.val_regi_scp,
					clean_path=self.val_clean_scp,
					mix_length=self.mix_length,
					regi_length=self.regi_length
				)
				self.valid_set = TargetSpecsDataset(underlying_dataset=val_audio_set, **target_specs_dataset_args)
			else:
				self.valid_set = None
				print("Validation scp files not fully provided. Validation set will not be available.")

		if stage == 'test' or stage is None:
			if all([self.test_mix_scp, self.test_regi_scp, self.test_clean_scp]):
				test_audio_set = TargetDataset(
					mix_path=self.test_mix_scp,
					regi_path=self.test_regi_scp,
					clean_path=self.test_clean_scp,
					mix_length=self.mix_length,
					regi_length=self.regi_length
				)
				self.test_set = TargetSpecsDataset(underlying_dataset=test_audio_set, **target_specs_dataset_args)
			else:
				self.test_set = None
				print("Test scp files not fully provided. Test set will not be available.")

	def train_dataloader(self):
		if not hasattr(self, 'train_set') or self.train_set is None:
			raise RuntimeError("Train set not initialized. Call setup(stage='fit') first or check configuration.")
		return DataLoader(
			self.train_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True
		)

	def val_dataloader(self):
		if not hasattr(self, 'valid_set') or self.valid_set is None:
			return None
		return DataLoader(
			self.valid_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False
		)

	def test_dataloader(self):
		if not hasattr(self, 'test_set') or self.test_set is None:
			raise RuntimeError("Test set not initialized. Call setup(stage='test') first or check configuration.")
		return DataLoader(
			self.test_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False
		)

