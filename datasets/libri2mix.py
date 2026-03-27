import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import pytorch_lightning as pl
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Optional, Union, List
import logging
import random

log = logging.getLogger(__name__)


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


def truc_wav(*audio: torch.Tensor, length):
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
        offset = torch.randint(0, audio_len - length + 1, (1,)).item()
        for a in audio:
            res.append(a[offset : offset + length])
    else:
        for a in audio:
            res.append(F.pad(a, (0, length - a.size(0)), "constant"))
    return res[0] if len(res) == 1 else res


class TargetDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        split: str = "train-100",  # train-100, train-360, dev, test
        n_spkr: int = 2,
        mix_dir: str = "mix_clean",
        cut: str = "min",
        max_n_samples: Optional[int] = None,
        mix_length: Optional[int] = None,
        regi_length: Optional[int] = None,
    ):
        """
        Dataset for loading Libri2Mix dataset directly from disk
        
        Args:
            path: Path to Libri2Mix dataset root
            split: Dataset split (train-100, train-360, dev, test)
            n_spkr: Number of speakers (2 or 3)
            mix_dir: Mix directory name (mix_clean, mix_both, etc.)
            cut: How sources were cut (min or max)
            max_n_samples: Maximum number of samples to load
            mix_length: Target length for mix/clean
            regi_length: Target length for registration
        """
        super().__init__()
        
        self.base_folder = Path(path)
        self.split = split
        self.n_spkr = n_spkr
        self.cut = cut
        self.mix_length = mix_length
        self.regi_length = regi_length
        
        # Validation
        if n_spkr not in [2, 3]:
            raise ValueError(f"Number of speakers must be 2 or 3 (got {n_spkr})")
        if cut not in ["min", "max"]:
            raise ValueError(f"Cut parameter must be 'min' or 'max' (got {cut})")
            
        # Construct paths
        self.dataset_path = (
            self.base_folder / 
            f"Libri{self.n_spkr}Mix/wav16k/{cut}/{split}"
        )
        
        self.mix_path = self.dataset_path / mix_dir
        self.s1_path = self.dataset_path / "s1"
        self.s2_path = self.dataset_path / "s2"
        
        # Verify paths exist
        if not self.mix_path.exists():
            raise FileNotFoundError(f"Mix path not found: {self.mix_path}")
        if not self.s1_path.exists():
            raise FileNotFoundError(f"Speaker 1 path not found: {self.s1_path}")
        if not self.s2_path.exists():
            raise FileNotFoundError(f"Speaker 2 path not found: {self.s2_path}")
        
        # Get file list
        self.file_list = sorted([f for f in os.listdir(self.mix_path) if f.endswith('.wav')])
        
        if max_n_samples is not None:
            self.file_list = self.file_list[:max_n_samples]

        # Create a mapping from speaker ID to a list of their files
        self.speaker_to_files = {}
        for filename in self.file_list:
            # Filename format is spk1_id-chapter-utt-spk2_id-chapter-utt.wav
            parts = filename.replace('.wav', '').split('-')
            spk1_id, spk2_id = parts[0], parts[3]
            
            if spk1_id not in self.speaker_to_files:
                self.speaker_to_files[spk1_id] = []
            self.speaker_to_files[spk1_id].append(filename)
            
            if spk2_id not in self.speaker_to_files:
                self.speaker_to_files[spk2_id] = []
            self.speaker_to_files[spk2_id].append(filename)

        log.info(f"Loaded Libri2Mix dataset: {len(self.file_list)} files from {self.dataset_path}")
        log.info(f"Built speaker-to-file map with {len(self.speaker_to_files)} speakers.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # Load audio files for the main mixture
        mix_audio, _ = torchaudio.load(self.mix_path / filename)
        s1_audio, _ = torchaudio.load(self.s1_path / filename)
        s2_audio, _ = torchaudio.load(self.s2_path / filename)
        
        # Remove channel dimension (assuming mono)
        mix_audio = mix_audio.squeeze(0)  # [T]
        s1_audio = s1_audio.squeeze(0)
        s2_audio = s2_audio.squeeze(0)
        
        # Randomly choose target speaker (1 or 2)
        target_speaker = random.randint(1, 2)
        
        filename_parts = filename.replace('.wav', '').split('-')
        spk1_id = filename_parts[0]
        spk2_id = filename_parts[3]

        if target_speaker == 1:
            clean_audio = s1_audio
            target_spk_id = spk1_id
        else:  # target_speaker == 2
            clean_audio = s2_audio
            target_spk_id = spk2_id

        # Get registration audio from a different utterance of the same speaker
        possible_regi_files = self.speaker_to_files[target_spk_id]

        # Choose a different file for registration, ensuring it's not the same as the current file
        regi_filename = filename
        # In the rare case a speaker has only one utterance in the dataset split
        if len(possible_regi_files) > 1:
            while regi_filename == filename:
                regi_filename = random.choice(possible_regi_files)

        # Load the registration file and determine which source (s1 or s2) corresponds to the target speaker
        regi_filename_parts = regi_filename.replace('.wav', '').split('-')
        regi_spk1_id = regi_filename_parts[0]
        
        if regi_spk1_id == target_spk_id:
            # Target speaker is s1 in the registration file
            regi_audio, _ = torchaudio.load(self.s1_path / regi_filename)
        else:
            # Target speaker is s2 in the registration file
            regi_audio, _ = torchaudio.load(self.s2_path / regi_filename)
            
        regi_audio = regi_audio.squeeze(0)
            
        # Unify energy
        (mix_audio, clean_audio, regi_audio), _ = unify_energy(mix_audio, clean_audio, regi_audio)
        
        # Apply compatibility lengths if specified
        if self.mix_length is not None:
            mix_audio, clean_audio = truc_wav(mix_audio, clean_audio, length=self.mix_length)
            
        if self.regi_length is not None:
            regi_audio = truc_wav(regi_audio, length=self.regi_length)
        elif self.mix_length is not None:
            # If regi_length not specified but mix_length is, use mix_length for regi too
            regi_audio = truc_wav(regi_audio, length=self.mix_length)
            
        return mix_audio, clean_audio, regi_audio


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
    """
    DataModule for Libri2Mix dataset that loads directly from dataset files
    """
    
    def __init__(
        self,
        train_dirs: Union[str, List[str]] = "train-100",  # Can be single dir or list for multiple splits
        val_split: str = "dev",
        test_split: str = "test",
        batch_size: int = 8,
        n_fft: int = 510,
        hop_length: int = 128,
        num_frames: int = 256,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        dataset_path: str = "/path/to/Libri2Mix",
        num_workers: int = 4,
        persistent_workers: bool = True,
        mix_dir: str = "mix_clean",
        cut: str = "max",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Convert single train dir to list for consistency
        if isinstance(train_dirs, str):
            self.train_dirs = [train_dirs]
        else:
            self.train_dirs = train_dirs
            
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        
        # Audio processing params
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window_type = "hann"
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        
        # Dataset params
        self.dataset_path = dataset_path
        self.mix_dir = mix_dir
        self.cut = cut
        
        # Calculate derived parameters
        self.win_length = self.n_fft
        self.stft_length = (self.num_frames - 1) * self.hop_length + self.win_length
        self.generated_window_tensor = get_window(self.window_type, self.n_fft)
        self.windows_cache = {}
        
        log.info(f"SpecsDataModuleLibri2Mix initialized:")
        log.info(f"  Train splits: {self.train_dirs}")
        log.info(f"  Val split: {self.val_split}")
        log.info(f"  Test split: {self.test_split}")
        log.info(f"  STFT params: n_fft={self.n_fft}, hop_length={self.hop_length}")
        log.info(f"  Spectrogram: factor={self.spec_factor}, exponent={self.spec_abs_exponent}")
        log.info(f"  Target length: {self.stft_length} samples ({self.stft_length/16000:.2f}s)")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        
        if stage == "fit" or stage is None:
            # Create training datasets
            train_datasets = []
            for split in self.train_dirs:
                dataset = TargetDataset(
                    path=self.dataset_path,
                    split=split,
                    mix_dir=self.mix_dir,
                    cut=self.cut,
                    mix_length=self.stft_length,
                    regi_length=self.stft_length,
                )
                train_datasets.append(dataset)
            
            # Concatenate multiple training splits if provided
            if len(train_datasets) == 1:
                self.train_dataset = train_datasets[0]
            else:
                self.train_dataset = ConcatDataset(train_datasets)
                log.info(f"Combined training dataset with {len(self.train_dataset)} total samples")
            
            # Validation dataset
            val_audio_set = TargetDataset(
                path=self.dataset_path,
                split=self.val_split,
                mix_dir=self.mix_dir,
                cut=self.cut,
                mix_length=self.stft_length,
                regi_length=self.stft_length,
            )
            self.val_dataset = TargetSpecsDataset(
                underlying_dataset=val_audio_set,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_tensor=self.generated_window_tensor,
                spec_transform_func=self.spec_fwd,
            )

        if stage == "test" or stage is None:
            # Test dataset
            test_audio_set = TargetDataset(
                path=self.dataset_path,
                split=self.test_split,
                mix_dir=self.mix_dir,
                cut=self.cut,
                mix_length=self.stft_length,
                regi_length=self.stft_length,
            )
            self.test_dataset = TargetSpecsDataset(
                underlying_dataset=test_audio_set,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_tensor=self.generated_window_tensor,
                spec_transform_func=self.spec_fwd,
            )

    def _get_window_for_device(self, device: torch.device):
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

    def _collate_fn(self, batch):
        """Custom collate function for batch processing"""
        mix_batch, clean_batch, regi_batch = zip(*batch)
        
        # Stack into batches
        mix_batch = torch.stack(mix_batch)
        clean_batch = torch.stack(clean_batch)
        regi_batch = torch.stack(regi_batch)
        
        # Convert to spectrograms
        with torch.no_grad():
            # Compute STFT for all signals
            mix_stft = self.stft(mix_batch)
            clean_stft = self.stft(clean_batch)
            regi_stft = self.stft(regi_batch)
            
            # Apply spec transform
            mix_spec = self.spec_fwd(mix_stft)
            clean_spec = self.spec_fwd(clean_stft)
            regi_spec = self.spec_fwd(regi_stft)
            
        return {
            'mix': mix_spec,
            'clean': clean_spec,
            'regi': regi_spec,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )


