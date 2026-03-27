import torch
from torch.utils.data import Dataset
import random
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torchaudio.compliance.kaldi as kaldi
import wespeakerruntime as wespeaker

def _activelev(*args):
    """
    need to update like matlab
    """
    res = torch.concat(list(args))
    return torch.max(torch.abs(res))


def unify_energy(*args):
    max_amp = _activelev(*args)
    mix_scale = 1.0 / max_amp
    return [x * mix_scale for x in args]


def generate_target_audio(spk1, spk2, regi, snr=5):
    """
    spk 1: T1
    spk 2: T2
    regi: T3
    """
    spk1, spk2 = unify_energy(spk1, spk2)
    snr_1 = random.random() * snr / 2
    snr_2 = -snr_1
    spk1 = spk1 * 10 ** (snr_1 / 20)
    spk2 = spk2 * 10 ** (snr_2 / 20)
    mix = spk1 + spk2
    mix, clean, regi = unify_energy(mix, spk1, regi)
    return (mix, clean, regi)

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

def compute_fbank(waveform, sample_rate,
                    resample_rate: int = 16000,
                    num_mel_bins: int = 80,
                    frame_length: int = 25,
                    frame_shift: int = 10,
                    dither: float = 0.0,
                    cmn: bool = True):
    """ Extract fbank, simlilar to the one in wespeaker.dataset.processor,
        While integrating the wave reading and CMN.
    """
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        sample_frequency=resample_rate,
                        window_type='hamming',
                        use_energy=False) #Error here
    mat = mat.numpy()
    if cmn:
        # CMN, without CVN
        mat = mat - np.mean(mat, axis=0)
    return mat

# def compute_fbank(waveform, sample_rate, resample_rate=16000, num_mel_bins=80, 
#                   frame_length=25.0, frame_shift=10.0, dither=0.0, cmn=True):
#     """
#     Extract fbank, similar to the one in wespeaker.dataset.processor,
#     While integrating the wave reading and CMN.
#     """
#     # Ensure input is properly shaped
#     if waveform.dim() == 1:
#         waveform = waveform.unsqueeze(0)
    
#     # Check for NaN or inf in input
#     if torch.isnan(waveform).any() or torch.isinf(waveform).any():
#         print("Warning: NaN or inf in input waveform, cleaning...")
#         waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    
#     # Resample if needed
#     if sample_rate != resample_rate:
#         waveform = torchaudio.transforms.Resample(
#             orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    
#     # Robust scaling to avoid overflow
#     max_val = torch.max(torch.abs(waveform))
#     if max_val > 0:
#         # Normalize to [-1, 1] range, then scale more conservatively
#         waveform = waveform / max_val
#         waveform = waveform * 16384.0  # Use 2^14 instead of 2^15 for more stability
#     else:
#         waveform = waveform * 16384.0
    
#     # Ensure no overflow after scaling
#     waveform = torch.clamp(waveform, -32767, 32767)
    
#     try:
#         mat = kaldi.fbank(waveform,
#                             num_mel_bins=num_mel_bins,
#                             frame_length=frame_length,
#                             frame_shift=frame_shift,
#                             dither=dither,
#                             sample_frequency=resample_rate,
#                             window_type='hamming',
#                             use_energy=False)
#         mat = mat.numpy()
        
#         # Check for NaN/inf and handle them
#         if np.isnan(mat).any() or np.isinf(mat).any():
#             print("Warning: NaN or inf in fbank features, cleaning...")
#             mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        
#     except Exception as e:
#         print(f"Error in fbank computation: {e}, using fallback")
#         # Fallback to torchaudio mel spectrogram
#         mel_transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=resample_rate,
#             n_fft=512,
#             win_length=int(frame_length * resample_rate / 1000),
#             hop_length=int(frame_shift * resample_rate / 1000),
#             n_mels=num_mel_bins,
#             window_fn=torch.hann_window
#         )
#         mel_spec = mel_transform(waveform.squeeze(0))
#         mat = torch.log(mel_spec + 1e-8).T.numpy()
        
#         # Clean any NaN/inf from fallback method too
#         if np.isnan(mat).any() or np.isinf(mat).any():
#             mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    
#     if cmn:
#         # CMN (Cepstral Mean Normalization)
#         mat_mean = np.mean(mat, axis=0)
#         if not np.isnan(mat_mean).any():
#             mat = mat - mat_mean
        
#         # Final check for NaN/inf after CMN
#         if np.isnan(mat).any() or np.isinf(mat).any():
#             mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    
#     return mat

def extract_spk_emb(spk_emb_extractor, waveform, sample_rate, wav_num=1):

    feats = compute_fbank(waveform, sample_rate)
    feats = np.expand_dims(feats, 0)
    spk_emb = spk_emb_extractor.extract_embedding_feat(feats)
    return spk_emb


class TargetDMDataset(Dataset):
    def __init__(
        self,
        scp_path,
        epoch_num=100000,
        mix_length=32640,  #48080 in TSELM code, should set to 32640 to match num_frames=256. Target_len = (num_frames - 1) * hop_length #(256-1)*128=32640
        # can be set to 48960 for about 3 seconds, 32640 for about 2 seconds
        regi_length=64080,
        extract_speaker_embedding=True,
    ):
        """
        Initialize the Target DM Dataset for waveform-based reference.
        This class is used for dynamic mixing of target speech extraction dataset
        and extracts speaker embeddings directly in the dataset.

        Args:
            scp_path: the .pt file which saves a dictionary of speker_name -> list of path to source files
            epoch_num: specifcy how many data to be considered as one epoch
            mix_length: the length of the mixing speech and clean speech
            regi_length: the length of the register speech
            extract_speaker_embedding: whether to extract speaker embedding from reference waveform
        """
        self.speaker_dict = torch.load(scp_path, weights_only=True)
        self.length = epoch_num
        self.mix_length = mix_length
        self.regi_length = regi_length
        self.extract_speaker_embedding = extract_speaker_embedding
        
        # Initialize wespeaker model for speaker embedding extraction
        if self.extract_speaker_embedding:
            self.spk_emb_extractor = wespeaker.Speaker(lang='en')

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
        
        spk1_audio, sr1 = torchaudio.load(spk1)
        spk1_audio = spk1_audio.squeeze(0)  # [T]
        spk2_audio, sr2 = torchaudio.load(spk2)
        spk2_audio = spk2_audio.squeeze(0)
        regi_audio, sr_regi = torchaudio.load(regi)
        regi_audio = regi_audio.squeeze(0)
        
        if self.regi_length is not None:
            regi_audio = truc_wav(regi_audio, length=self.regi_length)
        else:
            regi_audio = truc_wav(regi_audio, length=self.mix_length)
        spk1_audio = truc_wav(spk1_audio, length=self.mix_length)
        spk2_audio = truc_wav(spk2_audio, length=self.mix_length)
        
        mix, clean, regi = generate_target_audio(spk1_audio, spk2_audio, regi_audio)
        
        if self.extract_speaker_embedding:

            spk_emb = extract_spk_emb(self.spk_emb_extractor, regi_audio.unsqueeze(0), sr_regi)
            # spk_emb is already a numpy array, just convert to tensor
            spk_emb = torch.tensor(spk_emb)  # Convert to tensor

            return mix, clean, spk_emb, "dynamic_mix", "dynamic_mix", "dynamic_mix"
        
        else:
            return mix, clean, regi


class TargetDataset(Dataset):
    def __init__(
        self,
        mix_path: str,
        regi_path: str,
        clean_path: str,
        mix_length=32640,  #48080 in TSELM code, should set to 32640 to match num_frames=256
        regi_length=64080,
        extract_speaker_embedding=True,
    ):
        """
        The regular dataset for target speaker extraction with waveform reference.
        Has to provide three .scp files that have mix_path, regi_path, clean_path aligned
        """
        self.mix_list = get_source_list(mix_path)
        self.regi_list = get_source_list(regi_path)
        self.clean_list = get_source_list(clean_path)
        self.mix_length = mix_length
        self.regi_length = regi_length
        self.extract_speaker_embedding = extract_speaker_embedding
        
        # Initialize wespeaker model for speaker embedding extraction
        if self.extract_speaker_embedding:
            self.spk_emb_extractor = wespeaker.Speaker(lang='en')

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        mix_path = self.mix_list[idx]
        regi_path = self.regi_list[idx]
        clean_path = self.clean_list[idx]
        mix_audio, sr_mix = torchaudio.load(mix_path)
        mix_audio = mix_audio.squeeze(0)  # [T]
        regi_audio, sr_regi = torchaudio.load(regi_path)
        regi_audio = regi_audio.squeeze(0)
        clean_audio, sr_clean = torchaudio.load(clean_path)
        clean_audio = clean_audio.squeeze(0)
        
        mix_audio, clean_audio = truc_wav(
            mix_audio, clean_audio, length=self.mix_length
        )
        regi_audio = truc_wav(regi_audio, length=self.regi_length)
        
        if self.extract_speaker_embedding:
            spk_emb = extract_spk_emb(self.spk_emb_extractor, regi_audio.unsqueeze(0), sr_regi)
            # spk_emb is already a numpy array, just convert to tensor
            spk_emb = torch.tensor(spk_emb)  # Convert to tensor

            return mix_audio, clean_audio, spk_emb, mix_path, clean_path, regi_path
        
        else:
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
    def __init__(self, underlying_dataset, n_fft, hop_length, window_tensor, spec_transform_func, extract_speaker_embedding=True):
        """
        Dataset that processes waveform data into spectrograms for mix and clean,
        but keeps reference as raw waveform for wespeaker compatibility
        """
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
        self.extract_speaker_embedding = extract_speaker_embedding


    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx: int, return_time: bool = False) -> tuple:
        """
        Args:
            idx (int): Index of the item to fetch.
            return_time (bool, optional): If True, returns time-domain audio.
                                                 Defaults to False (returns STFT for mix/clean, waveform for ref).
        Returns:
            tuple: If return_time is True, returns time-domain audio signals:
                   (mix_time, clean_time, regi_time, spk_emb, *passthrough_data).
                   Otherwise, returns STFT processed signals for mix/clean and waveform for ref:
                   (mix_stft, clean_stft, regi_waveform, spk_emb, *passthrough_data).
        """
        raw_data = self.underlying_dataset[idx]

        if self.extract_speaker_embedding:
            mix_audio, clean_audio, spk_emb, *passthrough_data = raw_data
        else:
            mix_audio, clean_audio, regi_audio, *passthrough_data = raw_data
            spk_emb = None

        if return_time:
            if self.extract_speaker_embedding:
                # Return only time-domain audio signals, speaker embedding, and any passthrough data
                output_list = [mix_audio, clean_audio, spk_emb]
                output_list.extend(passthrough_data)
                return tuple(output_list)
            else:
                output_list = [mix_audio, clean_audio, regi_audio]
                output_list.extend(passthrough_data)
                return tuple(output_list)
            
        else: # return spectrograms 
            if self.extract_speaker_embedding:
                # Process mix and clean to STFT, keep regi as waveform
                processed_audios = []
                
                # Process mix and clean to STFT
                for audio_signal in [mix_audio, clean_audio]:
                    # Prepare STFT kwargs with window on the correct device
                    current_stft_kwargs = {
                        **self.stft_kwargs_template,
                        "window": self.window_tensor.to(audio_signal.device)
                    }
                    audio_stft_complex = torch.stft(audio_signal, **current_stft_kwargs)
                    audio_stft_processed = self.spec_transform_func(audio_stft_complex)
                    processed_audios.append(audio_stft_processed)

                mix_stft, clean_stft= processed_audios
                output_list = [mix_stft, clean_stft, spk_emb]
                output_list.extend(passthrough_data)
                return tuple(output_list)
            else:
                processed_audios = []
                
                # Process mix and clean to STFT
                for audio_signal in [mix_audio, clean_audio, regi_audio]:
                    # Prepare STFT kwargs with window on the correct device
                    current_stft_kwargs = {
                        **self.stft_kwargs_template,
                        "window": self.window_tensor.to(audio_signal.device)
                    }
                    audio_stft_complex = torch.stft(audio_signal, **current_stft_kwargs)
                    audio_stft_processed = self.spec_transform_func(audio_stft_complex)
                    processed_audios.append(audio_stft_processed)

                mix_stft, clean_stft, regi_stft = processed_audios

                output_list = [mix_stft, clean_stft, regi_stft]
                output_list.extend(passthrough_data)
                return tuple(output_list)


class SpecsDataModuleTSE(pl.LightningDataModule):
    def __init__(
        self,
        val_mix_scp=None, val_regi_scp=None, val_clean_scp=None,
        test_mix_scp=None, test_regi_scp=None, test_clean_scp=None,

        use_dynamic_mixing_train=True,  # This should be True
        train_speaker_scp_for_dm=None,  # Path to your train_100_360.pt file
        epoch_num_for_dm=100000,
        extract_speaker_embedding=True,  # New parameter to control speaker embedding extraction

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

        self.use_dynamic_mixing_train = use_dynamic_mixing_train
        self.train_speaker_scp_for_dm = train_speaker_scp_for_dm
        self.epoch_num_for_dm = epoch_num_for_dm
        self.extract_speaker_embedding = extract_speaker_embedding

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mix_length = mix_length  # Passed to underlying datasets
        self.regi_length = regi_length  # Passed to underlying datasets

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_type = window_type
        
        # Generate a base window tensor (typically on CPU). TargetSpecsDatasetWaveform will move to device if needed.
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
        }

        if stage == 'fit' or stage is None:
            if self.use_dynamic_mixing_train:
                if not self.train_speaker_scp_for_dm:
                    raise ValueError("train_speaker_scp_for_dm (path to speaker .pt file) must be provided when use_dynamic_mixing_train is enabled.")
                train_audio_set = TargetDMDataset(
                    scp_path=self.train_speaker_scp_for_dm,
                    epoch_num=self.epoch_num_for_dm,
                    mix_length=self.mix_length,
                    regi_length=self.regi_length,
                    extract_speaker_embedding=self.extract_speaker_embedding,
                )
                self.train_set = TargetSpecsDataset(underlying_dataset=train_audio_set, **target_specs_dataset_args)
            else:
                raise ValueError("Training configuration error: --use_dynamic_mixing_train must be specified, and --train_speaker_scp_for_dm must be provided with the path to the training .pt file.")

            if all([self.val_mix_scp, self.val_regi_scp, self.val_clean_scp]):
                val_audio_set = TargetDataset(
                    mix_path=self.val_mix_scp,
                    regi_path=self.val_regi_scp,
                    clean_path=self.val_clean_scp,
                    mix_length=self.mix_length,
                    regi_length=self.regi_length,
                    extract_speaker_embedding=self.extract_speaker_embedding,
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
                    regi_length=self.regi_length,
                    extract_speaker_embedding=self.extract_speaker_embedding,
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
