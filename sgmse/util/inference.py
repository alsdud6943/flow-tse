import sre_compile
import torch
from sgmse.util.other import si_sdr, pad_spec
from pesq import pesq
from tqdm import tqdm
from pystoi import stoi
import numpy as np
# from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
# import wespeakerruntime as wespeaker
import torchaudio
import os

# Settings
snr = 0.5
N = 50
corrector_steps = 1

# Plotting settings
MAX_VIS_SAMPLES = 10
n_fft = 512
hop_length = 128

def evaluate_model(model, num_eval_files, num_solver_steps, spec=False, audio=False, mode='standard'):
    if num_eval_files >50:
        audio=False
        spec = False

    model.eval()
    _pesq, _si_sdr, _estoi, _ovrl, _sig, _sim = 0., 0., 0., 0., 0., 0.
    _pesq_den, _si_sdr_den, _estoi_den = 0., 0., 0. # used for denoiser
    
    # Initialize DNSMOS and speaker model
    # dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
    # speaker_model = wespeaker.Speaker(lang='en')

    if spec:
        noisy_spec_list, estimate_spec_list, clean_spec_list = [], [], []
    if audio:
        noisy_audio_list, estimate_audio_list, clean_audio_list = [], [], []
    
    for i in range(num_eval_files):
        # Load wavs
        mix_audio, clean_audio, regi_audio, *_ = model.data_module.valid_set.__getitem__(i, return_time=True) # (T,)

        if mode == 'standard':
            clean_hat = model.extract(mix=mix_audio, ref=regi_audio, num_solver_steps=num_solver_steps)
        elif mode == 'direct':
            clean_hat = model.extract_direct(mix=mix_audio, ref=regi_audio, num_solver_steps=num_solver_steps)
            
        clean_audio_np = clean_audio.cpu().numpy()
        clean_hat_np = clean_hat.cpu().numpy()

        _si_sdr += si_sdr(clean_audio_np, clean_hat_np)
        _pesq += pesq(16000, clean_audio_np, clean_hat_np, 'wb') 
        _estoi += stoi(clean_audio_np, clean_hat_np, 16000, extended=True)
        
        # # Compute DNSMOS scores
        # if clean_hat.ndim == 1:
        #     clean_hat = clean_hat.unsqueeze(0)
        
        # dnsmos_scores = dnsmos(clean_hat)
        # _sig += dnsmos_scores[..., 1].item()  # Signal quality
        # _ovrl += dnsmos_scores[..., 3].item()  # Overall quality

        # # Compute speaker similarity
        # # Ensure audio is in correct format for wespeaker (16kHz, float32)
        # clean_hat_for_speaker = clean_hat_np.astype(np.float32)
        # clean_audio_for_speaker = clean_audio_np.astype(np.float32)
        
        # emb1 = speaker_model.extract_embedding(clean_hat_for_speaker)
        # emb2 = speaker_model.extract_embedding(clean_audio_for_speaker)
        # score = speaker_model.compute_cosine_score(emb1, emb2)
        # _sim += score

        if spec and i < MAX_VIS_SAMPLES:
            mix_stft, clean_hat_stft, clean_audio_stft= model._stft(mix_audio), model._stft(clean_hat), model._stft(clean_audio)
            noisy_spec_list.append(mix_stft)
            estimate_spec_list.append(clean_hat_stft)
            clean_spec_list.append(clean_audio_stft)

        if audio and i < MAX_VIS_SAMPLES:
            noisy_audio_list.append(mix_audio)
            estimate_audio_list.append(clean_hat)
            clean_audio_list.append(clean_audio)

    if spec:
        if audio:
            return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], [noisy_audio_list, estimate_audio_list, clean_audio_list], [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
        else:
            return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], None,  [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
    else: # not spec
        if audio:
            return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, None, [noisy_audio_list, estimate_audio_list, clean_audio_list],  [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
        else: # not audio
            return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, None, None,  [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
    
    # if spec:
    #     if audio:
    #         return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ovrl/num_eval_files, _sig/num_eval_files, _sim/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], [noisy_audio_list, estimate_audio_list, clean_audio_list], [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
    #     else:
    #         return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ovrl/num_eval_files, _sig/num_eval_files, _sim/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], None,  [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
    # else: # not spec
    #     if audio:
    #         return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ovrl/num_eval_files, _sig/num_eval_files, _sim/num_eval_files, None, [noisy_audio_list, estimate_audio_list, clean_audio_list],  [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
    #     else: # not audio
    #         return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _ovrl/num_eval_files, _sig/num_eval_files, _sim/num_eval_files, None, None,  [_pesq_den/num_eval_files, _si_sdr_den/num_eval_files, _estoi_den/num_eval_files ]
    

    '''
    for i in range(num_eval_files):
        # Load wavs

        # Audio-visual code
        x, y, visualFeatures= model.data_module.valid_set.__getitem__(i, raw=True) #d,t
        visualFeatures = torch.Tensor(visualFeatures).cuda()

        # Audio only code
        #x, y= model.data_module.valid_set.__getitem__(i, raw=True) #d,t test set으로 바꿈 원래는 valid set이었음
        
        # Do not change below code.
        norm_factor = y.abs().max().item()

        # Audio-visual code
        clean_hat, yden = model.enhance(torch.Tensor(y).cuda(), context = visualFeatures) #visualFeatures)

        # Audio only code
        #clean_hat = model.enhance(torch.tensor(y).clone().detach(), context= None) #.clone().detach()
        
        if clean_hat.ndim == 1:
            clean_hat = clean_hat.unsqueeze(0)
            yden = yden.unsqueeze(0)
            
        if x.ndim == 1:
            x = x.unsqueeze(0).cpu().numpy()
            clean_hat = clean_hat.unsqueeze(0).cpu().numpy()
            y = y.unsqueeze(0).cpu().numpy()
            yden = yden.unsqueeze(0).cpu().numpy()
        else: #eval only first channel
            x = x[0].unsqueeze(0).cpu().numpy()
            clean_hat = clean_hat[0].unsqueeze(0).cpu().numpy()
            yden = yden[0].unsqueeze(0).cpu().numpy()
            y = y[0].unsqueeze(0).cpu().numpy()

        _si_sdr += si_sdr(x[0], clean_hat[0])
        _pesq += pesq(16000, x[0], clean_hat[0], 'wb') 
        _estoi += stoi(x[0], clean_hat[0], 16000, extended=True)

        _si_sdr_den += si_sdr(x[0], yden[0])
        _pesq_den += pesq(16000, x[0], yden[0], 'wb') 
        _estoi_den += stoi(x[0], yden[0], 16000, extended=True)
        
        y, clean_hat, x, yden = torch.from_numpy(y), torch.from_numpy(clean_hat), torch.from_numpy(x), torch.from_numpy(yden)
        if spec and i < MAX_VIS_SAMPLES:
            y_stft, x_hat_stft, x_stft, yden_stft = model._stft(y[0]), model._stft(clean_hat[0]), model._stft(x[0]), model._stft(yden[0])
            noisy_spec_list.append(y_stft)
            estimate_spec_list.append(x_hat_stft)
            clean_spec_list.append(x_stft)

        if audio and i < MAX_VIS_SAMPLES:
            noisy_audio_list.append(y[0])
            estimate_audio_list.append(clean_hat[0])
            clean_audio_list.append(x[0])

    if spec:
        if audio:
            return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], [noisy_audio_list, estimate_audio_list, clean_audio_list]
        else:
            return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], None
    elif audio and not spec:
            return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, None, [noisy_audio_list, estimate_audio_list, clean_audio_list]
    else:
        return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, None, None

    '''
