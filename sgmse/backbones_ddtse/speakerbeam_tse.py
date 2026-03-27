import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxiliaryNet(nn.Module):
	"""
	Auxiliary neural network to process a reference spectrogram.
	This network uses three BLSTM layers to process the reference speech frame-by-frame 
	and averages the result to produce a speaker embedding.
	"""
	def __init__(self, in_dim, hidden_dim=128, out_dim=256):
		super().__init__()
		self.blstm = nn.LSTM(
			input_size=in_dim,
			hidden_size=hidden_dim,
			num_layers=3,
			bidirectional=True,
			batch_first=True
		)
		self.linear = nn.Linear(hidden_dim * 2, out_dim)


	def forward(self, ref):
		# ref: (B, 2, F, T_ref)
		# Reshape for LSTM: (B, T_ref, 2*F)
		ref = ref.permute(0, 3, 1, 2).contiguous()
		ref = torch.flatten(ref, start_dim=2)

		# BLSTM layers
		blstm_out, _ = self.blstm(ref)

		# Linear projection
		projected_out = self.linear(blstm_out) # (B, T_ref, out_dim)

		# Average pooling over time
		E_spk = torch.mean(projected_out, dim=1) # (B, out_dim)
		return E_spk
	
		# return projected_out # (B, T_ref, out_dim)

class SpeakerbeamFeatureFusion(nn.Module):

	def __init__(self, aux_in_dim, hidden_dim, x_dim, 
			# n_freqs, approx_qk_dim, 
			):
		super().__init__()
		self.auxnet = AuxiliaryNet(in_dim=aux_in_dim, hidden_dim=hidden_dim, out_dim=x_dim)

		# aux_output_dim = in_ch * all_resolutions[-1]
	def forward(self, x, ref):
		# x: (B, C, F_x, T_x) This is the intermediate feature map.
		# ref: (B, 2, F, T_ref) This is the original spectrogram of the reference speech.

		B, C, F_x, T_x = x.shape

		E_spk = self.auxnet(ref) # (B, C)
		E_spk = E_spk.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)

		return x * E_spk