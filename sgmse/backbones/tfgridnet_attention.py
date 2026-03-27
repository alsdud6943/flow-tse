"""
TFGridNet-style cross attention mechanism for time-frequency domain processing.
Adapted from the USEF-TSE TFGridNet implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.nn import init



class TFGridNetCrossAttention(nn.Module):
	"""TFGridNet-style cross attention mechanism for time-frequency domain processing"""
	
	def __init__(
		self,
		emb_dim,
		n_freqs,
		approx_qk_dim, # Changed from input_shape
		n_head=8,
		activation="prelu",
		eps=1e-5,
	):
		super().__init__()
		
		assert emb_dim % n_head == 0
		self.head_dim = emb_dim // n_head
		self.n_head = n_head
		self.emb_dim = emb_dim # Keep for consistency if used elsewhere, though TF_gridnet_attentionblock doesn't store it explicitly
		self.eps = eps # Keep for consistency

		# E: approximate channel dimension for Q, K per head
		E = math.ceil(approx_qk_dim * 1.0 / n_freqs)

		self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * E, 1) # 128 -> 8 * (512/256) is too small, so maybe we need to use n_freqs=256
		self.attn_norm_Q = AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps)

		self.attn_conv_K = nn.Conv2d(emb_dim, n_head * E, 1)
		self.attn_norm_K = AllHeadPReLULayerNormalization4DCF((n_head, E, n_freqs), eps=eps)

		self.attn_conv_V = nn.Conv2d(emb_dim, emb_dim, 1) # emb_dim = n_head * head_dim
		self.attn_norm_V = AllHeadPReLULayerNormalization4DCF((n_head, self.head_dim, n_freqs), eps=eps)
		
		act_layer = nn.PReLU() if activation == "prelu" else nn.ReLU()
		self.attn_concat_proj = nn.Sequential(
			nn.Conv2d(emb_dim, emb_dim, 1),
			act_layer,
			LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
		)
	
	def forward(self, x, ref):
		"""
		TFGridNet cross attention following the original algorithm.
		
		Args:
			x: Query tensor [B, C, T_q, F] (C=emb_dim, F=n_freqs)
			ref: Key/Value tensor [B, C, T_kv, F]
		Returns:
			Output tensor [B, C, T_q, F]
		"""
		B, C_in, T_q, F_n = x.shape # C_in is emb_dim, F_n is n_freqs
		T_kv = ref.shape[2]

		# 1. Compute Q, K, V
		# Q_conv: [B, n_head * E, T_q, F_n]
		# Q: [B, n_head, E, T_q, F_n] (output of AllHeadPReLULayerNormalization4DCF)
		Q = self.attn_norm_Q(self.attn_conv_Q(x))
		
		# K_conv: [B, n_head * E, T_kv, F_n]
		# K: [B, n_head, E, T_kv, F_n]
		K = self.attn_norm_K(self.attn_conv_K(ref))
		
		# V_conv: [B, emb_dim, T_kv, F_n] (emb_dim = n_head * head_dim)
		# V: [B, n_head, head_dim, T_kv, F_n]
		V = self.attn_norm_V(self.attn_conv_V(ref))

		# E_channels for Q,K from norm layer's perspective
		E_q_k = Q.shape[2] # This is E from __init__
		
		# 2. Reshape for attention computation (flatten heads into batch dim)
		# Q: [B*n_head, E, T_q, F_n]
		Q = Q.contiguous().view(B * self.n_head, E_q_k, T_q, F_n)
		# K: [B*n_head, E, T_kv, F_n]
		K = K.contiguous().view(B * self.n_head, E_q_k, T_kv, F_n)
		# V: [B*n_head, head_dim, T_kv, F_n]
		V = V.contiguous().view(B * self.n_head, self.head_dim, T_kv, F_n)

		# 3. Prepare Q, K, V for matmul (as in TF_gridnet_attentionblock)
		# Q: [B*n_head, T_q, E*F_n]
		Q = Q.transpose(1, 2).flatten(start_dim=2)
		
		# K: [B*n_head, E*F_n, T_kv]
		K = K.transpose(2, 3).contiguous().view(B * self.n_head, -1, T_kv)
		
		# V: [B*n_head, T_kv, head_dim*F_n]
		# Store shape for later unflattening V_attended
		V_shape_transposed = (B * self.n_head, T_kv, self.head_dim, F_n) 
		V = V.transpose(1, 2).flatten(start_dim=2)
		
		# 4. Attention computation
		attn_emb_dim = Q.shape[-1]  # This is E_q_k * F_n
		# attn_mat: [B*n_head, T_q, T_kv]
		attn_mat = torch.matmul(Q, K) / (attn_emb_dim ** 0.5)
		attn_mat = F.softmax(attn_mat, dim=2)
		
		# 5. Apply attention to V
		# V_attended: [B*n_head, T_q, head_dim*F_n]
		V_attended = torch.matmul(attn_mat, V)
		
		# 6. Reshape V_attended back
		# V_attended: [B*n_head, T_q, head_dim, F_n]
		V_attended = V_attended.reshape(B * self.n_head, T_q, self.head_dim, F_n)
		# V_attended: [B*n_head, head_dim, T_q, F_n]
		V_attended = V_attended.transpose(1, 2)

		# 7. Concatenate heads
		# output: [B, n_head * head_dim, T_q, F_n] = [B, emb_dim, T_q, F_n]
		output = V_attended.contiguous().view(B, self.emb_dim, T_q, F_n)
		
		# 8. Final projection
		output = self.attn_concat_proj(output)
		
		return output


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        # input_dimension is (emb_dim, n_freqs)
        # param_size is [1, C, 1, F]
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [B, C, T, F]
        if x.ndim != 4:
            raise ValueError("LayerNormalization4DCF expects 4D input")
        
        # Normalize over C and F for each T, then apply per-C, per-F affine transform
        # mu_ shape: [B, 1, T, 1]
        # std_ shape: [B, 1, T, 1]
        stat_dim = (1, 3) # Dimensions C and F
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class AllHeadPReLULayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x