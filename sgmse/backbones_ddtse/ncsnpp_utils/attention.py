import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, in_dim, context_dim, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(in_dim, n_heads * head_dim, bias=False)
        self.to_k = nn.Linear(context_dim, n_heads * head_dim, bias=False)
        self.to_v = nn.Linear(context_dim, n_heads * head_dim, bias=False)
        self.to_out = nn.Linear(n_heads * head_dim, in_dim)

    def forward(self, x, context=None):
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.n_heads), (q, k, v)
        )

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.n_heads)
        return self.to_out(out)
