import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head
        
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        
    def forward(self, x, xa=None, mask=None):
        q = self.query(x)
        
        # Self-attention or cross-attention
        if xa is None:
            k = self.key(x)
            v = self.value(x)
        else:
            k = self.key(xa)
            v = self.value(xa)
            
        # Split heads
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_head)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Merge heads
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out(out)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None
        
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4),
            nn.GELU(),
            nn.Linear(n_state * 4, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)
        
    def forward(self, x, xa=None, mask=None):
        x = x + self.attn(self.attn_ln(x), mask=mask)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        x = x + self.mlp(self.mlp_ln(x))
        return x