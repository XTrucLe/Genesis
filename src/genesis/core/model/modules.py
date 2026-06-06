import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, block_size: int = 2048):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (25000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        freqs = torch.outer(torch.arange(block_size).float(), inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=True)
        self.register_buffer("sin_cached", emb.sin(), persistent=True)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        T = q.size(1)
        cos = self.cos_cached[:T, None, :]
        sin = self.sin_cached[:T, None, :]

        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        assert heads % kv_heads == 0
        self.heads              = heads
        self.kv_heads           = kv_heads
        self.num_queries_per_kv = heads // kv_heads
        self.head_dim           = dim // heads

        self.q_proj             = nn.Linear(dim, dim, bias=False)
        self.kv_proj             = nn.Linear(dim, 2 * kv_heads * self.head_dim, bias=False)
        self.proj               = nn.Linear(dim, dim, bias=False)
        self.drop               = nn.Dropout(dropout)
        self.rope               = RotaryEmbedding(self.head_dim, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim)
        kv = self.kv_proj(x).view(B, T, 2, self.kv_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_queries_per_kv > 1:

            k = k.unsqueeze(2).expand(B, self.kv_heads, self.num_queries_per_kv, T, self.head_dim)
            k = k.reshape(B, self.heads, T, self.head_dim)
            
            v = v.unsqueeze(2).expand(B, self.kv_heads, self.num_queries_per_kv, T, self.head_dim)
            v = v.reshape(B, self.heads, T, self.head_dim)
            
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True,
            dropout_p = self.drop.p if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden    = ((int(dim * 8 / 3) + 63) // 64) * 64
        self.w1   = nn.Linear(dim, hidden, bias=False)
        self.w2   = nn.Linear(dim, hidden, bias=False)
        self.w3   = nn.Linear(hidden, dim,  bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, heads, kv_heads=kv_heads, block_size=block_size, dropout=dropout)
        self.ln2  = nn.RMSNorm(dim)
        self.ff   = FeedForward(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
