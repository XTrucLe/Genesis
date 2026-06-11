import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, block_size: int = 2048, base: int = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.block_size = block_size
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(block_size)

    def _build_cache(self, seq_len: int):
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self.block_size = seq_len

    def _get_cos_sin(
        self, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.block_size:
            self._build_cache(offset + seq_len)
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        return cos.view(1, seq_len, 1, self.dim // 2), sin.view(
            1, seq_len, 1, self.dim // 2
        )

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.size(1)
        cos, sin = self._get_cos_sin(T, offset)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        block_size: int,
        lora_rank: int = 512,
        rope_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.kv_lora_rank = lora_rank
        self.rope_dim = rope_dim
        self.nope_dim = self.head_dim

        self.q_proj = nn.Linear(
            dim, heads * (self.nope_dim + self.rope_dim), bias=False
        )
        self.kv_down_proj = nn.Linear(dim, self.kv_lora_rank, bias=False)
        self.k_rope_proj = nn.Linear(dim, self.rope_dim, bias=False)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank, heads * (self.nope_dim + self.nope_dim), bias=False
        )

        self.softmax_scale = (self.nope_dim + self.rope_dim) ** -0.5
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.rope_dim, block_size)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, C = x.shape

        q_nope, q_rope = (
            self.q_proj(x)
            .view(B, T, self.heads, self.nope_dim + self.rope_dim)
            .split([self.nope_dim, self.rope_dim], dim=-1)
        )

        k_nope, v = (
            self.kv_up_proj(self.kv_down_proj(x))
            .view(B, T, self.heads, self.head_dim + self.head_dim)
            .split([self.head_dim, self.head_dim], dim=-1)
        )

        k_rope = self.k_rope_proj(x).view(B, T, 1, self.rope_dim)
        q_rope, k_rope = self.rope(q_rope, k_rope, offset=offset)
        k_rope = k_rope.expand(B, T, self.heads, self.rope_dim)

        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.drop.p if self.training else 0.0,
            scale=self.softmax_scale,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(2 * dim * 4 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        gate_branch = self.act(self.w1(x))
        up_branch = self.w2(x)
        out = self.w3(gate_branch * up_branch)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        lora_rank: int,
        block_size: int,
        dropout: float = 0.1,
        rope_dim: int = 64,
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(dim)
        self.attn = MultiHeadLatentAttention(
            dim,
            heads,
            block_size=block_size,
            lora_rank=lora_rank,
            dropout=dropout,
            rope_dim=rope_dim,
        )
        self.ln2 = nn.RMSNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
