import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from genesis.core.ops.kernels.liger_swiglu import SiLUMulFunction


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rrms = torch.rsqrt((x_f32 * x_f32).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * rrms * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, block_size: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(block_size)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self._cached_seq_len = seq_len

    def _get_cos_sin(
        self, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        required = offset + seq_len
        if required > self._cached_seq_len:
            self._build_cache(required)
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        return (
            cos.view(1, seq_len, 1, self.dim // 2),
            sin.view(1, seq_len, 1, self.dim // 2),
        )

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(
            -2
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(q.size(1), offset)
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
        layer_idx: int = -1,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"

        self.heads = heads
        self.rope_dim = rope_dim
        self.layer_idx = layer_idx
        self.v_head_dim = dim // heads
        self.k_head_dim = self.v_head_dim
        self.qk_dim = self.k_head_dim + rope_dim
        self.softmax_scale = float(self.qk_dim) ** -0.5
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, heads * self.qk_dim, bias=False)
        self.kv_down_proj = nn.Linear(dim, lora_rank, bias=False)
        self.kv_norm = RMSNorm(lora_rank)
        self.kv_up_proj = nn.Linear(
            lora_rank,
            heads * (self.k_head_dim + self.v_head_dim),
            bias=False,
        )
        self.k_rope_proj = nn.Linear(dim, rope_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(rope_dim, block_size)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        H, Dk, Dv, Dr = self.heads, self.k_head_dim, self.v_head_dim, self.rope_dim

        q_full = self.q_proj(x).view(B, T, H, Dk + Dr)
        q_nope_raw, q_rope_raw = q_full.split([Dk, Dr], dim=-1)
        q_nope = q_nope_raw.permute(0, 2, 1, 3)

        kv = self.kv_up_proj(self.kv_norm(self.kv_down_proj(x))).view(B, T, H, Dk + Dv)
        k_nope_raw, v_raw = kv.split([Dk, Dv], dim=-1)
        k_nope = k_nope_raw.permute(0, 2, 1, 3)
        v_buf = v_raw.permute(0, 2, 1, 3)

        k_rope_shared = self.k_rope_proj(x).view(B, T, 1, Dr)

        q_rope_out, k_rope_out = self.rope(
            q_rope_raw,
            k_rope_shared,
            offset,
        )

        q_rope_out = q_rope_out.permute(0, 2, 1, 3)
        k_rope_out = k_rope_out.permute(0, 2, 1, 3).expand(B, H, T, Dr)

        q_buf = torch.cat([q_nope, q_rope_out], dim=-1)
        k_buf = torch.cat([k_nope, k_rope_out], dim=-1)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k_buf = torch.cat([k_prev, k_buf], dim=2)
            v_buf = torch.cat([v_prev, v_buf], dim=2)
        new_kv_cache = (k_buf, v_buf)

        is_causal = T > 1
        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_buf,
            k_buf,
            v_buf,
            attn_mask=None,
            is_causal=is_causal,
            dropout_p=dp,
            scale=self.softmax_scale,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out), new_kv_cache


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(2 * dim * 4 / 3)
        hidden = (hidden + 63) // 64 * 64

        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swiglu_output = SiLUMulFunction.apply(self.w1(x), self.w2(x))

        return self.drop(self.w3(swiglu_output))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        lora_rank: int,
        block_size: int,
        dropout: float = 0.1,
        rope_dim: int = 64,
        layer_idx: int = -1,
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
            layer_idx=layer_idx,
        )
        self.ln2 = nn.RMSNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv_cache = self.attn(
            self.ln1(x), offset=offset, kv_cache=kv_cache
        )

        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv_cache
