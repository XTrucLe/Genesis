import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


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
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_seq_len = seq_len

    def _get_cos_sin(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        required = offset + seq_len
        if required > self._cached_seq_len:
            self._build_cache(max(required, self._cached_seq_len * 2))
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        return (
            cos.view(1, seq_len, 1, self.dim),
            sin.view(1, seq_len, 1, self.dim),
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        return x * cos + self._rotate_half(x) * sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(q.size(1), offset)
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)


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
        q_nope, q_rope = q_full[..., :Dk], q_full[..., Dk:]

        kv_latent = self.kv_norm(self.kv_down_proj(x))
        kv = self.kv_up_proj(kv_latent).view(B, T, H, Dk + Dv)
        k_nope, v = kv[..., :Dk], kv[..., Dk:]

        k_rope_shared = self.k_rope_proj(x).view(B, T, 1, Dr)

        q_rope, k_rope_shared = self.rope(q_rope, k_rope_shared, offset)

        k_rope = k_rope_shared.expand(B, T, H, Dr)

        q_sdpa = torch.cat([q_nope, q_rope], dim=-1).permute(0, 2, 1, 3)
        k_sdpa = torch.cat([k_nope, k_rope], dim=-1).permute(0, 2, 1, 3)
        v_sdpa = v.permute(0, 2, 1, 3)

        if kv_cache is not None and self.training is False:
            latent_prev, k_rope_prev = kv_cache
            kv_latent_cat = torch.cat([latent_prev, kv_latent], dim=1)
            k_rope_cat = torch.cat([k_rope_prev, k_rope_shared], dim=1)
            T_full = kv_latent_cat.shape[1]
            kv_full = self.kv_up_proj(kv_latent_cat).view(B, T_full, H, Dk + Dv)
            k_nope_full, v_sdpa = kv_full[..., :Dk], kv_full[..., Dk:].permute(0, 2, 1, 3)
            k_rope_full = k_rope_cat.expand(B, T_full, H, Dr)
            k_sdpa = torch.cat([k_nope_full, k_rope_full], dim=-1).permute(0, 2, 1, 3)

            new_kv_cache = (kv_latent_cat, k_rope_cat)
        else:
            new_kv_cache = (kv_latent, k_rope_shared)

        is_causal = T > 1
        dp = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
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
        hidden = int(8 * dim / 3)
        hidden = (hidden + 63) // 64 * 64
        self.hidden = hidden

        self.w13 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        swiglu_output = F.silu(gate) * up
        return self.drop(self.w2(swiglu_output))


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
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadLatentAttention(
            dim,
            heads,
            block_size=block_size,
            lora_rank=lora_rank,
            dropout=dropout,
            rope_dim=rope_dim,
            layer_idx=layer_idx,
        )
        self.ln2 = RMSNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv_cache = self.attn(self.ln1(x), offset=offset, kv_cache=kv_cache)

        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv_cache
