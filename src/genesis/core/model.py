import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            
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


class Genesis(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        dim:         int   = 1536,
        layers:      int   = 32,
        heads:       int   = 12,
        kv_heads:    int   = 3,
        block_size: int   = 2048,
        dropout:     float = 0.1,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_gc = grad_checkpoint

        self.embedding = nn.Embedding(vocab_size, dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            Block(dim, heads, kv_heads, block_size, dropout) for _ in range(layers)
        ])
        self.ln_f    = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights_all(layers)

    def _init_weights_all(self, layers: int):
        self.apply(self._init_module)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.embedding(x))
        for block in self.blocks:
            if self.use_gc and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.lm_head(self.ln_f(h))

    def num_params(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{train/1e6:.2f}M trainable / {total/1e6:.2f}M total"