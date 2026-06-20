import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from genesis.configs.cfg import CFG
from genesis.core.model.modules import Block


class Genesis(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 1536,
        lora_rank: int = 512,
        layers: int = 32,
        heads: int = 12,
        block_size: int = 2048,
        dropout: float = 0.1,
        grad_checkpoint: bool = False,
        rope_dim: int = 64,
    ):
        super().__init__()
        self.use_gc = grad_checkpoint

        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, lora_rank, block_size, dropout, rope_dim) for _ in range(layers)])
        self.ln_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self._init_weights_all(layers)
        self.lm_head.weight = self.embedding.weight

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

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        offset: int = 0,
    ):
        B, T = x.shape

        h = self.drop(self.embedding(x))
        use_cache = kv_caches is not None or (y is None and not self.training)
        new_kv_caches = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            if self.use_gc and self.training:
                h = checkpoint(lambda inp, b=block: b(inp, offset=offset, kv_cache=None)[0], h, use_reentrant=False)
            else:
                past_cache = kv_caches[i] if (kv_caches is not None) else None
                h, new_cache = block(h, offset=offset, kv_cache=past_cache)

                if use_cache:
                    new_kv_caches.append(new_cache)

        logits = self.lm_head(self.ln_f(h))

        if y is None:
            return logits, new_kv_caches

        assert y.shape == (B, T)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return logits, loss

    def num_params(self) -> str:
        fmt = lambda n: f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.2f}M"

        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad) - self.embedding.weight.numel()

        return f"{fmt(train)} trainable / {fmt(total)} total"


if __name__ == "__main__":
    model = Genesis(
        vocab_size=CFG["vocab_size"],
        dim=CFG["dim"],
        layers=CFG["layers"],
        heads=CFG["heads"],
        block_size=CFG["block_size"],
        dropout=CFG["dropout"],
        grad_checkpoint=CFG["grad_checkpoint"],
    )
    print(model.num_params())
