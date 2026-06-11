import math
from torch import nn
import torch
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
        grad_checkpoint: bool = True,
        rope_dim: int = 64,
    ):
        super().__init__()
        self.use_gc = grad_checkpoint

        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(dim, heads, lora_rank, block_size, dropout, rope_dim)
                for _ in range(layers)
            ]
        )
        self.ln_f = nn.RMSNorm(dim)
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

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        B, T = x.shape

        h = self.drop(self.embedding(x))

        for block in self.blocks:
            h = (
                checkpoint(block, h, use_reentrant=False)
                if self.use_gc and self.training
                else block(h)
            )

        logits = F.linear(self.ln_f(h), self.lm_head.weight)
        if y is None:
            return logits, None
        assert y.shape == (B, T)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        return logits, loss

    def num_params(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        train -= self.embedding.weight.numel()
        return f"{train/1e6:.2f}M trainable / {total/1e6:.2f}M total"


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
