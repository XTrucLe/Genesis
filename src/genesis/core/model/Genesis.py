import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy as lflce
from genesis.core.model.modules import Block

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

        self.dim       = dim
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

    def _fused_loss(self,  hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = F.pad(labels, (0, 1), value=-1)
        shift_labels = labels[..., 1:].contiguous()

        hidden_states = hidden_states.view(-1, self.dim)
        shift_labels = shift_labels.view(-1).to(hidden_states.device)

        return lflce(
            _input=hidden_states, weight=self.lm_head.weight,
            target=shift_labels, ignore_index=-1, reduction="mean",
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.embedding(x))
        for block in self.blocks:
            if self.use_gc and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        
        h = self.ln_f(h)

        loss = self._fused_loss(h, y)
        return loss

    def num_params(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{train/1e6:.2f}M trainable / {total/1e6:.2f}M total"