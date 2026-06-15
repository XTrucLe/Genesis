from dataclasses import dataclass
from typing import Optional

import torch

from genesis.core.ops.kernels.liger_cross_entropy import LigerCrossEntropyFunction


@dataclass
class CrossEntropyOutput:
    loss: torch.Tensor
    z_loss: Optional[torch.Tensor] = None
    token_accuracy: Optional[torch.Tensor] = None
    predicted_tokens: Optional[torch.Tensor] = None


def liger_cross_entropy(
    input,
    target,
    weight=None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
):
    loss, z_loss, token_accuracy, predicted_tokens = LigerCrossEntropyFunction.apply(
        input,
        target,
        weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        return_token_accuracy,
        return_predicted_tokens,
    )

    if not return_z_loss and not return_token_accuracy and not return_predicted_tokens:
        return loss

    return CrossEntropyOutput(
        loss=loss,
        z_loss=z_loss,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )
