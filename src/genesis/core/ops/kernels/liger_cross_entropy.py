import operator
from typing import Optional

import torch
import triton
import triton.language as tl

from genesis.core.ops.kernels.utils import (
    infer_device,
    is_npu_available,
    compare_version,
    element_mul_kernel,
    is_hip,
)

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    token_accuracy_ptr,
    token_accuracy_stride,
    predicted_tokens_ptr,
    predicted_tokens_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    RETURN_TOKEN_ACCURACY: tl.constexpr,
    RETURN_PREDICTED_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)

        if RETURN_TOKEN_ACCURACY:
            token_accuracy_ptr += program_id * token_accuracy_stride
            tl.store(token_accuracy_ptr, 0.0)
        if RETURN_PREDICTED_TOKENS:
            predicted_tokens_ptr += program_id * predicted_tokens_stride
            tl.store(predicted_tokens_ptr, -1)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride
    if RETURN_TOKEN_ACCURACY:
        token_accuracy_ptr += program_id * token_accuracy_stride
    if RETURN_PREDICTED_TOKENS:
        predicted_tokens_ptr += program_id * predicted_tokens_stride

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    m = float("-inf")
    d = 0.0
    argmax_idx = 0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)

        if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
            is_max_mask = X_block == block_max
            masked_offsets = tl.where(is_max_mask, X_offsets, n_cols)
            current_block_argmax_idx = tl.min(masked_offsets)

            is_new_max = block_max > m
            argmax_idx = tl.where(is_new_max, current_block_argmax_idx, argmax_idx)

        if label_smoothing > 0:
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(
                    tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0)
                )
            else:
                scaled_x_sum += tl.sum(
                    tl.where(X_offsets < n_cols, -eps * X_block, 0.0)
                )
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
            ).cast(tl.float32)
            if HAS_SOFTCAPPING:
                intermediate = tanh(X_block / softcap)
                X_block = softcap * intermediate

            if not HAS_WEIGHT:
                X_block = tl.exp(X_block - m) / d
                X_block += 2 * lse_square_scale * lse * X_block
                X_block += -eps
                X_block = tl.where(
                    X_offsets != y, X_block, X_block - (1 - label_smoothing)
                )
                if reduction == "mean":
                    X_block = X_block / n_non_ignore
            else:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                softmax_X = tl.exp(X_block - m) / d

                dloss_ori = (1 - label_smoothing) * softmax_X
                dloss_ori = tl.where(
                    X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing)
                )
                dloss_ori = dloss_ori * weight_y

                dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
                dz_loss = 2 * lse_square_scale * lse * softmax_X
                if reduction == "mean":
                    dloss_ori = dloss_ori / sum_non_ignore_weight
                    dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    dz_loss = dz_loss / n_non_ignore
                X_block = dloss_ori + dloss_smooth + dz_loss

            if HAS_SOFTCAPPING:
                X_block = X_block * (1 - intermediate * intermediate)

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss

    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    z_loss = lse_square_scale * lse * lse
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        z_loss = z_loss / n_non_ignore
    loss += z_loss

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)
    if RETURN_TOKEN_ACCURACY:
        is_correct = 1.0 if argmax_idx == y else 0.0
        tl.store(token_accuracy_ptr, is_correct)
    if RETURN_PREDICTED_TOKENS:
        tl.store(predicted_tokens_ptr, argmax_idx)


if infer_device() == "xpu":
    MAX_FUSED_SIZE = 4096
elif infer_device() == "npu":
    MAX_FUSED_SIZE = 2048
else:
    MAX_FUSED_SIZE = 65536 // 2


def cross_entropy_forward(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    assert isinstance(
        return_z_loss, bool
    ), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(
        return_token_accuracy, bool
    ), f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    assert isinstance(
        return_predicted_tokens, bool
    ), f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"

    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = (
        torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
        if return_z_loss
        else None
    )
    token_accuracy_1d = (
        torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
        if return_token_accuracy
        else None
    )
    predicted_tokens_1d = (
        torch.full((n_rows,), -1, dtype=torch.int64, device=_input.device)
        if return_predicted_tokens
        else None
    )

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    assert (target * target_mask).max() < _input.shape[
        -1
    ], f"Target {target.max()} is out of bounds. Expected < {_input.shape[-1]}"
    assert (
        target * target_mask
    ).min() >= 0, f"Target {target.min()} is out of bounds. Expected >= 0"
    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0
    if weight is not None:
        assert (
            weight.shape[0] == V
        ), f"If given, weight has to be a Tensor of size V. Got: {weight.shape}"
        assert torch.is_floating_point(
            weight
        ), f"If given, weight has to be a Tensor of floating point dtype. Got: {weight.dtype}"
        sum_non_ignore_weight = (
            torch.gather(weight, dim=0, index=target.masked_select(target_mask))
            .sum()
            .item()
        )
        weight_sum = weight.sum().item()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    liger_cross_entropy_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        weight_ptr=weight,
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        loss_stride=loss_1d.stride(-1),
        token_accuracy_ptr=token_accuracy_1d,
        token_accuracy_stride=(
            token_accuracy_1d.stride(-1) if return_token_accuracy else 0
        ),
        predicted_tokens_ptr=predicted_tokens_1d,
        predicted_tokens_stride=(
            predicted_tokens_1d.stride(-1) if return_predicted_tokens else 0
        ),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=sum_non_ignore_weight,
        ignore_index=ignore_index,
        weight_sum=weight_sum,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=return_z_loss,
        RETURN_TOKEN_ACCURACY=return_token_accuracy,
        RETURN_PREDICTED_TOKENS=return_predicted_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
        HAS_GRADIENTS=_input.requires_grad,
        num_warps=32 if not is_hip() else 16,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = (
            torch.sum(token_accuracy_1d) / n_non_ignore
            if return_token_accuracy
            else None
        )

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    return loss, z_loss, token_accuracy, predicted_tokens, _input


def cross_entropy_backward(_input, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass
    elif grad_output.ndim > 0:
        _input = _input * grad_output.unsqueeze(dim=1)
    else:
        BT, V = _input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

    return _input


class LigerCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.FloatTensor],
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        input_requires_grad = _input.requires_grad

        loss, z_loss, token_accuracy, predicted_tokens, _input = cross_entropy_forward(
            _input,
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

        if input_requires_grad:
            ctx.save_for_backward(_input.detach())
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens

        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4

        (_input,) = ctx.saved_tensors
        _input = cross_entropy_backward(_input, grad_output)
        return (
            _input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
