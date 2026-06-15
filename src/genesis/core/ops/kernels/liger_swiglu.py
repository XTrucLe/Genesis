import torch
import triton
import triton.language as tl

from genesis.core.ops.kernels.utils import (
    ensure_contiguous,
    calculate_settings,
)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride,
    gate_multiplier,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    a_row = (
        tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        * gate_multiplier
    )
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dc_ptr,
    a_ptr,
    b_ptr,
    stride,
    gate_multiplier,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row = (
        tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        * gate_multiplier
    )
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row * gate_multiplier

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b, gate_multiplier: float = 1.0):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        float(gate_multiplier),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc, gate_multiplier: float = 1.0):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        float(gate_multiplier),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


class SiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b, gate_multiplier: float = 1.0, down_multiplier: float = 1.0):
        gate_multiplier = float(gate_multiplier)
        down_multiplier = float(down_multiplier)
        ctx.gate_multiplier = gate_multiplier
        ctx.down_multiplier = down_multiplier

        if isinstance(a, torch.distributed.tensor.DTensor) or isinstance(
            b, torch.distributed.tensor.DTensor
        ):
            device_mesh, placements = (
                (a.device_mesh, a.placements)
                if isinstance(a, torch.distributed.tensor.DTensor)
                else (b.device_mesh, b.placements)
            )

            if not isinstance(a, torch.distributed.tensor.DTensor):
                a = torch.distributed.tensor.distribute_tensor(
                    a, device_mesh=device_mesh, placements=placements
                )
            if not isinstance(b, torch.distributed.tensor.DTensor):
                b = torch.distributed.tensor.distribute_tensor(
                    b, device_mesh=device_mesh, placements=placements
                )
            a_local, b_local, c_local = swiglu_forward(
                a.to_local(), b.to_local(), gate_multiplier
            )
            if down_multiplier != 1.0:
                c_local = c_local * down_multiplier
            ctx.save_for_backward(a_local, b_local)
            ctx.dtensor_metadata = (device_mesh, placements)
            return torch.distributed.tensor.DTensor.from_local(
                c_local, device_mesh, placements
            )
        else:
            a, b, c = swiglu_forward(a, b, gate_multiplier)
            if down_multiplier != 1.0:
                c = c * down_multiplier
            ctx.save_for_backward(a, b)
            ctx.dtensor_metadata = None
            return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        gate_multiplier = ctx.gate_multiplier
        down_multiplier = ctx.down_multiplier

        if ctx.dtensor_metadata is not None:
            device_mesh, placements = ctx.dtensor_metadata

            dc_local = (
                dc.to_local()
                if isinstance(dc, torch.distributed.tensor.DTensor)
                else torch.distributed.tensor.distribute_tensor(
                    dc, device_mesh=device_mesh, placements=placements
                )
            )
            if down_multiplier != 1.0:
                dc_local = dc_local * down_multiplier
            a_local, b_local = swiglu_backward(a, b, dc_local, gate_multiplier)
            return (
                torch.distributed.tensor.DTensor.from_local(
                    a_local, device_mesh, placements
                ),
                torch.distributed.tensor.DTensor.from_local(
                    b_local, device_mesh, placements
                ),
                None,
                None,
            )

        if down_multiplier != 1.0:
            dc = dc * down_multiplier
        a, b = swiglu_backward(a, b, dc, gate_multiplier)
        return a, b, None, None
