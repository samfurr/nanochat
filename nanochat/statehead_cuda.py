"""Lazy-built native CUDA implementation of the fused StateHead scan.

The custom kernels cache gate activations once for the recurrent forward/reverse
scan. Large gate and output projections remain regular GEMMs, and every state
transition is accumulated in float32 even when gates and outputs are bfloat16.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import torch


_EXTENSION_NAME = "nanochat_statehead_cuda_ext"
_EXTENSION = None
_EXTENSION_LOCK = Lock()


def _load_extension(verbose: bool | None = None):
    """Build once per source/PyTorch/CUDA combination, then reuse the cache."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("The native StateHead scan requires a CUDA-enabled PyTorch runtime")

    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).resolve().parent / "csrc"
        if verbose is None:
            verbose = os.environ.get("NANOCHAT_STATEHEAD_CUDA_VERBOSE", "0") == "1"
        _EXTENSION = load(
            name=_EXTENSION_NAME,
            sources=[
                str(source_dir / "statehead_cuda.cpp"),
                str(source_dir / "statehead_cuda_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_ldflags=["-lcublas"],
            with_cuda=True,
            verbose=verbose,
        )
    return _EXTENSION


@torch.library.custom_op("nanochat::statehead_scan_backward", mutates_args=())
def _statehead_scan_backward(
    activated_gates: torch.Tensor,
    chunk_initials: torch.Tensor,
    grad_y: torch.Tensor,
    grad_final_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        _load_extension().backward(
            activated_gates,
            chunk_initials,
            grad_y,
            grad_final_state,
            chunk_size,
        )
    )


@_statehead_scan_backward.register_fake
def _statehead_scan_backward_fake(
    activated_gates: torch.Tensor,
    chunk_initials: torch.Tensor,
    grad_y: torch.Tensor,
    grad_final_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del chunk_initials, grad_y, chunk_size
    return torch.empty_like(activated_gates), torch.empty_like(grad_final_state)


@torch.library.custom_op("nanochat::statehead_projected_forward", mutates_args=())
def _statehead_projected_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    initial_state: torch.Tensor,
    n_head: int,
    chunk_size: int,
    projection_tile_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        _load_extension().forward_projected(
            x,
            gate_weight,
            gate_bias,
            initial_state,
            n_head,
            chunk_size,
            projection_tile_rows,
        )
    )


@_statehead_projected_forward.register_fake
def _statehead_projected_forward_fake(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    initial_state: torch.Tensor,
    n_head: int,
    chunk_size: int,
    projection_tile_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_len, n_embd = x.shape
    torch._check(gate_weight.shape == (4 * n_embd, n_embd))
    torch._check(gate_bias.numel() == 4 * n_embd)
    torch._check(n_embd % n_head == 0)
    head_dim = n_embd // n_head
    torch._check(initial_state.shape == (batch, n_head, head_dim))
    torch._check(projection_tile_rows > 0)
    n_chunks = (sequence_len + chunk_size - 1) // chunk_size
    output = x.new_empty((batch, sequence_len, n_head, head_dim))
    chunk_initials = x.new_empty(
        (batch, n_chunks, n_head, head_dim), dtype=torch.float32
    )
    activated_gates = x.new_empty(
        (batch, sequence_len, 4, n_head, head_dim)
    )
    return output, torch.empty_like(initial_state), chunk_initials, activated_gates


def _setup_projected_forward_context(ctx, inputs, output):
    x, gate_weight, gate_bias, _initial_state, _n_head, chunk_size, _tile_rows = inputs
    _y, _final_state, chunk_initials, activated_gates = output
    ctx.mark_non_differentiable(chunk_initials, activated_gates)
    ctx.save_for_backward(x, gate_weight, activated_gates, chunk_initials)
    ctx.gate_bias_shape = gate_bias.shape
    ctx.chunk_size = chunk_size


def _projected_forward_backward(
    ctx,
    grad_y,
    grad_final_state,
    _grad_chunk_initials,
    _grad_activated_gates,
):
    x, gate_weight, activated_gates, chunk_initials = ctx.saved_tensors
    batch, sequence_len, _four, n_head, head_dim = activated_gates.shape
    if grad_y is None:
        grad_y = activated_gates.new_zeros(
            (batch, sequence_len, n_head, head_dim)
        )
    if grad_final_state is None:
        grad_final_state = activated_gates.new_zeros((batch, n_head, head_dim))
    grad_gates, grad_initial_state = _statehead_scan_backward(
        activated_gates,
        chunk_initials,
        grad_y.contiguous(),
        grad_final_state.contiguous(),
        ctx.chunk_size,
    )
    flat_x = x.reshape(batch * sequence_len, -1)
    flat_grad_gates = grad_gates.reshape(batch * sequence_len, -1)
    grad_x = torch.mm(flat_grad_gates, gate_weight).reshape_as(x)
    grad_gate_weight = torch.mm(flat_grad_gates.t(), flat_x)
    grad_gate_bias = flat_grad_gates.sum(dim=0).reshape(ctx.gate_bias_shape)
    return (
        grad_x,
        grad_gate_weight,
        grad_gate_bias,
        grad_initial_state,
        None,
        None,
        None,
    )


_statehead_projected_forward.register_autograd(
    _projected_forward_backward,
    setup_context=_setup_projected_forward_context,
)


@torch.library.custom_op("nanochat::statehead_scan_forward", mutates_args=())
def _statehead_scan_forward(
    gates: torch.Tensor,
    gate_bias: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(_load_extension().forward(gates, gate_bias, initial_state, chunk_size))


@_statehead_scan_forward.register_fake
def _statehead_scan_forward_fake(
    gates: torch.Tensor,
    gate_bias: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_len, four, n_head, head_dim = gates.shape
    torch._check(four == 4)
    torch._check(initial_state.shape == (batch, n_head, head_dim))
    torch._check(gate_bias.numel() == 4 * n_head * head_dim)
    n_chunks = (sequence_len + chunk_size - 1) // chunk_size
    output = gates.new_empty((batch, sequence_len, n_head, head_dim))
    chunk_initials = gates.new_empty(
        (batch, n_chunks, n_head, head_dim), dtype=torch.float32
    )
    return (
        output,
        torch.empty_like(initial_state),
        chunk_initials,
        torch.empty_like(gates),
    )


def _setup_forward_context(ctx, inputs, output):
    _gates, gate_bias, _initial_state, chunk_size = inputs
    _y, _final_state, chunk_initials, activated_gates = output
    ctx.mark_non_differentiable(chunk_initials, activated_gates)
    ctx.save_for_backward(activated_gates, chunk_initials)
    ctx.gate_bias_shape = gate_bias.shape
    ctx.chunk_size = chunk_size


def _forward_backward(
    ctx,
    grad_y,
    grad_final_state,
    _grad_chunk_initials,
    _grad_activated_gates,
):
    activated_gates, chunk_initials = ctx.saved_tensors
    batch, sequence_len, _four, n_head, head_dim = activated_gates.shape
    if grad_y is None:
        grad_y = activated_gates.new_zeros(
            (batch, sequence_len, n_head, head_dim)
        )
    if grad_final_state is None:
        grad_final_state = activated_gates.new_zeros((batch, n_head, head_dim))
    grad_gates, grad_initial_state = _statehead_scan_backward(
        activated_gates,
        chunk_initials,
        grad_y.contiguous(),
        grad_final_state.contiguous(),
        ctx.chunk_size,
    )
    grad_gate_bias = grad_gates.sum(dim=(0, 1)).reshape(ctx.gate_bias_shape)
    return grad_gates, grad_gate_bias, grad_initial_state, None


_statehead_scan_forward.register_autograd(
    _forward_backward,
    setup_context=_setup_forward_context,
)


def statehead_scan_cuda(gates, initial_state, chunk_size=64, gate_bias=None):
    """Run the fused scan on raw gates shaped ``[B, T, 4, H, Dh]``."""
    if not gates.is_cuda or not initial_state.is_cuda:
        raise ValueError("statehead_scan_cuda requires CUDA tensors")
    if gates.ndim != 5 or gates.size(2) != 4:
        raise ValueError("gates must have shape [B, T, 4, H, Dh]")
    if chunk_size < 1 or chunk_size > 64:
        raise ValueError("native CUDA scan requires 1 <= chunk_size <= 64")
    gates = gates.contiguous()
    if gate_bias is None:
        gate_bias = gates.new_zeros((4, gates.size(3), gates.size(4)))
    if gate_bias.numel() != 4 * gates.size(3) * gates.size(4):
        raise ValueError("gate_bias must have 4 * H * Dh elements")
    gate_bias = gate_bias.to(dtype=gates.dtype, device=gates.device).contiguous()
    initial_state = initial_state.to(dtype=gates.dtype).contiguous()
    output, final_state, _chunk_initials, _activated_gates = _statehead_scan_forward(
        gates,
        gate_bias,
        initial_state,
        chunk_size,
    )
    return output, final_state


def statehead_projected_cuda(
    x,
    gate_weight,
    gate_bias,
    initial_state,
    n_head,
    chunk_size=64,
    projection_tile_rows=16384,
):
    """Pipeline cuBLAS gate tiles with the mixed activation and native scan."""
    tensors = (x, gate_weight, gate_bias, initial_state)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("statehead_projected_cuda requires CUDA tensors")
    if x.ndim != 3:
        raise ValueError("x must have shape [B, T, D]")
    batch, _sequence_len, n_embd = x.shape
    if n_head < 1 or n_embd % n_head != 0:
        raise ValueError("n_head must divide the model width")
    if gate_weight.shape != (4 * n_embd, n_embd):
        raise ValueError("gate_weight must have shape [4D, D]")
    if gate_bias.numel() != 4 * n_embd:
        raise ValueError("gate_bias must have 4D elements")
    if initial_state.shape != (batch, n_head, n_embd // n_head):
        raise ValueError("initial_state must have shape [B, H, Dh]")
    if chunk_size < 1 or chunk_size > 64:
        raise ValueError("native CUDA scan requires 1 <= chunk_size <= 64")
    if projection_tile_rows < 1:
        raise ValueError("projection_tile_rows must be positive")
    x = x.contiguous()
    gate_weight = gate_weight.to(dtype=x.dtype, device=x.device).contiguous()
    gate_bias = gate_bias.to(dtype=x.dtype, device=x.device).contiguous()
    initial_state = initial_state.to(dtype=x.dtype, device=x.device).contiguous()
    output, final_state, _chunk_initials, _activated_gates = (
        _statehead_projected_forward(
            x,
            gate_weight,
            gate_bias,
            initial_state,
            n_head,
            chunk_size,
            projection_tile_rows,
        )
    )
    return output, final_state


def preload_statehead_cuda(verbose: bool | None = None):
    """Compile/load the native extension before model compilation or DDP work."""
    return _load_extension(verbose=verbose)


if __name__ == "__main__":
    extension = preload_statehead_cuda(verbose=True)
    print(f"loaded {extension.__name__}")
