"""Lazy-built native CUDA implementation of the fused StateHead scan.

The custom kernels fuse gate activations with the recurrent forward/reverse
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
            extra_cuda_cflags=["-O3", "--lineinfo"],
            with_cuda=True,
            verbose=verbose,
        )
    return _EXTENSION


@torch.library.custom_op("nanochat::statehead_scan_backward", mutates_args=())
def _statehead_scan_backward(
    gates: torch.Tensor,
    chunk_initials: torch.Tensor,
    grad_y: torch.Tensor,
    grad_final_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        _load_extension().backward(
            gates,
            chunk_initials,
            grad_y,
            grad_final_state,
            chunk_size,
        )
    )


@_statehead_scan_backward.register_fake
def _statehead_scan_backward_fake(
    gates: torch.Tensor,
    chunk_initials: torch.Tensor,
    grad_y: torch.Tensor,
    grad_final_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del chunk_initials, grad_y, chunk_size
    return torch.empty_like(gates), torch.empty_like(grad_final_state)


@torch.library.custom_op("nanochat::statehead_scan_forward", mutates_args=())
def _statehead_scan_forward(
    gates: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(_load_extension().forward(gates, initial_state, chunk_size))


@_statehead_scan_forward.register_fake
def _statehead_scan_forward_fake(
    gates: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_len, four, n_head, head_dim = gates.shape
    torch._check(four == 4)
    torch._check(initial_state.shape == (batch, n_head, head_dim))
    n_chunks = (sequence_len + chunk_size - 1) // chunk_size
    output = gates.new_empty((batch, sequence_len, n_head, head_dim))
    chunk_initials = gates.new_empty(
        (batch, n_chunks, n_head, head_dim), dtype=torch.float32
    )
    return output, torch.empty_like(initial_state), chunk_initials


def _setup_forward_context(ctx, inputs, output):
    gates, _initial_state, chunk_size = inputs
    _y, _final_state, chunk_initials = output
    ctx.mark_non_differentiable(chunk_initials)
    ctx.save_for_backward(gates, chunk_initials)
    ctx.chunk_size = chunk_size


def _forward_backward(ctx, grad_y, grad_final_state, _grad_chunk_initials):
    gates, chunk_initials = ctx.saved_tensors
    batch, sequence_len, _four, n_head, head_dim = gates.shape
    if grad_y is None:
        grad_y = gates.new_zeros((batch, sequence_len, n_head, head_dim))
    if grad_final_state is None:
        grad_final_state = gates.new_zeros((batch, n_head, head_dim))
    grad_gates, grad_initial_state = _statehead_scan_backward(
        gates,
        chunk_initials,
        grad_y.contiguous(),
        grad_final_state.contiguous(),
        ctx.chunk_size,
    )
    return grad_gates, grad_initial_state, None


_statehead_scan_forward.register_autograd(
    _forward_backward,
    setup_context=_setup_forward_context,
)


def statehead_scan_cuda(gates, initial_state, chunk_size=64):
    """Run the fused scan on raw gates shaped ``[B, T, 4, H, Dh]``."""
    if not gates.is_cuda or not initial_state.is_cuda:
        raise ValueError("statehead_scan_cuda requires CUDA tensors")
    if gates.ndim != 5 or gates.size(2) != 4:
        raise ValueError("gates must have shape [B, T, 4, H, Dh]")
    if chunk_size < 1 or chunk_size > 64:
        raise ValueError("native CUDA scan requires 1 <= chunk_size <= 64")
    gates = gates.contiguous()
    initial_state = initial_state.to(dtype=gates.dtype).contiguous()
    output, final_state, _chunk_initials = _statehead_scan_forward(
        gates,
        initial_state,
        chunk_size,
    )
    return output, final_state


def preload_statehead_cuda(verbose: bool | None = None):
    """Compile/load the native extension before model compilation or DDP work."""
    return _load_extension(verbose=verbose)


if __name__ == "__main__":
    extension = preload_statehead_cuda(verbose=True)
    print(f"loaded {extension.__name__}")
