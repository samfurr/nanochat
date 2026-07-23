"""Bounded CUDA/DDP feasibility and full-step comparison probe.

This deliberately uses synthetic token rows. It checks whether the full d12 shape,
compiler, backward pass, and distributed Muon/AdamW optimizer execute for GPT and
StateHead before any dataset-backed training is authorized. It is not a
learning-quality run.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from nanochat.common import COMPUTE_DTYPE, compute_cleanup, compute_init, print0
from nanochat.gpt import GPT, GPTConfig
from nanochat.statehead import StateHead, StateHeadConfig


def distributed_max(value, device):
    result = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def parameter_checksum(model):
    checksum = torch.zeros((), dtype=torch.float64, device=model.get_device())
    with torch.no_grad():
        for parameter in model.parameters():
            checksum += parameter.double().sum()
    return checksum


def main():
    parser = argparse.ArgumentParser(description="CUDA/DDP full-step comparison probe")
    parser.add_argument("--arch", choices=["gpt", "statehead"], default="statehead")
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="exclude this many initial compile/warmup steps from the steady-state summary",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--model-width", type=int, default=768)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--scan-chunk-size", type=int, default=64)
    parser.add_argument(
        "--scan-backend",
        choices=["pytorch", "cuda", "cuda_projected", "cuda_projected_gates"],
        default="pytorch",
    )
    parser.add_argument(
        "--projection-tile-rows",
        type=int,
        default=16384,
        help="flattened token rows per cuBLAS tile for cuda_projected",
    )
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--eager", action="store_true", help="disable torch.compile")
    parser.add_argument(
        "--verify-gradients",
        action="store_true",
        help="check every parameter gradient for finiteness (stability mode, not timing mode)",
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="delimit post-warmup steps with the CUDA profiler API for Nsight Systems",
    )
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if COMPUTE_DTYPE != torch.bfloat16:
        raise RuntimeError(
            f"Preflight requires NANOCHAT_DTYPE=bfloat16, got {COMPUTE_DTYPE}"
        )
    if args.steps < 1 or args.device_batch_size < 1:
        raise ValueError("--steps and --device-batch-size must be positive")
    if args.projection_tile_rows < 1:
        raise ValueError("--projection-tile-rows must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be non-negative and less than --steps")
    if args.arch == "gpt" and args.scan_backend != "pytorch":
        raise ValueError("GPT has no StateHead scan backend; use --scan-backend=pytorch")

    ddp, rank, local_rank, world_size, device = compute_init("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.arch == "gpt":
        config = GPTConfig(
            sequence_len=args.sequence_length,
            vocab_size=args.vocab_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_kv_head=args.heads,
            n_embd=args.model_width,
            window_pattern="SSSL",
        )
        model_cls = GPT
    else:
        config = StateHeadConfig(
            sequence_len=args.sequence_length,
            vocab_size=args.vocab_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.model_width,
            scan_chunk_size=args.scan_chunk_size,
            scan_backend=args.scan_backend,
            projection_tile_rows=args.projection_tile_rows,
        )
        model_cls = StateHead
    with torch.device("meta"):
        model = model_cls(config)
    model.to_empty(device=device)
    model.init_weights()
    original_model = model

    if args.arch == "statehead" and args.scan_backend in (
        "cuda",
        "cuda_projected",
        "cuda_projected_gates",
    ):
        from nanochat.statehead_cuda import preload_statehead_cuda

        preload_statehead_cuda()

    counts = original_model.num_scaling_params()
    scaling_params = counts["transformer_matrices"] + counts["lm_head"]
    flops_per_token = original_model.estimate_flops()
    state_bytes_per_rank = (
        original_model.recurrent_state_bytes(
            batch_size=args.device_batch_size,
            dtype=torch.bfloat16,
        )
        if args.arch == "statehead"
        else 0
    )

    fp8_linear_count = 0
    if args.fp8:
        from nanochat.fp8 import convert_to_float8_training, is_float8_linear_eligible

        convert_to_float8_training(
            model,
            module_filter_fn=is_float8_linear_eligible,
        )
        fp8_linear_count = sum(
            "Float8" in type(module).__name__ for module in model.modules()
        )

    if not args.eager:
        model = torch.compile(model, dynamic=False)
    optimizer = model.setup_optimizer(
        unembedding_lr=0.008,
        embedding_lr=0.3,
        scalar_lr=0.5,
        matrix_lr=0.02,
        weight_decay=0.28,
    )

    group_summary = []
    for group in optimizer.param_groups:
        group_summary.append({
            "kind": group["kind"],
            "tensors": len(group["params"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        })

    input_generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    inputs = torch.randint(
        0,
        args.vocab_size,
        (args.device_batch_size, args.sequence_length),
        dtype=torch.long,
        device=device,
        generator=input_generator,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    step_records = []
    for step in range(args.steps):
        if args.nsys_capture and step == args.warmup_steps:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
        start = time.perf_counter()
        loss = model(inputs, targets)
        loss.backward()
        if args.verify_gradients:
            local_grad_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in original_model.parameters()
            )
            finite_flag = torch.tensor(int(local_grad_finite), device=device)
            if dist.is_initialized():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if not bool(finite_flag.item()):
                raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        local_seconds = time.perf_counter() - start
        loss_value = loss.item()
        if not torch.isfinite(torch.tensor(loss_value)):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss_value}")
        max_seconds = distributed_max(local_seconds, device)
        global_tokens = args.device_batch_size * args.sequence_length * world_size
        step_records.append({
            "step": step,
            "loss_rank0": loss_value if rank == 0 else None,
            "seconds_max_rank": max_seconds,
            "global_tokens_per_second": global_tokens / max_seconds,
        })
        print0(
            f"step={step} loss={loss_value:.6f} max_rank_seconds={max_seconds:.3f} "
            f"global_tok_per_sec={global_tokens / max_seconds:,.0f}"
        )
    if args.nsys_capture:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    checksum = parameter_checksum(original_model)
    checksum_min = checksum.clone()
    checksum_max = checksum.clone()
    if dist.is_initialized():
        dist.all_reduce(checksum_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(checksum_max, op=dist.ReduceOp.MAX)
    checksum_spread = (checksum_max - checksum_min).item()

    peak_allocated = distributed_max(torch.cuda.max_memory_allocated(device), device)
    peak_reserved = distributed_max(torch.cuda.max_memory_reserved(device), device)
    steady_records = step_records[args.warmup_steps:]
    steady_seconds = [record["seconds_max_rank"] for record in steady_records]
    steady_throughput = [
        record["global_tokens_per_second"] for record in steady_records
    ]
    steady_summary = {
        "excluded_warmup_steps": args.warmup_steps,
        "measured_steps": len(steady_records),
        "seconds_median_max_rank": statistics.median(steady_seconds),
        "seconds_mean_max_rank": statistics.mean(steady_seconds),
        "global_tokens_per_second_median": statistics.median(steady_throughput),
        "global_tokens_per_second_mean": statistics.mean(steady_throughput),
        "loss_first_rank0": steady_records[0]["loss_rank0"],
        "loss_last_rank0": steady_records[-1]["loss_rank0"],
    }
    result = {
        "probe": "cuda_full_step_comparison",
        "scientific_result": False,
        "architecture": args.arch,
        "repo_commit": os.environ.get("NANOCHAT_REPO_COMMIT", "unknown"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "world_size": world_size,
        "compiled": not args.eager,
        "seed": args.seed,
        "precision": str(COMPUTE_DTYPE),
        "fp8": args.fp8,
        "fp8_linear_count": fp8_linear_count,
        "gradient_finiteness_checked": args.verify_gradients,
        "config": vars(args),
        "parameter_counts": counts,
        "scaling_params": scaling_params,
        "flops_per_token": flops_per_token,
        "state_bytes_per_rank": state_bytes_per_rank,
        "optimizer_groups": group_summary,
        "steps": step_records,
        "steady_state": steady_summary,
        "peak_allocated_bytes_max_rank": int(peak_allocated),
        "peak_reserved_bytes_max_rank": int(peak_reserved),
        "parameter_checksum_spread": checksum_spread,
    }
    print0(json.dumps(result, indent=2))
    if rank == 0 and args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")

    compute_cleanup()


if __name__ == "__main__":
    main()
