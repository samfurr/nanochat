"""One-H100 audit for StateHead CORE scoring and recurrent numerical parity.

This is intentionally a token-level audit: it does not require the training
tokenizer, and it does not reproduce a benchmark accuracy. It exercises the
exact trained checkpoint through the production CUDA scan, PyTorch parallel
scan, sequential oracle, batched/padded candidate scoring, one-token state
carry, and optional FP8 projections.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.core_eval import forward_model, stack_sequences
from nanochat.fp8 import convert_to_float8_training, is_float8_linear_eligible
from nanochat.statehead import StateHead, StateHeadConfig
from nanochat.statehead_cuda import preload_statehead_cuda


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp8-examples", type=int, default=32)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    actual_argmax = actual.argmax(dim=-1)
    expected_argmax = expected.argmax(dim=-1)
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "argmax_matches": int((actual_argmax == expected_argmax).sum().item()),
        "argmax_total": actual_argmax.numel(),
    }


def state_stats(actual, expected):
    differences = [
        (actual_state.float() - expected_state.float()).abs()
        for actual_state, expected_state in zip(actual, expected)
    ]
    return {
        "max_abs": max(difference.max().item() for difference in differences),
        "mean_abs": sum(
            difference.mean().item() for difference in differences
        ) / len(differences),
    }


def make_candidates(seed, vocab_size, prefix_length=48):
    generator = torch.Generator().manual_seed(seed)
    prefix = torch.randint(
        1,
        vocab_size,
        (prefix_length,),
        generator=generator,
    ).tolist()
    lengths = [50, 55, 60, 65]
    sequences = []
    for length in lengths:
        answer = torch.randint(
            1,
            vocab_size,
            (length - prefix_length,),
            generator=generator,
        ).tolist()
        sequences.append(prefix + answer)
    return sequences, [prefix_length] * len(sequences), lengths


@torch.inference_mode()
def score_batched(model, sequences, starts, ends, scan_impl=None):
    input_ids = stack_sequences(sequences, pad_token_id=0).to(model.get_device())
    if scan_impl is None:
        losses, _ = forward_model(model, input_ids)
    else:
        logits, _, _ = model.forward_with_state(input_ids, scan_impl=scan_impl)
        targets = torch.roll(input_ids, shifts=-1, dims=1)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(input_ids)
        losses[:, -1] = torch.nan
    return torch.stack([
        losses[row, start - 1:end - 1].mean()
        for row, (start, end) in enumerate(zip(starts, ends))
    ])


@torch.inference_mode()
def score_individually(model, sequences, starts, ends, scan_impl):
    scores = []
    for sequence, start, end in zip(sequences, starts, ends):
        input_ids = torch.tensor(
            sequence,
            device=model.get_device(),
        ).unsqueeze(0)
        logits, _, _ = model.forward_with_state(
            input_ids,
            scan_impl=scan_impl,
        )
        scores.append(F.cross_entropy(
            logits[:, start - 1:end - 1].reshape(-1, logits.size(-1)),
            input_ids[:, start:end].reshape(-1),
        ))
    return torch.stack(scores)


def score_stats(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "predicted_choice_actual": int(actual.argmin().item()),
        "predicted_choice_expected": int(expected.argmin().item()),
        "prediction_matches": bool(actual.argmin() == expected.argmin()),
    }


def load_model(checkpoint, metadata_path):
    metadata = json.loads(metadata_path.read_text())
    config = StateHeadConfig(**metadata["model_config"])
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    with torch.device("meta"):
        model = StateHead(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model = model.cuda().eval()
    return model, metadata


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit(
            f"Hopper SM90 required, got {torch.cuda.get_device_capability()}"
        )

    started = time.perf_counter()
    preload_statehead_cuda()
    compile_seconds = time.perf_counter() - started
    model, metadata = load_model(args.checkpoint, args.metadata)
    device = model.get_device()
    torch.manual_seed(20260725)
    torch.cuda.manual_seed_all(20260725)
    torch.cuda.reset_peak_memory_stats()

    # Full production CUDA sequence versus both reference implementations.
    full_tokens = torch.randint(
        0,
        model.config.vocab_size,
        (2, 65),
        device=device,
    )
    with torch.inference_mode():
        cuda_logits, cuda_states, _ = model.forward_with_state(
            full_tokens,
            scan_impl="cuda",
        )
        parallel_logits, parallel_states, _ = model.forward_with_state(
            full_tokens,
            scan_impl="parallel",
        )
        sequential_logits, sequential_states, _ = model.forward_with_state(
            full_tokens,
            scan_impl="sequential",
        )

    full_sequence = {
        "cuda_vs_pytorch_parallel_logits": tensor_stats(
            cuda_logits,
            parallel_logits,
        ),
        "cuda_vs_pytorch_parallel_states": state_stats(
            cuda_states,
            parallel_states,
        ),
        "cuda_vs_sequential_logits": tensor_stats(
            cuda_logits,
            sequential_logits,
        ),
        "cuda_vs_sequential_states": state_stats(
            cuda_states,
            sequential_states,
        ),
        "pytorch_parallel_vs_sequential_logits": tensor_stats(
            parallel_logits,
            sequential_logits,
        ),
        "pytorch_parallel_vs_sequential_states": state_stats(
            parallel_states,
            sequential_states,
        ),
    }

    # Current CORE behavior: batched, right-padded candidates from fresh state.
    sequences, starts, ends = make_candidates(
        20260726,
        model.config.vocab_size,
    )
    cuda_batched = score_batched(model, sequences, starts, ends)
    cuda_individual = score_individually(
        model,
        sequences,
        starts,
        ends,
        "cuda",
    )
    parallel_individual = score_individually(
        model,
        sequences,
        starts,
        ends,
        "parallel",
    )
    sequential_individual = score_individually(
        model,
        sequences,
        starts,
        ends,
        "sequential",
    )
    permutation = [2, 0, 3, 1]
    permuted = score_batched(
        model,
        [sequences[index] for index in permutation],
        [starts[index] for index in permutation],
        [ends[index] for index in permutation],
    )
    restored = torch.empty_like(permuted)
    for new_index, old_index in enumerate(permutation):
        restored[old_index] = permuted[new_index]

    candidate_scoring = {
        "cuda_batched_vs_cuda_individual": score_stats(
            cuda_batched,
            cuda_individual,
        ),
        "cuda_batched_vs_pytorch_parallel_individual": score_stats(
            cuda_batched,
            parallel_individual,
        ),
        "cuda_batched_vs_sequential_individual": score_stats(
            cuda_batched,
            sequential_individual,
        ),
        "candidate_order_invariance": score_stats(
            restored,
            cuda_batched,
        ),
        "cuda_batched_scores": cuda_batched.float().cpu().tolist(),
        "sequential_individual_scores": (
            sequential_individual.float().cpu().tolist()
        ),
    }

    # The current returned-state behavior versus an FP32-state sequential oracle.
    decode_tokens = full_tokens[:1, :16]
    with torch.inference_mode():
        decode_full_cuda, decode_full_cuda_states, _ = (
            model.forward_with_state(decode_tokens, scan_impl="cuda")
        )
        states = None
        previous_embedding = None
        pieces = []
        for position in range(decode_tokens.size(1)):
            logits, states, previous_embedding = model.forward_with_state(
                decode_tokens[:, position:position + 1],
                states=states,
                prev_embedding=previous_embedding,
                scan_impl="cuda",
            )
            pieces.append(logits)
        decode_token_cuda = torch.cat(pieces, dim=1)
        decode_token_cuda_states = states

        fp32_initial_states = [
            block.state_bank.fresh_state(
                1,
                device,
                torch.float32,
            ).clone()
            for block in model.transformer.h
        ]
        fp32_full, fp32_full_states, _ = model.forward_with_state(
            decode_tokens,
            states=[state.clone() for state in fp32_initial_states],
            scan_impl="sequential",
        )
        states = [state.clone() for state in fp32_initial_states]
        previous_embedding = None
        pieces = []
        for position in range(decode_tokens.size(1)):
            logits, states, previous_embedding = model.forward_with_state(
                decode_tokens[:, position:position + 1],
                states=states,
                prev_embedding=previous_embedding,
                scan_impl="sequential",
            )
            pieces.append(logits)
        fp32_token = torch.cat(pieces, dim=1)

    recurrent_decode = {
        "current_bf16_cuda_full_vs_token_logits": tensor_stats(
            decode_full_cuda,
            decode_token_cuda,
        ),
        "current_bf16_cuda_full_vs_token_states": state_stats(
            decode_full_cuda_states,
            decode_token_cuda_states,
        ),
        "fp32_state_sequential_full_vs_token_logits": tensor_stats(
            fp32_full,
            fp32_token,
        ),
        "fp32_state_sequential_full_vs_token_states": state_stats(
            fp32_full_states,
            states,
        ),
        "fp32_returned_state_dtype": str(states[0].dtype),
    }
    # Save the actual BF16 CUDA token-by-token states before the variable is reused.
    # The logit comparison is the decisive symptom; state precision is recorded by dtype.
    recurrent_decode["current_cuda_returned_state_dtype"] = str(
        decode_full_cuda_states[0].dtype
    )

    # Hypothetical FP8 evaluation versus the BF16 evaluation actually used by CORE.
    bf16_score_sets = []
    generated_sets = []
    for example in range(args.fp8_examples):
        candidate_set = make_candidates(
            20261000 + example,
            model.config.vocab_size,
        )
        generated_sets.append(candidate_set)
        bf16_score_sets.append(score_batched(model, *candidate_set).cpu())
    convert_to_float8_training(
        model,
        module_filter_fn=is_float8_linear_eligible,
    )
    fp8_score_sets = [
        score_batched(model, *candidate_set).cpu()
        for candidate_set in generated_sets
    ]
    score_differences = [
        (fp8.float() - bf16.float()).abs()
        for fp8, bf16 in zip(fp8_score_sets, bf16_score_sets)
    ]
    prediction_flips = sum(
        int(fp8.argmin() != bf16.argmin())
        for fp8, bf16 in zip(fp8_score_sets, bf16_score_sets)
    )
    fp8_vs_bf16 = {
        "examples": args.fp8_examples,
        "candidate_scores": args.fp8_examples * 4,
        "prediction_flips": prediction_flips,
        "max_abs_mean_loss_difference": max(
            difference.max().item() for difference in score_differences
        ),
        "mean_abs_mean_loss_difference": sum(
            difference.mean().item() for difference in score_differences
        ) / len(score_differences),
    }

    core_path_checks = {
        "candidate_order_prediction_matches": (
            candidate_scoring["candidate_order_invariance"]["prediction_matches"]
        ),
        "batched_vs_individual_prediction_matches": (
            candidate_scoring["cuda_batched_vs_cuda_individual"][
                "prediction_matches"
            ]
        ),
        "cuda_vs_parallel_prediction_matches": (
            candidate_scoring[
                "cuda_batched_vs_pytorch_parallel_individual"
            ]["prediction_matches"]
        ),
        "cuda_vs_sequential_prediction_matches": (
            candidate_scoring["cuda_batched_vs_sequential_individual"][
                "prediction_matches"
            ]
        ),
        "full_cuda_vs_parallel_all_argmax_match": (
            full_sequence["cuda_vs_pytorch_parallel_logits"][
                "argmax_matches"
            ]
            == full_sequence["cuda_vs_pytorch_parallel_logits"]["argmax_total"]
        ),
        "full_cuda_vs_sequential_all_argmax_match": (
            full_sequence["cuda_vs_sequential_logits"]["argmax_matches"]
            == full_sequence["cuda_vs_sequential_logits"]["argmax_total"]
        ),
    }
    core_path_pass = all(core_path_checks.values())

    torch.cuda.synchronize()
    report = {
        "verdict": {
            "token_level_full_sequence_scoring_path_pass": core_path_pass,
            "current_bf16_token_decode_parity_pass": (
                recurrent_decode[
                    "current_bf16_cuda_full_vs_token_logits"
                ]["max_abs"] == 0.0
            ),
            "fp32_state_sequential_decode_parity_pass": (
                recurrent_decode[
                    "fp32_state_sequential_full_vs_token_logits"
                ]["max_abs"] == 0.0
            ),
        },
        "core_path_checks": core_path_checks,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "compute_dtype": str(COMPUTE_DTYPE),
            "master_parameter_dtype": str(next(model.parameters()).dtype),
            "compile_seconds": compile_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "step": metadata["step"],
            "model_config": metadata["model_config"],
        },
        "full_sequence": full_sequence,
        "candidate_scoring": candidate_scoring,
        "recurrent_decode": recurrent_decode,
        "fp8_vs_bf16": fp8_vs_bf16,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not core_path_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
