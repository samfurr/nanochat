"""Aggregate full CORE/BPB evaluations for a timed StateHead checkpoint curve."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_core_metric(path: Path) -> float:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip() == "CORE":
                return float(row[2])
    raise ValueError(f"CORE row missing from {path}")


def read_bpb(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    train = re.findall(r"^train bpb: ([0-9.]+)$", text, flags=re.MULTILINE)
    val = re.findall(r"^val bpb: ([0-9.]+)$", text, flags=re.MULTILINE)
    if len(train) != 1 or len(val) != 1:
        raise ValueError(
            f"expected one train/val BPB result in {path}, got {train=} {val=}"
        )
    return float(train[0]), float(val[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    target_by_step = {
        int(item["step"]): int(item["target_training_seconds"])
        for item in schedule["checkpoints"]
    }
    models = sorted(args.checkpoint_dir.glob("model_*.pt"))
    if not models:
        raise ValueError(f"no checkpoints found in {args.checkpoint_dir}")

    rows = []
    for model_path in models:
        match = re.fullmatch(r"model_(\d{6})\.pt", model_path.name)
        if match is None:
            continue
        step_padded = match.group(1)
        step = int(step_padded)
        meta_path = args.checkpoint_dir / f"meta_{step_padded}.json"
        eval_log = args.results_dir / f"{args.tag}-eval_{step_padded}.log"
        core_csv = args.results_dir / f"{args.tag}-core_{step_padded}.csv"
        for required in (meta_path, eval_log, core_csv):
            if not required.is_file():
                raise FileNotFoundError(required)
        if step not in target_by_step:
            raise ValueError(f"checkpoint step {step} missing from {args.schedule}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        train_bpb, val_bpb = read_bpb(eval_log)
        training_seconds = float(meta["loop_state"]["total_training_time"])
        rows.append({
            "step": step,
            "target_training_seconds": target_by_step[step],
            "training_seconds": training_seconds,
            "training_minutes": training_seconds / 60,
            "tokens": step * int(meta["total_batch_size"]),
            "train_bpb": train_bpb,
            "val_bpb": val_bpb,
            "core": read_core_metric(core_csv),
            "model_file": model_path.name,
            "meta_file": meta_path.name,
        })

    expected_steps = sorted(target_by_step)
    actual_steps = [row["step"] for row in rows]
    if actual_steps != expected_steps:
        raise ValueError(
            f"checkpoint steps do not match schedule: {actual_steps=} {expected_steps=}"
        )

    args.output_json.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "target_training_seconds": schedule["target_training_seconds"],
                "checkpoint_count": len(rows),
                "checkpoints": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "step",
        "target_training_seconds",
        "training_seconds",
        "training_minutes",
        "tokens",
        "train_bpb",
        "val_bpb",
        "core",
        "model_file",
        "meta_file",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
