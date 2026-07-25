"""Tests for StateHead checkpoint-curve aggregation."""

import csv
import json
import sys

from dev.aggregate_statehead_checkpoint_curve import main


def test_checkpoint_curve_aggregation(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    checkpoint_dir.mkdir()
    results_dir.mkdir()
    tag = "statehead-test"
    step = 123
    padded = f"{step:06d}"

    (checkpoint_dir / f"model_{padded}.pt").write_bytes(b"model")
    (checkpoint_dir / f"meta_{padded}.json").write_text(
        json.dumps({
            "step": step,
            "total_batch_size": 1024,
            "loop_state": {"total_training_time": 600.5},
        }),
        encoding="utf-8",
    )
    (results_dir / f"{tag}-eval_{padded}.log").write_text(
        "train bpb: 0.812345\nval bpb: 0.823456\n",
        encoding="utf-8",
    )
    (results_dir / f"{tag}-core_{padded}.csv").write_text(
        "Task, Accuracy, Centered\nCORE, , 0.234567\n",
        encoding="utf-8",
    )
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps({
            "target_training_seconds": 5940,
            "checkpoints": [{
                "target_training_seconds": 600,
                "step": step,
                "kind": "interval",
            }],
        }),
        encoding="utf-8",
    )
    output_json = tmp_path / "curve.json"
    output_csv = tmp_path / "curve.csv"
    monkeypatch.setattr(sys, "argv", [
        "aggregate_statehead_checkpoint_curve",
        f"--checkpoint-dir={checkpoint_dir}",
        f"--results-dir={results_dir}",
        f"--tag={tag}",
        f"--schedule={schedule}",
        f"--output-json={output_json}",
        f"--output-csv={output_csv}",
    ])

    main()

    aggregate = json.loads(output_json.read_text(encoding="utf-8"))
    assert aggregate["checkpoint_count"] == 1
    row = aggregate["checkpoints"][0]
    assert row["step"] == step
    assert row["training_seconds"] == 600.5
    assert row["tokens"] == step * 1024
    assert row["train_bpb"] == 0.812345
    assert row["val_bpb"] == 0.823456
    assert row["core"] == 0.234567
    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["step"] == str(step)
