"""Compare declared costs on saved training batches; never infer learned gains."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from training.rewards.successful_length import successful_length_costs


def reference_costs(lengths, correct, kind):
    values = [n for n, y in zip(lengths, correct, strict=True) if y]
    if kind == "linear":
        return [
            min(n, 4096) / 4096 if y else 0
            for n, y in zip(lengths, correct, strict=True)
        ]
    if not values:
        return [0.0] * len(lengths)
    mean = sum(values) / len(values)
    std = max((sum((n - mean) ** 2 for n in values) / len(values)) ** 0.5, 1)
    return [
        1 / (1 + math.exp((mean - n) / std)) if y else 0
        for n, y in zip(lengths, correct, strict=True)
    ]


def centered(values):
    return [v - statistics.mean(values) for v in values]


def summarize(rows):
    values = [r["mean_abs_length_advantage"] for r in rows]
    return {
        "groups": len(rows),
        "median_mean_abs_length_advantage": (
            statistics.median(values) if values else None
        ),
        "p90_mean_abs_length_advantage": (
            statistics.quantiles(values, n=10, method="inclusive")[8]
            if len(values) > 1
            else values[0]
            if values
            else None
        ),
        "active_groups": sum(v > 1e-12 for v in values),
    }


def compare(capture: Path, declaration: Path) -> dict:
    batches = [json.loads(line) for line in capture.read_text().splitlines()]
    if [b["optimizer_step"] for b in batches] != [1, 100, 200, *range(291, 301)]:
        raise ValueError("Expected declared E1 training captures, including late ten")
    rows = []
    max_error = 0.0
    for batch in batches:
        records = batch["records"]
        assert batch["group_size"] == 8 and len(records) == 32
        assert batch["global_step_before_update"] == batch["optimizer_step"] - 1
        for start in range(0, 32, 8):
            group = records[start : start + 8]
            assert len({r["seed"] for r in group}) == 1
            assert all(r["group_index"] == start // 8 for r in group)
            assert [r["slot_index"] for r in group] == list(range(8))
            lengths = [r["model_tokens"] for r in group]
            correct = [r["correct"] for r in group]
            assert all(
                r["raw_rewards"]["env_reward"] == int(r["correct"]) for r in group
            )
            for kind in ("linear", "relative", "historical_cosine"):
                weights = (0.4,) if kind == "historical_cosine" else (0.1, 0.2, 0.4)
                if kind == "historical_cosine":
                    raw = [
                        (1 if y else -1)
                        * (0.75 + 0.25 * math.cos(math.pi * min(n / 4096, 1)))
                        for n, y in zip(lengths, correct, strict=True)
                    ]
                else:
                    costs = successful_length_costs(lengths, correct, kind, 8)
                    expected = reference_costs(lengths, correct, kind)
                    max_error = max(
                        max_error,
                        max(abs(a - b) for a, b in zip(costs, expected, strict=True)),
                    )
                    assert max_error < 1e-12
                    assert all(
                        0 <= c <= 1 and (y or c == 0)
                        for c, y in zip(costs, correct, strict=True)
                    )
                    raw = [-c for c in costs]
                for weight in weights:
                    shaping = centered([weight * r for r in raw])
                    total = [
                        int(y) + weight * r for y, r in zip(correct, raw, strict=True)
                    ]
                    if kind != "historical_cosine":
                        assert all(
                            (1 - weight <= r <= 1) if y else r == 0
                            for r, y in zip(total, correct, strict=True)
                        )
                    rows.append(
                        {
                            "step": batch["optimizer_step"],
                            "seed": group[0]["seed"],
                            "kind": kind,
                            "weight": weight,
                            "correct_count": sum(correct),
                            "length_range": max(lengths) - min(lengths),
                            "mean_abs_length_advantage": statistics.mean(
                                abs(a) for a in shaping
                            ),
                            "max_abs_length_advantage": max(abs(a) for a in shaping),
                            "total_reward_range": max(total) - min(total),
                        }
                    )
    candidates = []
    for kind in ("linear", "relative", "historical_cosine"):
        for weight in (0.4,) if kind == "historical_cosine" else (0.1, 0.2, 0.4):
            subset = [r for r in rows if r["kind"] == kind and r["weight"] == weight]
            late = [r for r in subset if r["step"] >= 291]
            primary = summarize(
                [r for r in late if r["correct_count"] == 8 and r["length_range"] > 0]
            )
            eligible = (
                kind != "historical_cosine"
                and primary["groups"] > 0
                and primary["median_mean_abs_length_advantage"] >= 0.01
                and primary["p90_mean_abs_length_advantage"] <= 0.05
            )
            candidates.append(
                {
                    "kind": kind,
                    "weight": weight,
                    "eligible": eligible,
                    "late_all_correct_varying": primary,
                    "late_mixed": summarize(
                        [r for r in late if 0 < r["correct_count"] < 8]
                    ),
                    "all_captures": summarize(subset),
                }
            )
    eligible = [c for c in candidates if c["eligible"]]
    selected = (
        min(eligible, key=lambda c: (c["kind"] != "linear", c["weight"]))
        if eligible
        else None
    )
    return {
        "purpose": "development incentive calibration, not a learned compression result",
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                capture,
                declaration,
                Path(__file__),
                Path("training/rewards/successful_length.py"),
            )
        },
        "trajectories": sum(len(b["records"]) for b in batches),
        "independent_formula_max_error": max_error,
        "candidates": candidates,
        "selected": {k: selected[k] for k in ("kind", "weight")} if selected else None,
        "groups": rows,
        "limits": [
            "Recorded training data only; no held-out inputs.",
            "Native tokenizer/reward replay still required.",
            "Selection does not predict learned compression or success preservation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        Path("runs/table-pilot-e1-s4009/group_observations.jsonl"),
        Path("../docs/plans/read-table-compression.md"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            {k: result[k] for k in ("trajectories", "candidates", "selected")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
