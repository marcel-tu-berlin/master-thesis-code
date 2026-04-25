from __future__ import annotations

import json
import os
from dataclasses import asdict


def _metrics_dict(metrics) -> dict:
    if metrics is None:
        return {}
    return {
        "accuracy": round(metrics.accuracy, 4),
        "accuracy_ci": [round(metrics.accuracy_ci_low, 4), round(metrics.accuracy_ci_high, 4)],
        "mean_token_count": round(metrics.mean_token_count, 1),
        "underthinking_rate": round(metrics.underthinking_rate, 4),
        "pearson_difficulty_length": (
            round(metrics.pearson_difficulty_length, 4)
            if metrics.pearson_difficulty_length is not None else None
        ),
        "pearson_p_value": (
            round(metrics.pearson_p_value, 6)
            if metrics.pearson_p_value is not None else None
        ),
        "n_samples": metrics.n_samples,
        "n_correct": metrics.n_correct,
    }


def generate_report(config: dict, ood_results, run_dir: str) -> dict:
    """
    Build structured evaluation report.

    Compares against baseline (runs/e0-baseline-*/eval_report.json) if present.
    No deployment decision is made — output is purely informational for the thesis
    experiment matrix.
    """
    exp_id = config["experiment_id"]

    report = {
        "experiment_id": exp_id,
        "model_slug": config["model"]["slug"],
        "compose_method": config.get("rewards", {}).get("compose_method", "advantage_weighted"),
        "results": {
            "id_split": _metrics_dict(ood_results.id_split),
            "near_ood": _metrics_dict(ood_results.near_ood),
            "far_ood": _metrics_dict(ood_results.far_ood),
            "capability_floor": _metrics_dict(ood_results.capability_floor),
        },
    }

    # Compare against E0 baseline if it exists
    baseline_path = _find_baseline(run_dir, config)
    if baseline_path:
        with open(baseline_path) as f:
            baseline = json.load(f)
        base_acc = baseline.get("results", {}).get("id_split", {}).get("accuracy")
        curr_acc = report["results"]["id_split"].get("accuracy")
        if base_acc is not None and curr_acc is not None:
            report["vs_baseline"] = {
                "baseline_id": baseline.get("experiment_id"),
                "delta_accuracy": round(curr_acc - base_acc, 4),
                "baseline_accuracy": base_acc,
            }

    # Write human-readable markdown
    _write_markdown(report, run_dir)
    return report


def _find_baseline(run_dir: str, config: dict | None = None) -> str | None:
    runs_root = os.path.dirname(run_dir)

    # Explicit baseline_id from config takes priority
    if config and config.get("baseline_id"):
        candidate = os.path.join(runs_root, config["baseline_id"], "eval_report.json")
        if os.path.exists(candidate) and candidate != os.path.join(run_dir, "eval_report.json"):
            return candidate

    # Fallback heuristic: match entries starting with "e0-" or containing "-baseline-"
    for entry in sorted(os.listdir(runs_root)):
        if entry.startswith("e0-") or "-baseline-" in entry:
            candidate = os.path.join(runs_root, entry, "eval_report.json")
            if os.path.exists(candidate) and candidate != os.path.join(run_dir, "eval_report.json"):
                return candidate
    return None


def _write_markdown(report: dict, run_dir: str) -> None:
    exp_id = report["experiment_id"]
    lines = [
        f"# Evaluation Report — {exp_id}",
        "",
        f"**Model:** {report['model_slug']}  ",
        f"**Compose method:** {report['compose_method']}",
        "",
        "## Results",
        "",
    ]

    for split, metrics in report["results"].items():
        if not metrics:
            continue
        lines += [
            f"### {split.replace('_', ' ').title()}",
            f"- Accuracy: **{metrics.get('accuracy', '-')}** [{metrics.get('accuracy_ci', ['-', '-'])[0]}, {metrics.get('accuracy_ci', ['-', '-'])[1]}]",
            f"- Mean tokens: {metrics.get('mean_token_count', '-')}",
            f"- Underthinking rate: {metrics.get('underthinking_rate', '-')}",
        ]
        if metrics.get("pearson_difficulty_length") is not None:
            p_str = f" (p={metrics.get('pearson_p_value', '-')})" if metrics.get("pearson_p_value") is not None else ""
            lines.append(f"- Pearson(difficulty, length): {metrics['pearson_difficulty_length']}{p_str}")
        lines.append("")

    if "vs_baseline" in report:
        vb = report["vs_baseline"]
        delta = vb["delta_accuracy"]
        sign = "+" if delta >= 0 else ""
        lines += [
            "## vs Baseline",
            f"- Baseline: {vb['baseline_id']} (acc={vb['baseline_accuracy']})",
            f"- Δ accuracy: **{sign}{delta}**",
            "",
        ]

    md_path = os.path.join(run_dir, "eval_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown report written to {md_path}")
