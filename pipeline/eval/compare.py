"""
Multi-experiment comparison visualizations.

Usage:
    python -m eval.compare --runs runs/e0-baseline runs/e1-token-entropy runs/e2-multi-signal
    python -m eval.compare --runs runs/e0-baseline runs/e1-token-entropy --out runs/my-comparison
"""

import argparse
import json
import os

import numpy as np


def _load_reports(run_dirs: list[str]) -> list[dict]:
    reports = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "eval_report.json")
        if not os.path.exists(path):
            print(f"Warning: no eval_report.json in {run_dir}, skipping")
            continue
        with open(path) as f:
            r = json.load(f)
        r["_run_dir"] = run_dir
        reports.append(r)
    return reports


def _baseline_acc(reports: list[dict]) -> float | None:
    """Pick a baseline ID accuracy. If exactly one e0-* report is present, use it.
    With multiple e0-* reports, return None and warn — caller should set
    `baseline_id` upstream to make the choice explicit.
    """
    candidates = [r for r in reports if r.get("experiment_id", "").startswith("e0")]
    if len(candidates) == 1:
        return (candidates[0].get("results", {}).get("id_split") or {}).get("accuracy")
    if len(candidates) > 1:
        ids = [r.get("experiment_id") for r in candidates]
        print(f"Warning: multiple baseline (e0-*) reports {ids}; baseline line/delta omitted from comparison.")
    return None


def _plot_compare_accuracy(reports: list[dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    splits = [("id_split", "ID"), ("near_ood", "Near-OOD"), ("far_ood", "Far-OOD")]
    colors = ["steelblue", "coral", "seagreen"]
    exp_ids = [r.get("experiment_id", os.path.basename(r["_run_dir"])) for r in reports]
    n_splits = len(splits)
    x = np.arange(len(reports))
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(8, len(reports) * 1.8), 5))

    for i, (split_key, split_label) in enumerate(splits):
        accs, err_lo, err_hi = [], [], []
        for r in reports:
            split = (r.get("results") or {}).get(split_key)
            if split and split.get("accuracy") is not None:
                acc = split["accuracy"]
                ci = split.get("accuracy_ci", [acc, acc])
                accs.append(acc)
                err_lo.append(acc - ci[0])
                err_hi.append(ci[1] - acc)
            else:
                accs.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)

        offset = (i - n_splits / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=split_label, color=colors[i], alpha=0.8,
               yerr=[err_lo, err_hi], capsize=3, error_kw={"linewidth": 1.2})

    baseline = _baseline_acc(reports)
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                   label=f"Baseline ID ({baseline:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by Experiment and Split")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "compare_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def _plot_compare_efficiency(reports: list[dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in reports:
        exp_id = r.get("experiment_id", os.path.basename(r["_run_dir"]))
        split = (r.get("results") or {}).get("id_split")
        if not split:
            continue
        acc = split.get("accuracy")
        tokens = split.get("mean_token_count")
        if acc is None or tokens is None:
            continue
        ax.scatter(tokens, acc, s=80, zorder=3)
        ax.annotate(exp_id, (tokens, acc), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Mean Token Count (ID Split)")
    ax.set_ylabel("Accuracy (ID Split)")
    ax.set_title("Accuracy vs Efficiency — Experiment Comparison")
    ax.annotate("← fewer tokens\n↑ better accuracy", xy=(0.02, 0.90),
                xycoords="axes fraction", fontsize=8, color="gray")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "compare_efficiency.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def _write_compare_summary(reports: list[dict], out_dir: str) -> None:
    baseline = _baseline_acc(reports)

    rows = [
        "| Experiment | Accuracy (ID) | Δ vs Baseline | Mean Tokens | Underthinking |",
        "|-----------|:------------:|:------------:|:-----------:|:-------------:|",
    ]
    for r in reports:
        exp_id = r.get("experiment_id", os.path.basename(r["_run_dir"]))
        split = (r.get("results") or {}).get("id_split") or {}
        acc = split.get("accuracy")
        tokens = split.get("mean_token_count")
        under = split.get("underthinking_rate")

        acc_str = f"{acc:.4f}" if acc is not None else "—"
        delta_str = "—"
        if acc is not None and baseline is not None:
            d = acc - baseline
            delta_str = f"{'+' if d >= 0 else ''}{d:.4f}"
        tokens_str = f"{tokens:.1f}" if tokens is not None else "—"
        under_str = f"{under:.4f}" if under is not None else "—"

        rows.append(f"| {exp_id} | {acc_str} | {delta_str} | {tokens_str} | {under_str} |")

    out_path = os.path.join(out_dir, "compare_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Summary written: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval results across experiments")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directories containing eval_report.json")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: runs/comparison)")
    args = parser.parse_args()

    reports = _load_reports(args.runs)
    if not reports:
        print("No valid reports found.")
        return

    out_dir = args.out or "runs/comparison"
    os.makedirs(out_dir, exist_ok=True)

    _plot_compare_accuracy(reports, out_dir)
    _plot_compare_efficiency(reports, out_dir)
    _write_compare_summary(reports, out_dir)


if __name__ == "__main__":
    main()
