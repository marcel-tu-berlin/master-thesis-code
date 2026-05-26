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
        # NaN signals "data unavailable" to matplotlib so the bar is omitted
        # rather than rendered as 0% — the latter falsely reads as "model
        # got 0/N" instead of "this experiment didn't run that probe".
        accs, err_lo, err_hi = [], [], []
        for r in reports:
            split = (r.get("results") or {}).get(split_key)
            if split and split.get("accuracy") is not None:
                acc = split["accuracy"]
                # Explicit None check: a missing accuracy_ci means "no CI was
                # measured", which is different from "CI is zero-width". The
                # old default of [acc, acc] silently rendered zero-width error
                # bars that visually read as a tightly-measured estimate.
                ci = split.get("accuracy_ci")
                accs.append(acc)
                if ci is None:
                    err_lo.append(0.0)
                    err_hi.append(0.0)
                else:
                    err_lo.append(acc - ci[0])
                    err_hi.append(ci[1] - acc)
            else:
                accs.append(np.nan)
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


def _reward_family(exp_id: str, model_slug: str | None) -> str:
    """Strip the trailing model-slug variant from experiment_id, leaving the
    reward-stack identity. Examples:
      e0-baseline-math-1.5b           -> e0-baseline-math    (model: qwen-1.5b)
      e1-token-length-qwen-7b         -> e1-token-length     (model: qwen-7b)
      e1-token-entropy-forkmask       -> e1-token-entropy-forkmask
    Falls back to the full id if no known suffix matches.
    """
    if not model_slug:
        return exp_id
    # Try a few common ways the slug appears in ids.
    for suffix in (f"-{model_slug}", f"-{model_slug.replace('-', '')}", f"-{model_slug.split('-')[-1]}"):
        if exp_id.endswith(suffix):
            return exp_id[: -len(suffix)]
    return exp_id


def _pareto_indices(points: list[tuple[float, float]]) -> set[int]:
    """Return indices of Pareto-optimal points. Each point is (tokens, accuracy);
    minimize tokens, maximize accuracy. A point i is dominated if there exists j
    with tokens[j] <= tokens[i] AND accuracy[j] >= accuracy[i] AND at least one
    strict inequality.
    """
    optimal = set()
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            optimal.add(i)
    return optimal


def _group_by_model(reports: list[dict]) -> dict[str, list[dict]]:
    """Group reports by model_slug. Reports without a slug fall into 'unknown'."""
    groups: dict[str, list[dict]] = {}
    for r in reports:
        slug = r.get("model_slug") or "unknown"
        groups.setdefault(slug, []).append(r)
    return groups


def _draw_efficiency_panel(ax, reports: list[dict], panel_title: str | None = None) -> None:
    """Draw one accuracy-vs-tokens panel. Colour per reward family, marker per
    compose method, error bars from CIs, bold edge on Pareto-optimal points.
    """
    import matplotlib.pyplot as plt

    points: list[tuple[float, float]] = []
    rows = []  # (tokens, acc, x_err, y_err, family, compose, exp_id)
    for r in reports:
        exp_id = r.get("experiment_id", os.path.basename(r["_run_dir"]))
        split = (r.get("results") or {}).get("id_split")
        if not split:
            continue
        acc = split.get("accuracy")
        tokens = split.get("mean_token_count")
        if acc is None or tokens is None:
            continue
        acc_ci = split.get("accuracy_ci")
        tok_ci = split.get("mean_token_count_ci")
        y_err = (acc - acc_ci[0], acc_ci[1] - acc) if acc_ci else (0.0, 0.0)
        x_err = (tokens - tok_ci[0], tok_ci[1] - tokens) if tok_ci else (0.0, 0.0)
        family = _reward_family(exp_id, r.get("model_slug"))
        compose = r.get("compose_method", "advantage_weighted")
        rows.append((tokens, acc, x_err, y_err, family, compose, exp_id))
        points.append((tokens, acc))

    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    pareto = _pareto_indices(points)

    # Stable colour per family + marker per compose method
    families = sorted({r[4] for r in rows})
    cmap = plt.get_cmap("tab10")
    fam_color = {f: cmap(i % 10) for i, f in enumerate(families)}
    compose_marker = {"advantage_weighted": "o", "naive_sum": "s"}

    for i, (tokens, acc, x_err, y_err, family, compose, exp_id) in enumerate(rows):
        marker = compose_marker.get(compose, "^")
        edge = "black" if i in pareto else "none"
        lw = 1.8 if i in pareto else 0.0
        ax.errorbar(
            tokens, acc,
            xerr=[[x_err[0]], [x_err[1]]],
            yerr=[[y_err[0]], [y_err[1]]],
            fmt=marker, markersize=10, color=fam_color[family],
            ecolor="gray", elinewidth=0.8, capsize=2.5, alpha=0.9,
            markeredgecolor=edge, markeredgewidth=lw, zorder=3,
        )
        ax.annotate(exp_id, (tokens, acc), textcoords="offset points",
                    xytext=(6, 6), fontsize=6, alpha=0.8)

    ax.set_xlabel("Mean Token Count (ID)")
    ax.set_ylabel("Accuracy (ID)")
    if panel_title:
        ax.set_title(panel_title, fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_compare_efficiency(
    reports: list[dict],
    out_dir: str,
    facet_by: str | None = None,
) -> None:
    """Pareto plot: accuracy vs mean token count.

    facet_by='model' produces one subplot per model_slug; otherwise a single
    panel. Reward family is encoded as colour; compose method as marker.
    Pareto-optimal points (non-dominated on accuracy↑ × tokens↓) get a bold
    black edge.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return

    if facet_by == "model":
        groups = _group_by_model(reports)
        n = len(groups)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
        flat_axes = axes.flatten()
        for ax, (slug, group_reports) in zip(flat_axes, sorted(groups.items())):
            _draw_efficiency_panel(ax, group_reports, panel_title=f"model: {slug}")
        for ax in flat_axes[n:]:
            ax.set_visible(False)
        fig.suptitle("Accuracy vs Efficiency — by Model", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_efficiency_panel(ax, reports)
        ax.set_title("Accuracy vs Efficiency — Experiment Comparison")

    # Build a shared legend for reward families + compose methods
    all_families = sorted({_reward_family(r.get("experiment_id", ""), r.get("model_slug")) for r in reports})
    all_composes = sorted({r.get("compose_method", "advantage_weighted") for r in reports})
    cmap = plt.get_cmap("tab10")
    handles = [
        Line2D([], [], marker="o", color="w", markerfacecolor=cmap(i % 10),
               markersize=9, label=fam)
        for i, fam in enumerate(all_families)
    ]
    compose_marker = {"advantage_weighted": "o", "naive_sum": "s"}
    for c in all_composes:
        handles.append(Line2D([], [], marker=compose_marker.get(c, "^"),
                              color="gray", markersize=9, linestyle="None", label=f"compose: {c}"))
    handles.append(Line2D([], [], marker="o", color="w", markerfacecolor="lightgray",
                          markersize=10, markeredgecolor="black", markeredgewidth=1.8,
                          label="Pareto-optimal"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "compare_efficiency.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def _paired_bootstrap_delta(
    samples_a: list[dict],
    samples_b: list[dict],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float, float] | None:
    """Paired bootstrap on per-sample correctness vectors.

    Returns (delta, ci_low, ci_high, p_value) for accuracy(A) − accuracy(B),
    or None when pairing is not possible (length mismatch or missing data).

    The test resamples *indices* with replacement and recomputes the delta on
    the resampled vector. Pairing preserves the per-sample correlation between
    experiments (both saw the same prompts), so the resulting CI is tighter
    than an unpaired test would give and reflects within-prompt variance.

    The two-sided p-value is the bootstrap-percentile version: `2 × min(P(Δ ≤ 0),
    P(Δ ≥ 0))`, capped at 1.0. It answers "is zero a plausible value for the
    paired difference under resampling".

    Deterministic via a fixed RNG seed so the comparison report is reproducible.
    """
    if not samples_a or not samples_b:
        return None
    if len(samples_a) != len(samples_b):
        return None

    a = np.array([1.0 if s.get("correct") else 0.0 for s in samples_a])
    b = np.array([1.0 if s.get("correct") else 0.0 for s in samples_b])
    delta = float(a.mean() - b.mean())

    rng = np.random.default_rng(42)
    n = len(a)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = a[idx].mean() - b[idx].mean()

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1 - alpha)))
    # Two-sided p via bootstrap percentile.
    p_le = float((boot <= 0).mean())
    p_ge = float((boot >= 0).mean())
    p = min(1.0, 2.0 * min(p_le, p_ge))
    return delta, lo, hi, p


def _write_compare_pairwise(reports: list[dict], out_dir: str) -> None:
    """Pairwise paired-bootstrap accuracy tests on the ID split.

    Writes a markdown matrix of Δ-accuracy (A − B) with 95% CI and p-value
    for every ordered pair of experiments where both reports carry per-sample
    series of equal length. Pairs without matched samples are reported as "—"
    with the reason (missing samples, length mismatch).
    """
    rows_with_samples: list[tuple[str, list[dict]]] = []
    skipped: list[str] = []
    for r in reports:
        exp_id = r.get("experiment_id", os.path.basename(r["_run_dir"]))
        split = (r.get("results") or {}).get("id_split") or {}
        samples = split.get("samples")
        if not samples:
            skipped.append(exp_id)
            continue
        rows_with_samples.append((exp_id, samples))

    lines = ["# Pairwise paired-bootstrap comparison (ID split)", ""]
    if skipped:
        lines.append(
            "_Skipped (no per-sample series in eval_report.json — rerun eval to "
            f"populate): {', '.join(skipped)}_"
        )
        lines.append("")

    if len(rows_with_samples) < 2:
        lines.append("_Need at least two experiments with per-sample series to compare._")
        out_path = os.path.join(out_dir, "compare_pairwise.md")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Pairwise comparison written: {out_path}")
        return

    lines.append(
        "Each cell is Δ-accuracy = row − column on matched samples, with the "
        "95% bootstrap CI and a two-sided p-value. Bold p < 0.05."
    )
    lines.append("")
    header = "| A \\ B | " + " | ".join(eid for eid, _ in rows_with_samples) + " |"
    sep = "|" + "---|" * (len(rows_with_samples) + 1)
    lines.append(header)
    lines.append(sep)

    for a_id, a_samples in rows_with_samples:
        cells = [a_id]
        for b_id, b_samples in rows_with_samples:
            if a_id == b_id:
                cells.append("—")
                continue
            result = _paired_bootstrap_delta(a_samples, b_samples)
            if result is None:
                cells.append("(length mismatch)")
                continue
            d, lo, hi, p = result
            sign = "+" if d >= 0 else ""
            p_str = f"**p={p:.3f}**" if p < 0.05 else f"p={p:.3f}"
            cells.append(f"{sign}{d:.4f} [{lo:+.4f}, {hi:+.4f}] {p_str}")
        lines.append("| " + " | ".join(cells) + " |")

    out_path = os.path.join(out_dir, "compare_pairwise.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Pairwise comparison written: {out_path}")


def _write_compare_summary(reports: list[dict], out_dir: str) -> None:
    baseline = _baseline_acc(reports)

    rows = [
        "| Experiment | Accuracy (ID) | Δ vs Baseline | Mean Tokens | Underthinking | Overthinking |",
        "|-----------|:------------:|:------------:|:-----------:|:-------------:|:------------:|",
    ]
    for r in reports:
        exp_id = r.get("experiment_id", os.path.basename(r["_run_dir"]))
        split = (r.get("results") or {}).get("id_split") or {}
        acc = split.get("accuracy")
        tokens = split.get("mean_token_count")
        under = split.get("underthinking_rate")
        over = split.get("overthinking_rate")

        acc_str = f"{acc:.4f}" if acc is not None else "—"
        delta_str = "—"
        if acc is not None and baseline is not None:
            d = acc - baseline
            delta_str = f"{'+' if d >= 0 else ''}{d:.4f}"
        tokens_str = f"{tokens:.1f}" if tokens is not None else "—"
        under_str = f"{under:.4f}" if under is not None else "—"
        over_str = f"{over:.4f}" if over is not None else "—"

        rows.append(f"| {exp_id} | {acc_str} | {delta_str} | {tokens_str} | {under_str} | {over_str} |")

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
    parser.add_argument("--facet-by", choices=["model", "none"], default="none",
                        help="Split the efficiency Pareto plot into subplots (default: none)")
    args = parser.parse_args()

    reports = _load_reports(args.runs)
    if not reports:
        print("No valid reports found.")
        return

    out_dir = args.out or "runs/comparison"
    os.makedirs(out_dir, exist_ok=True)

    facet = args.facet_by if args.facet_by != "none" else None
    _plot_compare_accuracy(reports, out_dir)
    _plot_compare_efficiency(reports, out_dir, facet_by=facet)
    _write_compare_summary(reports, out_dir)
    _write_compare_pairwise(reports, out_dir)


if __name__ == "__main__":
    main()
