import json
import os


def _metrics_dict(metrics) -> dict:
    if metrics is None:
        return {}

    def _ci(lo, hi, prec):
        if lo is None or hi is None:
            return None
        return [round(lo, prec), round(hi, prec)]

    return {
        "accuracy": round(metrics.accuracy, 4),
        "accuracy_ci": [round(metrics.accuracy_ci_low, 4), round(metrics.accuracy_ci_high, 4)],
        "mean_token_count": round(metrics.mean_token_count, 1),
        "mean_token_count_ci": _ci(metrics.mean_token_count_ci_low, metrics.mean_token_count_ci_high, 1),
        "underthinking_rate": (
            round(metrics.underthinking_rate, 4)
            if metrics.underthinking_rate is not None else None
        ),
        "underthinking_rate_ci": _ci(metrics.underthinking_rate_ci_low, metrics.underthinking_rate_ci_high, 4),
        "overthinking_rate": (
            round(metrics.overthinking_rate, 4)
            if metrics.overthinking_rate is not None else None
        ),
        "overthinking_rate_ci": _ci(metrics.overthinking_rate_ci_low, metrics.overthinking_rate_ci_high, 4),
        "overthinking_threshold": (
            round(metrics.overthinking_threshold, 1)
            if metrics.overthinking_threshold is not None else None
        ),
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


def generate_report(
    config: dict,
    ood_results,
    run_dir: str,
    output_dir: str | None = None,
    skip_baseline_compare: bool = False,
) -> dict:
    """
    Build structured assessment report.

    Compares against baseline (runs/e0-baseline-*/eval_report.json) if present.
    No deployment decision is made — output is purely informational for the thesis
    experiment matrix.

    output_dir: where the markdown report is written. Defaults to run_dir.
                Used by baseline mode to write under runs/<exp>/baseline/.
    skip_baseline_compare: when True, skip the sibling-run lookup. Set by
                           baseline mode (a baseline cannot be compared to
                           another baseline meaningfully).
    """
    exp_id = config["experiment_id"]
    output_dir = output_dir or run_dir

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

    # Compare against E0 reward baseline if it exists. This is a
    # trained-vs-trained comparison: a different reward stack against the
    # accuracy-only stack. Useful for reward-ablation deltas, NOT for
    # measuring what finetuning itself did.
    curr_acc = report["results"]["id_split"].get("accuracy")
    if not skip_baseline_compare:
        baseline_path = _find_baseline(run_dir, config)
        if baseline_path:
            with open(baseline_path) as f:
                baseline = json.load(f)
            base_acc = baseline.get("results", {}).get("id_split", {}).get("accuracy")
            if base_acc is not None and curr_acc is not None:
                report["vs_reward_baseline"] = {
                    "baseline_id": baseline.get("experiment_id"),
                    "delta_accuracy": round(curr_acc - base_acc, 4),
                    "baseline_accuracy": base_acc,
                }

        # Compare against the pre-finetune base model assessment, if produced
        # by `eval.runner --baseline`. This is the before-and-after delta:
        # what the finetune did to *this* (model, config) combination.
        base_model_path = os.path.join(run_dir, "baseline", "eval_report.json")
        if os.path.exists(base_model_path):
            with open(base_model_path) as f:
                base_model_report = json.load(f)
            base_results = base_model_report.get("results", {})
            base_id = base_results.get("id_split", {}).get("accuracy")
            base_tokens = base_results.get("id_split", {}).get("mean_token_count")
            curr_tokens = report["results"]["id_split"].get("mean_token_count")
            if base_id is not None and curr_acc is not None:
                delta_tokens = None
                if base_tokens is not None and curr_tokens is not None:
                    delta_tokens = round(curr_tokens - base_tokens, 1)
                report["vs_base_model"] = {
                    "model_slug": base_model_report.get("model_slug"),
                    "baseline_accuracy": base_id,
                    "delta_accuracy": round(curr_acc - base_id, 4),
                    "baseline_mean_tokens": base_tokens,
                    "delta_mean_tokens": delta_tokens,
                }

    # Write human-readable markdown
    _write_markdown(report, output_dir)
    return report


def _find_baseline(run_dir: str, config: dict | None = None) -> str | None:
    """Locate baseline eval_report.json. Explicit `baseline_id` in config wins.
    Heuristic fallback only fires when exactly one e0-* candidate exists; with
    multiple, the function returns None and logs a warning rather than guessing.
    """
    runs_root = os.path.dirname(run_dir)
    self_report = os.path.join(run_dir, "eval_report.json")

    if config and config.get("baseline_id"):
        candidate = os.path.join(runs_root, config["baseline_id"], "eval_report.json")
        if candidate == self_report:
            return None
        if os.path.exists(candidate):
            return candidate
        print(f"Warning: baseline_id={config['baseline_id']!r} given but {candidate} not found")
        return None

    if not os.path.isdir(runs_root):
        return None

    candidates = []
    for entry in sorted(os.listdir(runs_root)):
        if entry.startswith("e0-") or "-baseline-" in entry:
            path = os.path.join(runs_root, entry, "eval_report.json")
            if path != self_report and os.path.exists(path):
                candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(
            f"Warning: multiple baseline candidates found {[os.path.basename(os.path.dirname(c)) for c in candidates]}; "
            "set `baseline_id` in config to disambiguate. Skipping baseline section."
        )
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

    def _ci_suffix(ci):
        if ci is None:
            return ""
        return f" [{ci[0]}, {ci[1]}]"

    for split, metrics in report["results"].items():
        if not metrics:
            continue
        acc_ci = metrics.get("accuracy_ci", ["-", "-"])
        lines += [
            f"### {split.replace('_', ' ').title()}",
            f"- Accuracy: **{metrics.get('accuracy', '-')}** [{acc_ci[0]}, {acc_ci[1]}]",
            f"- Mean tokens: {metrics.get('mean_token_count', '-')}{_ci_suffix(metrics.get('mean_token_count_ci'))}",
        ]
        under_rate = metrics.get("underthinking_rate")
        if under_rate is not None:
            lines.append(f"- Underthinking rate: {under_rate}{_ci_suffix(metrics.get('underthinking_rate_ci'))}")
        else:
            lines.append("- Underthinking rate: -")
        over_rate = metrics.get("overthinking_rate")
        if over_rate is not None:
            over_thr = metrics.get("overthinking_threshold")
            thr_str = f" (threshold: {over_thr} tokens, P75)" if over_thr is not None else ""
            lines.append(f"- Overthinking rate: {over_rate}{_ci_suffix(metrics.get('overthinking_rate_ci'))}{thr_str}")
        else:
            lines.append("- Overthinking rate: -")
        if metrics.get("pearson_difficulty_length") is not None:
            p_str = f" (p={metrics.get('pearson_p_value', '-')})" if metrics.get("pearson_p_value") is not None else ""
            lines.append(f"- Pearson(difficulty, length): {metrics['pearson_difficulty_length']}{p_str}")
        lines.append("")

    if "vs_reward_baseline" in report:
        vb = report["vs_reward_baseline"]
        delta = vb["delta_accuracy"]
        sign = "+" if delta >= 0 else ""
        lines += [
            "## vs Reward Baseline (trained-vs-trained)",
            f"- Baseline: {vb['baseline_id']} (acc={vb['baseline_accuracy']})",
            f"- Δ accuracy: **{sign}{delta}**",
            "",
        ]

    if "vs_base_model" in report:
        vbm = report["vs_base_model"]
        d_acc = vbm["delta_accuracy"]
        sign = "+" if d_acc >= 0 else ""
        lines += [
            "## vs Base Model (before-vs-after finetune)",
            f"- Base model: {vbm.get('model_slug')} (id-split acc={vbm['baseline_accuracy']}, mean tokens={vbm['baseline_mean_tokens']})",
            f"- Δ accuracy: **{sign}{d_acc}**",
        ]
        if vbm.get("delta_mean_tokens") is not None:
            d_tok = vbm["delta_mean_tokens"]
            tsign = "+" if d_tok >= 0 else ""
            lines.append(f"- Δ mean tokens: **{tsign}{d_tok}** (negative = more efficient)")
        lines.append("")

    md_path = os.path.join(run_dir, "eval_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown report written to {md_path}")
