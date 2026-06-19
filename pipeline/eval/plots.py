"""Figures from agentic eval reports (and optional training logs).

Reads runs/<exp>/eval_report.json (the "agentic" split) and renders the thesis
figures with matplotlib (Agg backend, headless - no display needed):

- comparison.png       success rate + mean completion tokens (with CIs), bars
                       across experiments. The ablation headline.
- distributions.png    per-experiment token histograms, correct vs wrong. Wrong
                       completions pile at the generation cap (the model never
                       emits the tool call, so the stream is truncated), so the
                       split separates real efficiency from a failure artifact.
- efficiency.png       success rate vs mean tokens on CORRECT episodes - the
                       token-efficiency frontier (down-and-right is better).
- training_curves_<exp>.png  reward / completion length / KL / loss and the
                       per-component raw reward over steps. Drawn only for runs
                       that carry a train_log.json (written by training; runs
                       from before that hook existed have none).

The eval report already stores Wilson (accuracy) and bootstrap (tokens) CIs, so
the bar plots just read them; only "mean tokens on correct" is recomputed here,
reusing eval.metrics._bootstrap_ci.

CLI:
  python -m eval.plots runs/e5-... runs/e6-... -o runs/plots
  python -m eval.plots --glob 'runs/e*' -o runs/plots
"""
import argparse
import glob as globmod
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.metrics import _bootstrap_ci  # noqa: E402

_SPLIT = "agentic"

# Standard TRL/GRPO per-step series, in panel order. Plotted only when present
# (key names vary across TRL versions, so missing keys are skipped, not errors).
_CURVE_KEYS = [
    ("reward", "mean reward"),
    ("completions/mean_length", "mean completion length (tokens)"),
    ("completions/mean_terminated_length", "terminated-only length (tokens)"),
    ("completions/clipped_ratio", "clipped at cap (fraction)"),
    ("kl", "KL"),
    ("loss", "loss"),
]


def load_report(path: str) -> dict:
    """Load an eval report from a run dir or a JSON file.

    Returns {"experiment_id", "agentic" (the split metrics dict), "samples"}.
    Raises FileNotFoundError if the report is missing, ValueError if it has no
    agentic split.
    """
    json_path = os.path.join(path, "eval_report.json") if os.path.isdir(path) else path
    with open(json_path) as f:                       # FileNotFoundError propagates
        report = json.load(f)
    split = (report.get("results") or {}).get(_SPLIT)
    if split is None:
        raise ValueError(f"{json_path}: no {_SPLIT!r} split in results")
    return {
        "experiment_id": report.get("experiment_id", "?"),
        "agentic": split,
        "samples": split.get("samples") or [],
    }


def _short(exp_id: str) -> str:
    """Compact label for axes: the leading handle, e.g. e5-agentic-... -> e5."""
    return (exp_id or "?").split("-")[0]


def _correct_wrong_tokens(samples):
    c = np.array([s["n_tokens"] for s in samples if s.get("correct")], dtype=float)
    w = np.array([s["n_tokens"] for s in samples if not s.get("correct")], dtype=float)
    return c, w


def _mean_ci_on_correct(samples):
    """Mean completion tokens over CORRECT episodes, with a bootstrap CI.

    This is the honest efficiency number: wrong episodes hit the generation cap
    (truncated before the tool call), so including them measures failure, not
    length. Returns (0, 0, 0) when there are no correct episodes.
    """
    c, _ = _correct_wrong_tokens(samples)
    if len(c) == 0:
        return 0.0, 0.0, 0.0
    lo, hi = _bootstrap_ci(c)
    return float(c.mean()), lo, hi


def _err(center, lo, hi):
    """matplotlib 1D asymmetric error pair [down, up], clamped >= 0."""
    return [max(0.0, center - lo), max(0.0, hi - center)]


def plot_comparison(reports, fig=None):
    """Two bar panels: success rate (Wilson CI) and mean tokens (bootstrap CI)."""
    if fig is None:
        fig = plt.figure(figsize=(max(7.0, 1.6 * len(reports) + 4), 4.3))
    ax_acc, ax_tok = fig.subplots(1, 2)
    labels = [_short(r["experiment_id"]) for r in reports]
    x = np.arange(len(reports))

    acc = [r["agentic"]["accuracy"] for r in reports]
    acc_err = np.array([_err(r["agentic"]["accuracy"], r["agentic"]["accuracy_ci_low"],
                             r["agentic"]["accuracy_ci_high"]) for r in reports]).T
    ax_acc.bar(x, acc, color="#4C72B0")
    ax_acc.errorbar(x, acc, yerr=acc_err, fmt="none", ecolor="black", capsize=4)
    ax_acc.set_xticks(x), ax_acc.set_xticklabels(labels)
    ax_acc.set_ylim(0, 1), ax_acc.set_ylabel("success rate")
    ax_acc.set_title("Success rate (Wilson 95%)")

    tok = [r["agentic"]["mean_token_count"] for r in reports]
    tok_err = np.array([_err(r["agentic"]["mean_token_count"], r["agentic"]["mean_token_count_ci_low"],
                             r["agentic"]["mean_token_count_ci_high"]) for r in reports]).T
    ax_tok.bar(x, tok, color="#C44E52")
    ax_tok.errorbar(x, tok, yerr=tok_err, fmt="none", ecolor="black", capsize=4)
    ax_tok.set_xticks(x), ax_tok.set_xticklabels(labels)
    ax_tok.set_ylabel("mean completion tokens")
    ax_tok.set_title("Token cost (bootstrap 95%)")

    fig.suptitle("Ablation: success vs token cost")
    fig.tight_layout()
    return fig


def plot_distributions(reports, fig=None):
    """One token histogram per experiment, correct (green) vs wrong (red)."""
    n = len(reports)
    ncols = min(3, n) or 1
    nrows = (n + ncols - 1) // ncols
    if fig is None:
        fig = plt.figure(figsize=(4.6 * ncols, 3.5 * nrows))
    for i, r in enumerate(reports):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        c, w = _correct_wrong_tokens(r["samples"])
        if len(c):
            ax.hist(c, bins=20, alpha=0.6, color="#55A868", label=f"correct (n={len(c)})")
        if len(w):
            ax.hist(w, bins=20, alpha=0.6, color="#C44E52", label=f"wrong (n={len(w)})")
        ax.set_title(_short(r["experiment_id"]))
        ax.set_xlabel("completion tokens"), ax.set_ylabel("episodes")
        if len(c) or len(w):
            ax.legend(fontsize=8)
    fig.suptitle("Token distribution: correct vs wrong")
    fig.tight_layout()
    return fig


def plot_efficiency(reports, fig=None):
    """Scatter: success rate vs mean tokens on correct episodes (the frontier)."""
    if fig is None:
        fig = plt.figure(figsize=(6.6, 5.0))
    ax = fig.subplots(1, 1)
    for r in reports:
        acc = r["agentic"]["accuracy"]
        mean_c, lo, hi = _mean_ci_on_correct(r["samples"])
        ax.errorbar(
            mean_c, acc,
            xerr=[[max(0.0, mean_c - lo)], [max(0.0, hi - mean_c)]],
            yerr=[[max(0.0, acc - r["agentic"]["accuracy_ci_low"])],
                  [max(0.0, r["agentic"]["accuracy_ci_high"] - acc)]],
            fmt="o", capsize=3, markersize=8,
        )
        ax.annotate(_short(r["experiment_id"]), (mean_c, acc),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel("mean tokens on correct episodes")
    ax.set_ylabel("success rate")
    ax.set_title("Token-efficiency frontier (down-and-right is better)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _series(log, key):
    xs, ys = [], []
    for i, e in enumerate(log):
        v = e.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            xs.append(e.get("step", i))
            ys.append(v)
    return xs, ys


def plot_training_curves(log_history, fig=None):
    """Per-step training curves from TRL log_history. None if there is no data.

    Standard panels (reward / length / KL / loss) are drawn when present, plus a
    "reward components" panel overlaying each reward/<name>/raw_mean series.
    """
    if not log_history:
        return None
    panels = [(k, lbl) for k, lbl in _CURVE_KEYS if _series(log_history, k)[1]]
    comp_keys = sorted({k for e in log_history for k in e
                        if k.startswith("reward/") and k.endswith("/raw_mean")})
    total = len(panels) + (1 if comp_keys else 0)
    if total == 0:
        return None
    ncols = min(2, total)
    nrows = (total + ncols - 1) // ncols
    if fig is None:
        fig = plt.figure(figsize=(6.4 * ncols, 3.4 * nrows))
    idx = 1
    for key, label in panels:
        ax = fig.add_subplot(nrows, ncols, idx)
        idx += 1
        xs, ys = _series(log_history, key)
        ax.plot(xs, ys, color="#4C72B0")
        ax.set_xlabel("step"), ax.set_ylabel(label), ax.set_title(label)
        ax.grid(True, alpha=0.3)
    if comp_keys:
        ax = fig.add_subplot(nrows, ncols, idx)
        for k in comp_keys:
            xs, ys = _series(log_history, k)
            ax.plot(xs, ys, label=k.split("/")[1])
        ax.set_xlabel("step"), ax.set_ylabel("raw reward mean")
        ax.set_title("reward components"), ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training curves")
    fig.tight_layout()
    return fig


def make_figures(report_paths, out_dir, dpi=130):
    """Render all figures for the given run dirs / report paths into out_dir.

    Returns the list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    reports = [load_report(p) for p in report_paths]
    written = []

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    _save(plot_comparison(reports), "comparison.png")
    _save(plot_distributions(reports), "distributions.png")
    _save(plot_efficiency(reports), "efficiency.png")

    for p, r in zip(report_paths, reports):
        run_dir = p if os.path.isdir(p) else os.path.dirname(p)
        train_log = os.path.join(run_dir, "train_log.json")
        if not os.path.exists(train_log):
            continue
        with open(train_log) as f:
            fig = plot_training_curves(json.load(f))
        if fig is not None:
            _save(fig, f"training_curves_{_short(r['experiment_id'])}.png")
    return written


def main():
    ap = argparse.ArgumentParser(description="Render thesis figures from eval reports.")
    ap.add_argument("runs", nargs="*", help="run dirs or eval_report.json paths")
    ap.add_argument("--glob", help="glob for run dirs, e.g. 'runs/e*'")
    ap.add_argument("-o", "--out", default="runs/plots", help="output dir (default runs/plots)")
    args = ap.parse_args()

    paths = list(args.runs) + (sorted(globmod.glob(args.glob)) if args.glob else [])
    valid = []
    for p in paths:
        jp = os.path.join(p, "eval_report.json") if os.path.isdir(p) else p
        if os.path.exists(jp):
            valid.append(p)
        else:
            print(f"skip {p}: no eval_report.json")
    if not valid:
        ap.error("no runs with an eval_report.json (give run dirs or --glob)")

    for w in make_figures(valid, args.out):
        print(f"wrote {w}")


if __name__ == "__main__":
    main()
