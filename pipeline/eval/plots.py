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
    # The optimization panel. Every one of these was already written to
    # train_log.json and plotted by nothing, which is why a campaign's "is this
    # signal or noise" question had to be answered from the aggregate report
    # alone: frac_reward_zero_std is the gradient-share caveat, and grad_norm and
    # entropy are where a collapsing policy shows up before the reward does.
    ("frac_reward_zero_std", "prompt-groups with zero reward variance"),
    ("grad_norm", "grad norm"),
    ("entropy", "entropy"),
]

# Cross-arm overlay panels, in order. Deliberately fewer than _CURVE_KEYS: this
# figure answers "did the arms diverge, and when", so it carries the series whose
# between-arm difference is interpretable, not every series that exists.
_OVERLAY_KEYS = [
    ("reward/EnvReward/raw_mean", "env reward (raw mean)"),
    ("completions/mean_length", "mean completion length (tokens)"),
    ("frac_reward_zero_std", "groups with zero reward variance"),
    ("kl", "KL"),
    ("entropy", "entropy"),
    ("grad_norm", "grad norm"),
]


def load_report(path: str, split_name: str | None = None) -> dict:
    """Load an eval report from a run dir or a JSON file.

    Returns {"experiment_id", "agentic" (the split metrics dict), "samples"}.
    Raises FileNotFoundError if the report is missing, ValueError if the
    requested split is absent.

    `split_name` selects one split of a multi-split report. Unnamed, it takes
    "agentic" (the single-split shape every earlier run wrote), or the only split
    present if a report uses a protocol name instead. A multi-split report with
    no name given raises rather than guessing: plotting the held-out split of one
    arm against the shifted split of another, silently, is worse than an error.
    """
    json_path = os.path.join(path, "eval_report.json") if os.path.isdir(path) else path
    with open(json_path) as f:                       # FileNotFoundError propagates
        report = json.load(f)
    results = report.get("results") or {}
    wanted = split_name
    if wanted is None:
        wanted = _SPLIT if _SPLIT in results or len(results) != 1 else next(iter(results))
    split = results.get(wanted)
    if split is None:
        raise ValueError(
            f"{json_path}: no {wanted!r} split in results (have: {sorted(results)})"
        )
    return {
        "experiment_id": report.get("experiment_id", "?"),
        "split_name": wanted,
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


def _smooth(xs, ys, window):
    """Bucket means of `window` consecutive points, at each bucket's mean x.

    Raw per-step series on a 150-step run are noisy enough that two arms'
    trajectories overlap visually even when their levels differ; the bucket mean
    is drawn on top of the raw line so both the trend and the scatter it came
    from stay visible.
    """
    if window <= 1 or len(ys) < window:
        return xs, ys
    n = (len(ys) // window) * window
    bx = np.asarray(xs[:n], dtype=float).reshape(-1, window).mean(axis=1)
    by = np.asarray(ys[:n], dtype=float).reshape(-1, window).mean(axis=1)
    return bx.tolist(), by.tolist()


def plot_training_overlay(logs, keys=None, smooth=10, fig=None):
    """One panel per series, every arm overlaid on a shared x-axis.

    `logs` is [(label, log_history), ...]. This is the figure the per-run
    `training_curves_<exp>.png` cannot be: separate figures per arm answer "what
    did this run do", never "did the arms diverge, and at which step" - and that
    second question is what separates a reward effect (arms split early and stay
    split) from optimization noise (they wander around each other).

    A final panel overlays the shaped component's `contrib_l1` where a run has
    one, so an arm whose penalty goes inert mid-run is visible as a decaying
    line rather than as an unexplained flat result. Returns None when no log
    carries any of the keys.
    """
    keys = _OVERLAY_KEYS if keys is None else keys
    logs = [(lbl, log) for lbl, log in logs if log]
    panels = [(k, lbl) for k, lbl in keys
              if any(_series(log, k)[1] for _, log in logs)]
    # Shaped components only: EnvReward is the task signal and already has its
    # own panel, so overlaying it here would just repeat it.
    comp_keys = sorted({k for _, log in logs for e in log for k in e
                        if k.startswith("reward/") and k.endswith("/contrib_l1")
                        and "EnvReward" not in k})
    total = len(panels) + (1 if comp_keys else 0)
    if total == 0:
        return None
    ncols = min(3, total)
    nrows = (total + ncols - 1) // ncols
    if fig is None:
        fig = plt.figure(figsize=(5.2 * ncols, 3.4 * nrows))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (key, label) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        for j, (lbl, log) in enumerate(logs):
            xs, ys = _series(log, key)
            if not ys:
                continue
            c = colors[j % len(colors)]
            ax.plot(xs, ys, color=c, alpha=0.22, linewidth=0.8)
            sx, sy = _smooth(xs, ys, smooth)
            ax.plot(sx, sy, color=c, linewidth=1.8, label=lbl)
        ax.set_xlabel("step"), ax.set_ylabel(label), ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if i == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    if comp_keys:
        ax = fig.add_subplot(nrows, ncols, total)
        for j, (lbl, log) in enumerate(logs):
            for k in comp_keys:
                xs, ys = _series(log, k)
                if not ys:
                    continue
                sx, sy = _smooth(xs, ys, smooth)
                ax.plot(sx, sy, color=colors[j % len(colors)],
                        label=f"{lbl}: {k.split('/')[1]}")
        ax.set_xlabel("step"), ax.set_ylabel("|contribution| to the advantage")
        ax.set_title("shaped component contribution"), ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training dynamics across arms")
    fig.tight_layout()
    return fig


def _panel_value(row, key):
    """(value, yerr) for a panel cell. A (v, lo, hi) triple carries a CI."""
    v = row.get(key)
    if isinstance(v, (tuple, list)) and len(v) == 3:
        return float(v[0]), [[max(0.0, v[0] - v[1])], [max(0.0, v[2] - v[0])]]
    return (None if v is None else float(v)), None


def plot_dose_response(rows, panels, refs=None, fig=None):
    """Dose-response: each panel is one metric against the shaping weight lambda.

    `rows` are dicts with `condition` (the series, e.g. "E2 cosine"), `lam`, and
    the panel keys - each either a scalar or a (value, ci_low, ci_high) triple.
    `refs` is {panel_key: (label, value)}, drawn as a dashed horizontal line: the
    E0 base model, which has no lambda and so cannot be a point on these axes.

    The lambda=0 point is the task-reward-only control, so every series starts
    from the same place by construction and a panel reads as "what does turning
    this knob up buy, and what does it cost".
    """
    n = len(panels)
    ncols = min(2, n) or 1
    nrows = (n + ncols - 1) // ncols
    if fig is None:
        fig = plt.figure(figsize=(5.6 * ncols, 3.8 * nrows))
    conditions = []
    for r in rows:
        if r["condition"] not in conditions:
            conditions.append(r["condition"])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (key, label) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        for j, cond in enumerate(conditions):
            pts = sorted((r for r in rows if r["condition"] == cond),
                         key=lambda r: r["lam"])
            xs, ys, errs = [], [], []
            for r in pts:
                v, err = _panel_value(r, key)
                if v is None:
                    continue
                xs.append(r["lam"]), ys.append(v)
                errs.append(err)
            if not xs:
                continue
            c = colors[j % len(colors)]
            ax.plot(xs, ys, "o-", color=c, label=cond, markersize=6)
            for x, y, e in zip(xs, ys, errs):
                if e is not None:
                    ax.errorbar([x], [y], yerr=e, fmt="none", ecolor=c, capsize=4)
        if refs and key in (refs or {}):
            rlabel, rval = refs[key]
            ax.axhline(rval, linestyle="--", color="#666666", linewidth=1)
            ax.annotate(rlabel, (0.02, rval), xycoords=("axes fraction", "data"),
                        fontsize=8, color="#666666", va="bottom")
        ax.set_xlabel("shaping weight lambda"), ax.set_ylabel(label)
        ax.set_title(label), ax.grid(True, alpha=0.3)
        if i == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Dose-response: what the shaping weight buys and costs")
    fig.tight_layout()
    return fig


def plot_paired_deltas(series, fig=None):
    """Per-episode token difference against the control, sorted, one panel per arm.

    `series` is [(label, [diff, ...]), ...] over the JOINTLY CORRECT episodes.
    A summary median cannot distinguish "every episode got a bit shorter" from
    "a handful collapsed and the rest did not move", and those are different
    claims about what the reward did. The zero line is the control.
    """
    series = [(lbl, list(d)) for lbl, d in series if len(d)]
    if not series:
        return None
    ncols = min(3, len(series))
    nrows = (len(series) + ncols - 1) // ncols
    if fig is None:
        fig = plt.figure(figsize=(4.6 * ncols, 3.4 * nrows))
    for i, (label, diffs) in enumerate(series):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        d = np.sort(np.asarray(diffs, dtype=float))
        ax.bar(np.arange(d.size), d,
               color=["#55A868" if v <= 0 else "#C44E52" for v in d])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(float(np.median(d)), color="#4C72B0", linestyle="--", linewidth=1,
                   label=f"median {np.median(d):.0f}")
        ax.set_title(f"{label} (n={d.size})")
        ax.set_xlabel("episodes, sorted"), ax.set_ylabel("tokens vs control")
        ax.legend(fontsize=8), ax.grid(True, alpha=0.3)
    fig.suptitle("Per-episode token difference on jointly correct episodes")
    fig.tight_layout()
    return fig


def _report_splits(path: str) -> list[str]:
    """Split names a report carries, in file order."""
    jp = os.path.join(path, "eval_report.json") if os.path.isdir(path) else path
    with open(jp) as f:
        return list((json.load(f).get("results") or {}))


def make_figures(report_paths, out_dir, dpi=130, split=None):
    """Render all figures for the given run dirs / report paths into out_dir.

    Without `split`, renders one figure set per split name found across the
    reports. A protocol report (held_out + shifted) used to raise here, so the
    browsergym campaign could produce no figures at all; a single-split report
    still writes the unsuffixed filenames every earlier run used.

    Returns the list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if split is not None:
        splits = [split]
    else:
        splits = []
        for p in report_paths:
            for s in _report_splits(p):
                if s not in splits:
                    splits.append(s)
        splits = splits or [_SPLIT]

    for sp in splits:
        # A run missing this split is skipped, not fatal: a mixed glob of
        # single-split and protocol runs used to die on the first mismatch and
        # lose the figures for the runs that would have worked.
        reports = []
        for p in report_paths:
            try:
                reports.append(load_report(p, sp))
            except (ValueError, OSError) as exc:
                print(f"skip {p} [{sp}]: {exc}")
        if not reports:
            continue
        tag = f"_{sp}" if len(splits) > 1 else ""
        for plot_fn, name in ((plot_comparison, "comparison"),
                              (plot_distributions, "distributions"),
                              (plot_efficiency, "efficiency")):
            # Pre-agentic dataset-mode reports lack the CI fields these read.
            # One such report in a glob used to abort the whole batch; say what
            # broke and keep drawing the rest.
            try:
                _save(plot_fn(reports), f"{name}{tag}.png")
            except (KeyError, TypeError, ValueError) as exc:
                print(f"skip {name}{tag}.png: {type(exc).__name__}: {exc}")

    # Training curves are per-run, not per-split - drawn once, outside the loop.
    logs = []
    for p in report_paths:
        run_dir = p if os.path.isdir(p) else os.path.dirname(p)
        train_log = os.path.join(run_dir, "train_log.json")
        if not os.path.exists(train_log):
            continue
        with open(train_log) as f:
            log = json.load(f)
        logs.append((_short(os.path.basename(run_dir)), log))
        fig = plot_training_curves(log)
        if fig is not None:
            _save(fig, f"training_curves_{_short(os.path.basename(run_dir))}.png")
    # The cross-arm view. Only meaningful from two runs up, which is also when
    # the per-run figures stop being able to answer the question.
    if len(logs) > 1:
        fig = plot_training_overlay(logs)
        if fig is not None:
            _save(fig, "training_overlay.png")
    return written


def make_dose_figures(base, runs, out_dir, dpi=130, ref=None, family=None,
                      held_out="held_out", shifted="shifted"):
    """Dose-response and per-episode paired figures, against a control run.

    Both need the pairing, so the numbers come from `eval.paired` (which owns the
    statistics) and only the drawing happens here. Returns the written paths.
    """
    from eval import paired

    os.makedirs(out_dir, exist_ok=True)
    written = []
    arms = [r for r in runs if os.path.normpath(r) != os.path.normpath(base)]
    rows = paired.dose_rows(base, arms, held_out=held_out, shifted=shifted, family=family)
    panels = [
        ("losses", f"{held_out} losses vs control (paired)"),
        ("median_dtok", "paired median token diff, correct episodes"),
        ("shifted_nonterm", f"{shifted} non-termination rate"),
    ]
    if family:
        panels.append(("family_acc", f"{shifted} {family} accuracy"))
    # The control's config supplies the family mapping for a base-model run,
    # which never trained and so froze no config of its own.
    refs = (paired.dose_refs(ref, shifted, family, config=paired.load_config(base))
            if ref else None)
    fig = plot_dose_response(rows, panels, refs)
    path = os.path.join(out_dir, "dose_response.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    base_eps = paired.load_episodes(base, held_out)
    series = []
    for arm in arms:
        try:
            arm_eps = paired.load_episodes(arm, held_out)
        except OSError:
            continue
        c = paired.compare(base_eps, arm_eps, arm_id=paired._short(arm))
        series.append((paired._short(arm), c.token_diffs))
    fig = plot_paired_deltas(series)
    if fig is not None:
        path = os.path.join(out_dir, "paired_deltas.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description="Render thesis figures from eval reports.")
    ap.add_argument("runs", nargs="*", help="run dirs or eval_report.json paths")
    ap.add_argument("--glob", help="glob for run dirs, e.g. 'runs/e*'")
    ap.add_argument("-o", "--out", default="runs/plots", help="output dir (default runs/plots)")
    ap.add_argument("--split", default=None,
                    help="render only this eval split (default: one figure set per split)")
    ap.add_argument("--base", help="control run dir; adds the dose-response and "
                                   "per-episode paired figures")
    ap.add_argument("--ref", help="reference run dir for the dashed E0 line (with --base)")
    ap.add_argument("--family", help="task family whose accuracy gets a dose panel")
    ap.add_argument("--held-out", default="held_out", help="paired split (default held_out)")
    ap.add_argument("--shifted", default="shifted", help="off-target split (default shifted)")
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

    for w in make_figures(valid, args.out, split=args.split):
        print(f"wrote {w}")
    if args.base:
        for w in make_dose_figures(args.base, valid, args.out, ref=args.ref,
                                   family=args.family, held_out=args.held_out,
                                   shifted=args.shifted):
            print(f"wrote {w}")


if __name__ == "__main__":
    main()
