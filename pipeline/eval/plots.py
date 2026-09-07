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
                       token-efficiency frontier (up-and-left is better).
- offtarget.png        RQ2's substitution panel: non-termination, wrong
                       termination, unsupported claims and action instability
                       per arm, with the Wilson intervals the report carries.
- stop_reasons.png     why each arm's episodes ended, stacked. Keeps the budget
                       artifact (hit_generation_cap) visually apart from the
                       behaviours, which is the conflation that voided e9-e21.
- turn_profile.png     where an arm's tokens sit inside the trajectory, from the
                       per-turn records in episodes_<split>.jsonl. Absent for
                       runs evaluated before trajectory text was recorded.
- training_curves_<exp>.png  one run's per-step series: reward, length, KL,
                       loss, the optimization panel and the diagnostics
                       (reward SD, clipping, importance-sampling ratio, tool-call
                       failures, step time), plus the per-component raw reward.
                       This is the sheet to read when a run looks wrong. Drawn
                       only for runs that carry a train_log.json (written by
                       training; runs from before that hook existed have none).
- training_overlay.png every arm on one axis, so "diverged at step 40 and stayed"
                       is distinguishable from "wandered around each other".

With `--base`, `dose_response.png` (each metric against the shaping weight, one
series per condition, seed replicates averaged with the replicates still drawn)
and `paired_deltas.png` (the per-episode token difference against the control).

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
import re

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
    # The diagnostic panel. Every key here was already in train_log.json and
    # plotted by nothing, so a run that went wrong silently could only be caught
    # by reading the JSON. reward_std going to zero is reward collapse;
    # clip_ratio rising means the update is being clipped away; an importance
    # ratio drifting off 1.0 is a generation-vs-training mismatch (the vLLM
    # colocate path's own failure mode); tools/failure_frequency rising is the
    # agent breaking its tool calls rather than solving anything; step_time is
    # where a stalled or thrashing run shows up first.
    ("reward_std", "reward SD within a step"),
    ("clip_ratio/region_mean", "clipped policy-ratio fraction"),
    ("sampling/importance_sampling_ratio/mean", "importance sampling ratio"),
    ("tools/call_frequency", "tool calls per completion"),
    ("tools/failure_frequency", "tool-call failures per completion"),
    ("learning_rate", "learning rate"),
    ("step_time", "step time (s)"),
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
    ("reward_std", "reward SD within a step"),
    ("clip_ratio/region_mean", "clipped policy-ratio fraction"),
    ("tools/failure_frequency", "tool-call failures per completion"),
]

# RQ2's off-target panel, in bar order. Each is a rate in [0, 1] whose Wilson
# interval the report carries under <key>_ci_low / <key>_ci_high. A report that
# predates one of them omits the key, and a missing rate is drawn as nothing
# rather than as zero: "no failures" and "never measured" are different claims.
_OFFTARGET_KEYS = [
    ("non_termination_rate", "non-termination"),
    ("wrong_termination_rate", "wrong termination"),
    ("unsupported_claim_rate", "unsupported claim"),
    ("invalid_action_rate", "invalid action"),
    ("repeated_action_rate", "repeated action"),
]

# Per-episode counts, plotted next to the rates. No CI in the report, so they
# get bare bars.
_COUNT_KEYS = [
    ("mean_steps", "mean env steps"),
    ("mean_verification_depth", "mean verification depth"),
]

# Fixed colours so `hit_generation_cap` - a budget artifact, never a behaviour -
# never shares a colour with the reasons that are behaviour.
_STOP_COLORS = {
    "env_done": "#55A868",
    "max_turns": "#DD8452",
    "no_tool_call": "#8172B3",
    "hit_generation_cap": "#C44E52",
}


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
    with open(json_path) as f:  # FileNotFoundError propagates
        report = json.load(f)
    results = report.get("results") or {}
    wanted = split_name
    if wanted is None:
        wanted = (
            _SPLIT if _SPLIT in results or len(results) != 1 else next(iter(results))
        )
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
    """Compact label for axes: the leading handle, e.g. e5-agentic-... -> e5.

    A seed replicate (`<exp>-s43`, the dir name `training.batch --seeds` writes)
    keeps its seed. Without it every seed of one arm collapses onto the same
    label, so a seed sweep renders as several identically-named lines - in the
    one figure whose job is to separate an effect from seed noise.
    """
    parts = (exp_id or "?").split("-")
    if len(parts) > 1 and re.fullmatch(r"s\d+", parts[-1]):
        return f"{parts[0]}-{parts[-1]}"
    return parts[0]


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
    acc_err = np.array(
        [
            _err(
                r["agentic"]["accuracy"],
                r["agentic"]["accuracy_ci_low"],
                r["agentic"]["accuracy_ci_high"],
            )
            for r in reports
        ]
    ).T
    ax_acc.bar(x, acc, color="#4C72B0")
    ax_acc.errorbar(x, acc, yerr=acc_err, fmt="none", ecolor="black", capsize=4)
    ax_acc.set_xticks(x), ax_acc.set_xticklabels(labels)
    ax_acc.set_ylim(0, 1), ax_acc.set_ylabel("success rate")
    ax_acc.set_title("Success rate (Wilson 95%)")

    tok = [r["agentic"]["mean_token_count"] for r in reports]
    tok_err = np.array(
        [
            _err(
                r["agentic"]["mean_token_count"],
                r["agentic"]["mean_token_count_ci_low"],
                r["agentic"]["mean_token_count_ci_high"],
            )
            for r in reports
        ]
    ).T
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
            ax.hist(
                c, bins=20, alpha=0.6, color="#55A868", label=f"correct (n={len(c)})"
            )
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
            mean_c,
            acc,
            xerr=[[max(0.0, mean_c - lo)], [max(0.0, hi - mean_c)]],
            yerr=[
                [max(0.0, acc - r["agentic"]["accuracy_ci_low"])],
                [max(0.0, r["agentic"]["accuracy_ci_high"] - acc)],
            ],
            fmt="o",
            capsize=3,
            markersize=8,
        )
        ax.annotate(
            _short(r["experiment_id"]),
            (mean_c, acc),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=9,
        )
    ax.set_xlabel("mean tokens on correct episodes")
    ax.set_ylabel("success rate")
    ax.set_title("Token-efficiency frontier (up-and-left is better)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _grouped_bars(ax, reports, keys, with_ci):
    """Grouped bars: one x group per present metric, one bar per arm.

    A metric no report carries gets no group, and an arm whose report lacks a
    present metric gets no bar in that group - never a zero-height one, which
    would read as a measured absence of the behaviour. Returns False when
    nothing was drawn.
    """
    labels = [_short(r["experiment_id"]) for r in reports]
    present = [
        (k, lbl)
        for k, lbl in keys
        if any(r["agentic"].get(k) is not None for r in reports)
    ]
    if not present:
        return False
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    width = 0.8 / max(1, len(reports))
    x = np.arange(len(present))
    for j, r in enumerate(reports):
        off = (j - (len(reports) - 1) / 2) * width
        xs, ys, errs = [], [], []
        for i, (k, _) in enumerate(present):
            v = r["agentic"].get(k)
            if v is None:
                continue
            lo, hi = r["agentic"].get(f"{k}_ci_low"), r["agentic"].get(f"{k}_ci_high")
            xs.append(x[i] + off)
            ys.append(float(v))
            errs.append(
                _err(float(v), float(lo), float(hi))
                if with_ci and lo is not None and hi is not None
                else [0.0, 0.0]
            )
        if not xs:
            continue
        ax.bar(xs, ys, width=width, color=colors[j % len(colors)], label=labels[j])
        if with_ci:
            ax.errorbar(
                xs,
                ys,
                yerr=np.array(errs).T,
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in present], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    return True


def plot_offtarget(reports, fig=None):
    """RQ2's off-target panel: the substitution rates per arm, and the counts.

    RQ1 asks what the shaping buys; RQ2 asks what it costs somewhere the reward
    was not looking, and every rate that answers it was already in the report
    and plotted by nothing - so the question had to be read out of a table. The
    rates share one axis on purpose: they are all "fraction of episodes", so the
    comparison that matters is between arms within a rate and between rates
    within an arm, and separate panels would hide the second one.

    Returns None when no report carries any off-target rate (a run evaluated
    before termination tracking existed).
    """
    if fig is None:
        fig = plt.figure(figsize=(max(9.0, 1.1 * len(reports) + 7.0), 4.4))
    ax_rate, ax_cnt = fig.subplots(1, 2)
    if not _grouped_bars(ax_rate, reports, _OFFTARGET_KEYS, with_ci=True):
        plt.close(fig)
        return None
    ax_rate.set_ylabel("fraction of episodes")
    ax_rate.set_title("Off-target rates (Wilson 95%)")
    if ax_rate.get_legend_handles_labels()[0]:
        ax_rate.legend(fontsize=8, ncol=2)
    if _grouped_bars(ax_cnt, reports, _COUNT_KEYS, with_ci=False):
        ax_cnt.set_ylabel("count per episode")
        ax_cnt.set_title("Trajectory shape")
    else:
        ax_cnt.set_visible(False)
    fig.suptitle("Off-target behaviour (RQ2)")
    fig.tight_layout()
    return fig


def plot_stop_reasons(reports, fig=None):
    """Stacked bars: why each arm's episodes ended, as a fraction.

    `hit_generation_cap` is a budget artifact and every other reason is a
    behaviour. Accuracy cannot tell them apart, and reading a truncation as a
    wrong answer is exactly what made the e9-e21 sweep uninterpretable. Stacked
    rather than grouped because the reasons partition the episodes: the quantity
    of interest is the composition, not four independent bars.

    Returns None when no report carries stop reasons.
    """
    counts = [r["agentic"].get("stop_reasons") or {} for r in reports]
    seen = {k for c in counts for k in c}
    if not seen:
        return None
    order = [k for k in _STOP_COLORS if k in seen] + sorted(
        k for k in seen if k not in _STOP_COLORS
    )
    if fig is None:
        fig = plt.figure(figsize=(max(6.0, 1.1 * len(reports) + 3.5), 4.2))
    ax = fig.subplots(1, 1)
    x = np.arange(len(reports))
    bottom = np.zeros(len(reports))
    for reason in order:
        vals = np.array(
            [c.get(reason, 0) / max(1, sum(c.values())) for c in counts], dtype=float
        )
        ax.bar(x, vals, bottom=bottom, label=reason, color=_STOP_COLORS.get(reason))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([_short(r["experiment_id"]) for r in reports])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of episodes")
    ax.set_title("Why episodes ended")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig


def plot_turn_profile(series, fig=None):
    """Where an arm's tokens sit inside the trajectory.

    `series` is [(label, [[turn, ...], ...]), ...]: one list of per-turn records
    per episode, as episodes_<split>.jsonl stores them. The aggregate report says
    an arm got shorter; it cannot say whether the tokens came out of the think
    block, out of the spoken content, or out of the later turns - and "shorter
    because it stopped verifying" and "shorter because it stopped rambling" are
    opposite answers to RQ2.

    Left: mean tokens at each turn index, averaged only over the episodes that
    reached that turn, so an arm that finishes earlier ends its curve earlier
    instead of being dragged toward zero. Right: mean per-episode reasoning
    against content tokens. They are drawn side by side rather than stacked
    because they do not sum to the episode total - the template framing and the
    tool-call JSON are in neither.

    Returns None when no episode carries turn records.
    """
    series = [(lbl, [t for t in eps if t]) for lbl, eps in series]
    series = [(lbl, eps) for lbl, eps in series if eps]
    if not series:
        return None
    if fig is None:
        fig = plt.figure(figsize=(11.5, 4.2))
    ax_turn, ax_split = fig.subplots(1, 2)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    width = 0.8 / len(series)
    has_split = False
    for j, (lbl, eps) in enumerate(series):
        c = colors[j % len(colors)]
        xs, ys = [], []
        for i in range(max(len(t) for t in eps)):
            vals = [float(t[i].get("n_tokens", 0)) for t in eps if len(t) > i]
            if vals:
                xs.append(i + 1)
                ys.append(float(np.mean(vals)))
        ax_turn.plot(xs, ys, "o-", color=c, label=lbl)
        if not any(x.get("n_reasoning_tokens") is not None for t in eps for x in t):
            continue  # counted only when a tokenizer was passed
        has_split = True
        reas = [sum(float(x.get("n_reasoning_tokens") or 0) for x in t) for t in eps]
        cont = [sum(float(x.get("n_content_tokens") or 0) for x in t) for t in eps]
        off = (j - (len(series) - 1) / 2) * width
        ax_split.bar(
            [off, 1 + off],
            [float(np.mean(reas)), float(np.mean(cont))],
            width=width,
            color=c,
            label=lbl,
        )
    ax_turn.set_xlabel("turn")
    ax_turn.set_ylabel("mean tokens in the turn")
    ax_turn.set_title("Token profile across the trajectory")
    ax_turn.grid(True, alpha=0.3)
    ax_turn.legend(fontsize=8)
    if has_split:
        ax_split.set_xticks([0, 1])
        ax_split.set_xticklabels(["reasoning", "content"])
        ax_split.set_ylabel("mean tokens per episode")
        ax_split.set_title("Where the tokens are")
        ax_split.grid(True, axis="y", alpha=0.3)
    else:
        ax_split.set_visible(False)
    fig.suptitle("Per-turn trajectory profile")
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
    comp_keys = sorted(
        {
            k
            for e in log_history
            for k in e
            if k.startswith("reward/") and k.endswith("/raw_mean")
        }
    )
    total = len(panels) + (1 if comp_keys else 0)
    if total == 0:
        return None
    ncols = min(3, total)
    nrows = (total + ncols - 1) // ncols
    if fig is None:
        fig = plt.figure(figsize=(5.4 * ncols, 3.2 * nrows))
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
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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
    panels = [(k, lbl) for k, lbl in keys if any(_series(log, k)[1] for _, log in logs)]
    # Shaped components only: EnvReward is the task signal and already has its
    # own panel, so overlaying it here would just repeat it.
    comp_keys = sorted(
        {
            k
            for _, log in logs
            for e in log
            for k in e
            if k.startswith("reward/")
            and k.endswith("/contrib_l1")
            and "EnvReward" not in k
        }
    )
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
                ax.plot(
                    sx,
                    sy,
                    color=colors[j % len(colors)],
                    label=f"{lbl}: {k.split('/')[1]}",
                )
        ax.set_xlabel("step"), ax.set_ylabel("|contribution| to the advantage")
        ax.set_title("shaped component contribution"), ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training dynamics across arms")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
            # Several rows can share one lambda: that is a seed replicate of the
            # same arm. They are averaged into the line and also drawn
            # individually, so the between-seed spread stays visible instead of
            # disappearing into a mean.
            pts = {}
            for r in rows:
                if r["condition"] != cond:
                    continue
                v, err = _panel_value(r, key)
                if v is not None:
                    pts.setdefault(r["lam"], []).append((v, err))
            if not pts:
                continue
            c = colors[j % len(colors)]
            xs = sorted(pts)
            ax.plot(
                xs,
                [float(np.mean([v for v, _ in pts[x]])) for x in xs],
                "o-",
                color=c,
                label=cond,
                markersize=6,
            )
            for x in xs:
                vals = pts[x]
                if len(vals) > 1:
                    # With replicates the seed spread is the honest error bar, so
                    # the per-run CI is not drawn on top of it.
                    ax.plot(
                        [x] * len(vals),
                        [v for v, _ in vals],
                        "o",
                        color=c,
                        markersize=4,
                        alpha=0.45,
                    )
                elif vals[0][1] is not None:
                    ax.errorbar(
                        [x],
                        [vals[0][0]],
                        yerr=vals[0][1],
                        fmt="none",
                        ecolor=c,
                        capsize=4,
                    )
        if refs and key in (refs or {}):
            rlabel, rval = refs[key]
            ax.axhline(rval, linestyle="--", color="#666666", linewidth=1)
            ax.annotate(
                rlabel,
                (0.02, rval),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                color="#666666",
                va="bottom",
            )
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
        ax.bar(
            np.arange(d.size), d, color=["#55A868" if v <= 0 else "#C44E52" for v in d]
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(
            float(np.median(d)),
            color="#4C72B0",
            linestyle="--",
            linewidth=1,
            label=f"median {np.median(d):.0f}",
        )
        ax.set_title(f"{label} (n={d.size})")
        ax.set_xlabel("episodes, sorted"), ax.set_ylabel("tokens vs control")
        ax.legend(fontsize=8), ax.grid(True, alpha=0.3)
    fig.suptitle("Per-episode token difference on jointly correct episodes")
    fig.tight_layout()
    return fig


def _episode_turns(run_dir: str, split: str) -> list:
    """Per-episode turn records from episodes_<split>.jsonl.

    [] when the file is absent or when the run was evaluated before trajectory
    text was recorded, so an old campaign silently gets no turn figure rather
    than an empty one.
    """
    path = os.path.join(run_dir, f"episodes_{split}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns = json.loads(line).get("turns")
            if turns:
                out.append(turns)
    return out


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
        reports, kept = [], []
        for p in report_paths:
            try:
                reports.append(load_report(p, sp))
                kept.append(p)
            except (ValueError, OSError) as exc:
                print(f"skip {p} [{sp}]: {exc}")
        if not reports:
            continue
        tag = f"_{sp}" if len(splits) > 1 else ""
        for plot_fn, name in (
            (plot_comparison, "comparison"),
            (plot_distributions, "distributions"),
            (plot_efficiency, "efficiency"),
            (plot_offtarget, "offtarget"),
            (plot_stop_reasons, "stop_reasons"),
        ):
            # Pre-agentic dataset-mode reports lack the CI fields these read.
            # One such report in a glob used to abort the whole batch; say what
            # broke and keep drawing the rest. A figure whose whole signal is
            # missing from every report returns None and is simply not written.
            try:
                fig = plot_fn(reports)
                if fig is not None:
                    _save(fig, f"{name}{tag}.png")
            except (KeyError, TypeError, ValueError) as exc:
                print(f"skip {name}{tag}.png: {type(exc).__name__}: {exc}")

        # Trajectory text lives beside the report, not inside it.
        turns = []
        for p, r in zip(kept, reports):
            eps = _episode_turns(p if os.path.isdir(p) else os.path.dirname(p), sp)
            if eps:
                turns.append((_short(r["experiment_id"]), eps))
        if turns:
            fig = plot_turn_profile(turns)
            if fig is not None:
                _save(fig, f"turn_profile{tag}.png")

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


def make_dose_figures(
    base,
    runs,
    out_dir,
    dpi=130,
    ref=None,
    family=None,
    held_out="held_out",
    shifted="shifted",
):
    """Dose-response and per-episode paired figures, against a control run.

    Both need the pairing, so the numbers come from `eval.paired` (which owns the
    statistics) and only the drawing happens here. Returns the written paths.
    """
    from eval import paired

    os.makedirs(out_dir, exist_ok=True)
    written = []
    arms = [r for r in runs if os.path.normpath(r) != os.path.normpath(base)]
    rows = paired.dose_rows(
        base, arms, held_out=held_out, shifted=shifted, family=family
    )
    panels = [
        ("losses", f"{held_out} losses vs control (paired)"),
        ("median_dtok", "paired median token diff, correct episodes"),
        ("shifted_nonterm", f"{shifted} non-termination rate"),
    ]
    if family:
        panels.append(("family_acc", f"{shifted} {family} accuracy"))
    else:
        # Every family the shifted split drew from gets a panel. Naming one on
        # the command line was the old behaviour, and it left the other families
        # of a three-family split out of the figure entirely.
        panels += [
            (k, f"{shifted} {k.split(':', 1)[1]} accuracy")
            for k in sorted({k for r in rows for k in r if k.startswith("family_acc:")})
        ]
    # The control's config supplies the family mapping for a base-model run,
    # which never trained and so froze no config of its own.
    refs = (
        paired.dose_refs(ref, shifted, family, config=paired.load_config(base))
        if ref
        else None
    )
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
    ap.add_argument(
        "-o", "--out", default="runs/plots", help="output dir (default runs/plots)"
    )
    ap.add_argument(
        "--split",
        default=None,
        help="render only this eval split (default: one figure set per split)",
    )
    ap.add_argument(
        "--base",
        help="control run dir; adds the dose-response and "
        "per-episode paired figures",
    )
    ap.add_argument(
        "--ref", help="reference run dir for the dashed E0 line (with --base)"
    )
    ap.add_argument(
        "--family",
        help="restrict the per-family dose panel to this task family "
        "(default: one panel per family in the shifted split)",
    )
    ap.add_argument(
        "--held-out", default="held_out", help="paired split (default held_out)"
    )
    ap.add_argument(
        "--shifted", default="shifted", help="off-target split (default shifted)"
    )
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
        for w in make_dose_figures(
            args.base,
            valid,
            args.out,
            ref=args.ref,
            family=args.family,
            held_out=args.held_out,
            shifted=args.shifted,
        ):
            print(f"wrote {w}")


if __name__ == "__main__":
    main()
