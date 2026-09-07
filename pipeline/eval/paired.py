"""Paired arm-vs-arm statistics over episode records, plus training noise floors.

Every comparison in `runs/*_findings.md` so far was computed ad hoc in a throwaway
snippet: the flip counts, the exact McNemar p, the paired token median and its
bootstrap CI. Nothing re-derived them, so a number in the thesis could not be
checked against the data it came from. This module is that path.

Two things it answers:

- **Is the arm different from its control?** `compare()` pairs the two runs
  seed-for-seed (never mean-vs-mean: unpaired means shift when the treatment arm
  solves questions the control missed, which are usually the long ones, so the
  mean can move the wrong way while the paired median falls), then reports the
  win/loss flips with an exact McNemar p and the token difference on the jointly
  correct episodes with a sign test and a paired bootstrap CI.
- **Is a difference bigger than the run's own noise?** `training_tail()` takes the
  last N steps of a train_log and reports each series' mean, SD and slope. A
  between-arm delta smaller than the within-arm SD is not a result, and a series
  still sloping at the last step means the arms differ in training progress as
  well as in reward.

Contract: the statistics are pure functions of plain lists; only `load_episodes`,
`load_train_log` and the CLI touch disk. Nothing here re-scores an episode -
`correct` is read as the eval wrote it, so no threshold is silently redefined.

CLI (data to stdout, diagnostics to stderr):
  python -m eval.paired --base runs/e30-... runs/e31-... --split held_out
  python -m eval.paired --base runs/e30-... runs/e3[1-6]-* --by-family -o out.md
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy.stats import binomtest

# Replicates for the paired bootstrap. 20k puts the Monte-Carlo error on a
# percentile bound at ~0.3% of the interval, which is below the resolution any
# token median is quoted at, and the draw is vectorized so it costs milliseconds.
N_BOOTSTRAP = 20_000
# Fixed so a re-run reproduces a published CI exactly. Changing it changes every
# interval this module has ever printed.
BOOTSTRAP_SEED = 0
# Steps of the training tail used for the noise floor. 30 of 150 steps is long
# enough for a stable SD and short enough to stay inside the converged regime.
TAIL_STEPS = 30


@dataclass
class PairedComparison:
    """One arm against one control, on the episodes both actually ran."""

    base_id: str
    arm_id: str
    split: str
    family: str | None = None  # None = all families pooled
    n_paired: int = 0
    base_correct: int = 0
    arm_correct: int = 0
    wins: int = 0  # arm correct, control wrong
    losses: int = 0  # control correct, arm wrong
    mcnemar_p: float | None = None
    n_joint_correct: int = 0
    median_token_diff: float | None = None
    median_ci_low: float | None = None
    median_ci_high: float | None = None
    mean_token_diff: float | None = None
    sign_p: float | None = None
    base_nonterm: float | None = None
    arm_nonterm: float | None = None
    # arm minus control token count, per jointly-correct seed, seed-sorted. The
    # per-episode view (is compression uniform, or a few episodes?) reads this.
    token_diffs: list[float] = field(default_factory=list)


def exact_binomial_p(k: int, n: int) -> float:
    """Two-sided exact binomial p for k successes in n trials at p=0.5.

    The engine under both the McNemar test on flips and the sign test on token
    differences: with n discordant pairs, k of them in one direction, the null is
    a fair coin. Exact rather than the chi-square approximation because the
    interesting cells here are tiny (0 wins against 14 losses) and the
    approximation is unreliable there. n=0 (no discordant pairs at all) is not
    evidence of anything, so it returns 1.0 rather than raising.
    """
    if n == 0:
        return 1.0
    return float(binomtest(int(k), int(n), 0.5).pvalue)


def sign_test_p(diffs) -> float:
    """Two-sided sign test on paired differences. Zeros are dropped (ties carry
    no directional information), so an all-ties vector returns 1.0."""
    d = np.asarray(list(diffs), dtype=float)
    nz = d[d != 0]
    return exact_binomial_p(int((nz < 0).sum()), int(nz.size))


def bootstrap_median_ci(
    diffs, n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED, ci: float = 0.95
):
    """(median, lo, hi) for the median of paired differences.

    Resamples the PAIRS, not the two arms independently: the pairing is the whole
    point of the design, and resampling the arms separately would throw it away
    and widen the interval for no reason. Returns (None, None, None) for an empty
    input.
    """
    d = np.asarray(list(diffs), dtype=float)
    if d.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_bootstrap, d.size))
    boot = np.median(d[idx], axis=1)
    alpha = (1 - ci) / 2
    return (
        float(np.median(d)),
        float(np.percentile(boot, 100 * alpha)),
        float(np.percentile(boot, 100 * (1 - alpha))),
    )


def compare(
    base_episodes: dict,
    arm_episodes: dict,
    *,
    base_id: str = "base",
    arm_id: str = "arm",
    split: str = "?",
    family: str | None = None,
) -> PairedComparison:
    """Pair two runs' episodes on their seeds and compute the paired statistics.

    Both arguments map seed -> episode record (`load_episodes` produces them).
    Only seeds present in both are used; a seed one arm never ran contributes
    nothing rather than being counted as a failure. Episodes are compared as the
    eval scored them - this never re-thresholds a reward.
    """
    seeds = sorted(set(base_episodes) & set(arm_episodes))
    wins = losses = 0
    diffs = []
    for s in seeds:
        b, a = base_episodes[s], arm_episodes[s]
        if a["correct"] and not b["correct"]:
            wins += 1
        elif b["correct"] and not a["correct"]:
            losses += 1
        if b["correct"] and a["correct"]:
            diffs.append(float(a["n_tokens"] - b["n_tokens"]))
    med, lo, hi = bootstrap_median_ci(diffs)
    return PairedComparison(
        base_id=base_id,
        arm_id=arm_id,
        split=split,
        family=family,
        n_paired=len(seeds),
        base_correct=sum(1 for s in seeds if base_episodes[s]["correct"]),
        arm_correct=sum(1 for s in seeds if arm_episodes[s]["correct"]),
        wins=wins,
        losses=losses,
        # k = wins among the discordant pairs: the McNemar statistic is exactly
        # the fair-coin question about which direction the flips went.
        mcnemar_p=exact_binomial_p(wins, wins + losses),
        n_joint_correct=len(diffs),
        median_token_diff=med,
        median_ci_low=lo,
        median_ci_high=hi,
        mean_token_diff=float(np.mean(diffs)) if diffs else None,
        sign_p=sign_test_p(diffs) if diffs else None,
        base_nonterm=_nonterm_rate(base_episodes, seeds),
        arm_nonterm=_nonterm_rate(arm_episodes, seeds),
        token_diffs=diffs,
    )


def _nonterm_rate(episodes: dict, seeds) -> float | None:
    """Fraction of the paired episodes that never reported env_done. None when
    the records predate termination tracking (absent, not zero - a missing field
    read as "no failures" is the confound this panel exists to avoid)."""
    known = [episodes[s] for s in seeds if episodes[s].get("terminated") is not None]
    if not known:
        return None
    return sum(1 for e in known if not e["terminated"]) / len(known)


def stop_reason_transitions(base_episodes: dict, arm_episodes: dict) -> dict:
    """(control stop_reason, arm stop_reason) -> count, over the paired seeds.

    Where an arm's episodes moved: env_done -> max_turns is a behavior change,
    env_done -> hit_generation_cap is a budget artifact, and the aggregate
    stop-reason counts in the report cannot tell those apart because they do not
    follow individual episodes.
    """
    out: dict[tuple, int] = {}
    for s in sorted(set(base_episodes) & set(arm_episodes)):
        key = (base_episodes[s].get("stop_reason"), arm_episodes[s].get("stop_reason"))
        out[key] = out.get(key, 0) + 1
    return out


def training_tail(log_history, keys=None, window: int = TAIL_STEPS) -> dict:
    """Per-series mean, SD and slope over the last `window` logged steps.

    The noise floor a between-arm delta has to clear: each step already averages
    batch_size * n_rollouts rollouts, so the step-to-step SD of a converged run is
    a free estimate of how much a number moves for no reason at all. `slope` (OLS
    per step over the same window) says whether the series had converged: an arm
    still climbing at the last step differs from its control in training progress,
    not only in reward.

    Returns {key: {"mean", "sd", "slope", "n"}}, skipping keys with fewer than two
    numeric points in the window.
    """
    entries = [e for e in (log_history or []) if isinstance(e, dict)]
    tail = entries[-window:] if window else entries
    if keys is None:
        keys = sorted(
            {
                k
                for e in tail
                for k, v in e.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )
    out = {}
    for k in keys:
        pts = [
            (float(e.get("step", i)), float(e[k]))
            for i, e in enumerate(tail)
            if isinstance(e.get(k), (int, float)) and not isinstance(e.get(k), bool)
        ]
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        out[k] = {
            "mean": float(ys.mean()),
            "sd": float(ys.std(ddof=1)),
            "slope": float(np.polyfit(xs, ys, 1)[0]),
            "n": len(pts),
        }
    return out


def gradient_liveness(log_history) -> dict | None:
    """How much of training actually carried a gradient.

    `frac_reward_zero_std` is the fraction of prompt-groups in a step whose
    rollouts all scored the same; under advantage weighting such a group
    contributes nothing. The mean of that series is the standing caveat's number.
    `frac_steps_live` is the better yardstick named in LAB_NOTES: the fraction of
    steps where at least one group had reward variance, i.e. where the optimizer
    saw anything at all. None when the series is absent.
    """
    vals = [
        float(e["frac_reward_zero_std"])
        for e in (log_history or [])
        if isinstance(e, dict)
        and isinstance(e.get("frac_reward_zero_std"), (int, float))
        and not isinstance(e.get("frac_reward_zero_std"), bool)
    ]
    if not vals:
        return None
    a = np.array(vals)
    return {
        "mean_frac_zero_std": float(a.mean()),
        "frac_steps_live": float((a < 1.0).mean()),
        "n_steps": int(a.size),
    }


# --- disk boundary -----------------------------------------------------------


def load_episodes(run_dir: str, split: str) -> dict:
    """seed -> episode record from runs/<exp>/episodes_<split>.jsonl.

    Raises FileNotFoundError when the split was never evaluated. A duplicated
    seed (a resumed or re-run split appending to the same file) keeps the LAST
    record, which is the one the run's own report was computed from.
    """
    path = os.path.join(run_dir, f"episodes_{split}.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[int(rec["seed"])] = rec
    return out


def load_config(run_dir: str) -> dict:
    """The run's frozen config, or {} when it has none."""
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_train_log(run_dir: str) -> list:
    """The run's per-step TRL log history, or [] when training wrote none."""
    path = os.path.join(run_dir, "train_log.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def split_tasks(config: dict, split: str) -> list:
    """Task families the named eval split drew from, falling back to training's.

    Read from the frozen config rather than recorded per episode: the family is a
    pure function of the seed, and storing a derived value in the episode record
    would give it a second source of truth that can disagree.
    """
    for s in ((config.get("eval") or {}).get("agentic") or {}).get("splits") or []:
        if str(s.get("name")) == split:
            tasks = (s.get("env_config") or {}).get("tasks")
            if tasks:
                return list(tasks)
            break
    return list(
        ((config.get("training") or {}).get("env_config") or {}).get("tasks") or []
    )


def episode_families(config: dict, split: str, seeds) -> dict:
    """seed -> task family, or {} when the split is single-family or not browsergym.

    browsergym maps a seed onto a family with `tasks[seed % len(tasks)]`; the
    mapping is imported from the adapter that owns it rather than re-implemented
    here, because a second copy of it would silently diverge and mislabel every
    per-family number.
    """
    if ((config.get("training") or {}).get("env")) != "browsergym":
        return {}
    tasks = split_tasks(config, split)
    if len(tasks) < 2:
        return {}
    from domains.browsergym.adapter import task_for_seed

    return {int(s): task_for_seed(tasks, s) for s in seeds}


# Reward-registry key -> the thesis condition that key defines. A config enabling
# neither is the lambda=0 control (E1); one enabling both would be E4, which no
# run has yet, so it is labelled and left off the dose axis rather than guessed at.
_SHAPED_CONDITIONS = {
    "token_length": "E2 cosine",
    "non_termination": "E3 non-termination",
}


def arm_condition(config: dict) -> tuple:
    """(condition label, lambda) for a run, read from its frozen config.

    lambda is the shaped component's `weight` - the coefficient of the penalty in
    `R_task - lambda * C_target` - and 0.0 for the control. Returns lambda None
    for a config enabling more than one shaped reward, which has no single dose.
    """
    rewards = config.get("rewards") or {}
    on = [
        (k, (rewards.get(k) or {}).get("weight", 1.0))
        for k in _SHAPED_CONDITIONS
        if (rewards.get(k) or {}).get("enabled")
    ]
    if not on:
        return "control (task reward only)", 0.0
    if len(on) == 1:
        return _SHAPED_CONDITIONS[on[0][0]], float(on[0][1])
    return "E4 combined", None


def _split_accuracy(episodes: dict, seeds=None) -> float | None:
    seeds = list(episodes) if seeds is None else list(seeds)
    if not seeds:
        return None
    return sum(1 for s in seeds if episodes[s]["correct"]) / len(seeds)


def dose_panel(
    run_dir: str,
    shifted: str = "shifted",
    family: str | None = None,
    config: dict | None = None,
) -> dict:
    """Off-target readings for one arm: shifted non-termination and one family's
    accuracy. Missing pieces come back as None so an arm that never ran the
    shifted split still contributes its held-out numbers.

    `config` overrides the run's own frozen config for the family mapping. Only
    a base-model eval needs it: nothing trained, so nothing froze a config into
    its run dir, and without a task list its episodes cannot be grouped by family.
    Passing the control's config is sound exactly because the splits are the same
    object in both runs' eval configs - never pass one from a differently split run.

    Every family in the split also gets a `family_acc:<family>` key, so a figure
    can panel all of them; `family` only decides which one is repeated under the
    plain `family_acc` key.
    """
    try:
        eps = load_episodes(run_dir, shifted)
    except OSError:
        return {"shifted_nonterm": None, "family_acc": None}
    cfg = load_config(run_dir) if config is None else config
    fams = episode_families(cfg, shifted, eps)
    out = {"shifted_nonterm": _nonterm_rate(eps, list(eps)), "family_acc": None}
    for fam in sorted(set(fams.values())):
        out[f"family_acc:{fam}"] = _split_accuracy(
            eps, [s for s, f in fams.items() if f == fam]
        )
    if family:
        out["family_acc"] = out.get(f"family_acc:{family}")
    return out


def dose_rows(
    base_dir: str,
    arm_dirs,
    held_out: str = "held_out",
    shifted: str = "shifted",
    family: str | None = None,
) -> list:
    """Rows for the dose-response figure: one per arm, plus the control at lambda=0.

    The control is emitted once per condition present among the arms, because it
    is the lambda=0 point of every one of them - that is what makes the arms a
    dose series rather than a set of unrelated runs. Its own paired numbers
    against itself are 0 by construction.
    """
    base_eps = load_episodes(base_dir, held_out)
    rows, conditions = [], []
    for d in arm_dirs:
        cond, lam = arm_condition(load_config(d))
        if lam is None:
            print(f"skip {_short(d)}: no single lambda ({cond})", file=sys.stderr)
            continue
        try:
            arm_eps = load_episodes(d, held_out)
        except OSError as exc:
            print(f"skip {_short(d)}: {exc}", file=sys.stderr)
            continue
        c = compare(
            base_eps,
            arm_eps,
            base_id=_short(base_dir),
            arm_id=_short(d),
            split=held_out,
        )
        if cond not in conditions:
            conditions.append(cond)
        rows.append(
            {
                "condition": cond,
                "lam": lam,
                "arm": _short(d),
                "losses": c.losses,
                "wins": c.wins,
                "median_dtok": (c.median_token_diff, c.median_ci_low, c.median_ci_high),
                **dose_panel(d, shifted, family),
            }
        )
    control = {
        "lam": 0.0,
        "arm": _short(base_dir),
        "losses": 0,
        "wins": 0,
        "median_dtok": 0.0,
        **dose_panel(base_dir, shifted, family),
    }
    return [{**control, "condition": c} for c in conditions] + rows


def dose_refs(
    ref_dir: str,
    shifted: str = "shifted",
    family: str | None = None,
    label: str = "E0 base",
    config: dict | None = None,
) -> dict:
    """Dashed reference lines for the dose figure, from a run with no lambda
    (the E0 base-model eval). `config` supplies the family mapping that a
    never-trained run has no frozen config to provide."""
    panel = dose_panel(ref_dir, shifted, family, config)
    return {k: (label, v) for k, v in panel.items() if v is not None}


def _short(run_dir: str) -> str:
    """Compact arm label: the leading handle of the run dir, e.g. e31.

    A seed replicate (`<exp>-s43`) keeps its seed, or every seed of one arm
    lands on the same row label and a seed sweep reads as one arm reported
    several times.
    """
    parts = os.path.basename(os.path.normpath(run_dir)).split("-")
    if len(parts) > 1 and re.fullmatch(r"s\d+", parts[-1]):
        return f"{parts[0]}-{parts[-1]}"
    return parts[0]


# --- reporting ---------------------------------------------------------------


def _fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _fmt_p(p):
    return "n/a" if p is None else (f"{p:.1e}" if p < 1e-4 else f"{p:.3g}")


def comparison_rows(comparisons) -> str:
    """Markdown table: one row per comparison (pooled rows first, then families)."""
    head = (
        "| arm | split | family | n | acc base | acc arm | wins | losses | McNemar p "
        "| n joint | median dtok | 95% CI | sign p | nonterm base | nonterm arm |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for c in comparisons:
        ci = (
            "n/a"
            if c.median_ci_low is None
            else f"[{c.median_ci_low:.0f}, {c.median_ci_high:.0f}]"
        )
        rows.append(
            f"| {c.arm_id} | {c.split} | {c.family or 'all'} | {c.n_paired} "
            f"| {c.base_correct / c.n_paired:.3f} | {c.arm_correct / c.n_paired:.3f} "
            f"| {c.wins} | {c.losses} | {_fmt_p(c.mcnemar_p)} | {c.n_joint_correct} "
            f"| {_fmt(c.median_token_diff, 0)} | {ci} | {_fmt_p(c.sign_p)} "
            f"| {_fmt(c.base_nonterm, 3)} | {_fmt(c.arm_nonterm, 3)} |"
            if c.n_paired
            else f"| {c.arm_id} | {c.split} | {c.family or 'all'} | 0 | - | - | - | - | - | - | - | - | - | - | - |"
        )
    return head + "\n".join(rows) + "\n"


_NOISE_KEYS = [
    "reward",
    "reward/EnvReward/raw_mean",
    "completions/mean_length",
    "kl",
]


def noise_rows(run_dirs, window: int = TAIL_STEPS) -> str:
    """Markdown table: the last-`window`-step noise floor and gradient liveness.

    Skips runs with no train_log.json (an eval-only run, or one killed before
    the log was written - grpo_runner writes it after trainer.train() returns).
    """
    head = (
        f"| run | series | last-{window} mean | SD | slope/step "
        "| mean frac_zero_std | frac steps live |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for d in run_dirs:
        log = load_train_log(d)
        if not log:
            continue
        tail = training_tail(log, keys=_NOISE_KEYS, window=window)
        live = gradient_liveness(log) or {}
        for i, (k, v) in enumerate(tail.items()):
            rows.append(
                f"| {_short(d) if i == 0 else ''} | `{k}` | {v['mean']:.4g} "
                f"| {v['sd']:.3g} | {v['slope']:+.3g} "
                f"| {_fmt(live.get('mean_frac_zero_std'), 3) if i == 0 else ''} "
                f"| {_fmt(live.get('frac_steps_live'), 3) if i == 0 else ''} |"
            )
    return head + "\n".join(rows) + "\n"


def render_markdown(comparisons, run_dirs, base_dir, split, window=TAIL_STEPS) -> str:
    """The full paired-comparison report for one split."""
    return (
        f"# Paired comparison vs {_short(base_dir)} [{split}]\n\n"
        f"Exact McNemar on the correctness flips; sign test and a {N_BOOTSTRAP}-replicate "
        f"paired bootstrap (RNG seed {BOOTSTRAP_SEED}) on the token difference over the "
        "jointly correct episodes. `dtok` is arm minus control, so negative is shorter.\n\n"
        + comparison_rows(comparisons)
        + f"\n## Training noise floor (last {window} steps)\n\n"
        "A between-arm delta smaller than a series' own SD is not a result; a non-zero "
        "slope means the arms also differ in training progress. `frac steps live` is the "
        "fraction of steps where at least one prompt-group had reward variance.\n\n"
        + noise_rows(run_dirs, window)
    )


def main():
    ap = argparse.ArgumentParser(description="Paired arm-vs-control statistics.")
    ap.add_argument("arms", nargs="+", help="arm run dirs")
    ap.add_argument("--base", required=True, help="control run dir (e.g. the E1 arm)")
    ap.add_argument("--split", default="held_out", help="eval split (default held_out)")
    ap.add_argument(
        "--by-family",
        action="store_true",
        help="also break each comparison down by task family",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=TAIL_STEPS,
        help=f"training tail length for the noise floor (default {TAIL_STEPS})",
    )
    ap.add_argument("-o", "--out", help="write markdown here instead of stdout")
    args = ap.parse_args()

    base_eps = load_episodes(args.base, args.split)
    comparisons = []
    for arm in args.arms:
        try:
            arm_eps = load_episodes(arm, args.split)
        except OSError as exc:  # arm never ran this split
            print(f"skip {arm}: {exc}", file=sys.stderr)
            continue
        comparisons.append(
            compare(
                base_eps,
                arm_eps,
                base_id=_short(args.base),
                arm_id=_short(arm),
                split=args.split,
            )
        )
        if not args.by_family:
            continue
        fams = episode_families(load_config(arm), args.split, arm_eps)
        for fam in sorted(set(fams.values())):
            keep = {s for s, f in fams.items() if f == fam}
            comparisons.append(
                compare(
                    {s: e for s, e in base_eps.items() if s in keep},
                    {s: e for s, e in arm_eps.items() if s in keep},
                    base_id=_short(args.base),
                    arm_id=_short(arm),
                    split=args.split,
                    family=fam,
                )
            )

    md = render_markdown(
        comparisons, [args.base, *args.arms], args.base, args.split, args.window
    )
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
