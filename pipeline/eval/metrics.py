import json
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import pearsonr


@dataclass
class SampleResult:
    correct: bool
    n_tokens: int
    difficulty: float | None = None


@dataclass
class EvalMetrics:
    accuracy: float = 0.0
    accuracy_ci_low: float = 0.0
    accuracy_ci_high: float = 0.0
    mean_token_count: float = 0.0
    mean_token_count_ci_low: float = 0.0
    mean_token_count_ci_high: float = 0.0
    # Fraction of correct completions whose token count is at or below the
    # per-split P10 (configurable). Captures "correct with too little
    # reasoning — likely pattern-match luck". None when there are no
    # correct samples or too few total samples for a stable percentile.
    underthinking_rate: float | None = None
    underthinking_rate_ci_low: float | None = None
    underthinking_rate_ci_high: float | None = None
    underthinking_threshold: float | None = None  # absolute token threshold (P10 of all samples)
    # Fraction of correct completions whose token count exceeds the per-split
    # P75 (configurable). Captures "correct answer with wasted reasoning" —
    # the inverse failure mode to underthinking. None when there are no
    # correct samples or too few total samples for a stable percentile.
    overthinking_rate: float | None = None
    overthinking_rate_ci_low: float | None = None
    overthinking_rate_ci_high: float | None = None
    overthinking_threshold: float | None = None  # absolute token threshold (P75 of all samples)
    pearson_difficulty_length: float | None = None
    pearson_p_value: float | None = None
    n_samples: int = 0
    n_correct: int = 0
    raw: list[SampleResult] = field(default_factory=list)


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI for the mean of `values`. Works for both binary rates
    (mean of 0/1 = rate) and continuous values (e.g. token counts).
    Deterministic via fixed RNG seed so reports are reproducible.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot[i] = sample.mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


def compute_metrics(
    results: list[SampleResult],
    underthinking_percentile: int = 10,
    overthinking_percentile: int = 75,
    n_bootstrap: int = 2000,
    underthinking_threshold: float | None = None,
    overthinking_threshold: float | None = None,
) -> EvalMetrics:
    """
    Compute all thesis metrics from per-sample results.

    underthinking_rate: fraction of CORRECT completions whose token count is
    at or below the per-split percentile (default P10) of ALL completions.
    Flags correct answers produced with unusually little reasoning relative
    to the split — likely lucky pattern-matching rather than genuine
    derivation. The threshold adapts to dataset verbosity so the metric
    behaves consistently across GSM-8K (short) and MATH (long).

    overthinking_rate: fraction of CORRECT completions whose token count exceeds
    the per-split percentile (default P75) of ALL completions' token counts.
    Captures wasted reasoning — the inverse failure mode to underthinking.
    Both thresholds are computed over all samples (correct + incorrect) so the
    threshold reflects the population's overall verbosity, not just the
    correct-subset distribution. Requires at least 4 samples for the
    percentile to be meaningful.

    pearson_difficulty_length: Pearson r between difficulty score and mean token count.
    Positive r means the model spends more tokens on harder problems - desired behaviour.
    Requires difficulty labels (MATH levels 1-5); returns None for GSM-8K / DAPO.

    accuracy_ci_low/high: 95% bootstrap confidence interval for accuracy.
    """
    if not results:
        return EvalMetrics()

    n = len(results)
    n_correct = sum(r.correct for r in results)
    accuracy = n_correct / n

    all_tokens = np.array([r.n_tokens for r in results], dtype=float)
    mean_tokens = float(all_tokens.mean())
    tokens_ci_low, tokens_ci_high = _bootstrap_ci(all_tokens, n_bootstrap=n_bootstrap)

    correct_results = [r for r in results if r.correct]

    # Absolute-threshold overrides (e.g. from a fixed reference run via
    # load_reference_thresholds) take precedence over the per-run percentile.
    # When set, they make the over/under-thinking rate comparable across runs;
    # when None we fall back to the per-run percentile (back-compatible). An
    # override also bypasses the n>=4 percentile-stability guard, since the
    # threshold no longer depends on this run's sample size.
    under_override = underthinking_threshold
    over_override = overthinking_threshold

    # Both under- and overthinking use a per-split percentile of all token
    # counts as the threshold (unless overridden), then count correct samples
    # on the matching side. Thresholds are computed once on the observed sample
    # and held fixed during the bootstrap — the reported CI is on the rate
    # conditional on the observed threshold.
    underthinking_rate = None
    underthinking_threshold = None
    under_ci_low: float | None = None
    under_ci_high: float | None = None
    if correct_results and (n >= 4 or under_override is not None):
        underthinking_threshold = (
            under_override if under_override is not None
            else float(np.percentile(all_tokens, underthinking_percentile))
        )
        under_flags = np.array(
            [1.0 if r.n_tokens <= underthinking_threshold else 0.0 for r in correct_results]
        )
        underthinking_rate = float(under_flags.mean())
        under_ci_low, under_ci_high = _bootstrap_ci(under_flags, n_bootstrap=n_bootstrap)

    overthinking_rate = None
    overthinking_threshold = None
    over_ci_low: float | None = None
    over_ci_high: float | None = None
    if correct_results and (n >= 4 or over_override is not None):
        overthinking_threshold = (
            over_override if over_override is not None
            else float(np.percentile(all_tokens, overthinking_percentile))
        )
        over_flags = np.array(
            [1.0 if r.n_tokens > overthinking_threshold else 0.0 for r in correct_results]
        )
        overthinking_rate = float(over_flags.mean())
        over_ci_low, over_ci_high = _bootstrap_ci(over_flags, n_bootstrap=n_bootstrap)

    corrects = np.array([r.correct for r in results], dtype=float)
    ci_low, ci_high = _bootstrap_ci(corrects, n_bootstrap=n_bootstrap)

    pearson_val = None
    pearson_p = None
    with_difficulty = [(r.difficulty, r.n_tokens) for r in results if r.difficulty is not None]
    if len(with_difficulty) >= 10:
        difficulties = [d for d, _ in with_difficulty]
        lengths = [l for _, l in with_difficulty]
        # pearsonr returns NaN with a warning when either input is constant.
        # Skip the call entirely so the report stays free of NaNs.
        if len(set(difficulties)) > 1 and len(set(lengths)) > 1:
            r_val, p_val = pearsonr(difficulties, lengths)
            pearson_val = float(r_val)
            pearson_p = float(p_val)

    return EvalMetrics(
        accuracy=accuracy,
        accuracy_ci_low=ci_low,
        accuracy_ci_high=ci_high,
        mean_token_count=mean_tokens,
        mean_token_count_ci_low=tokens_ci_low,
        mean_token_count_ci_high=tokens_ci_high,
        underthinking_rate=underthinking_rate,
        underthinking_rate_ci_low=under_ci_low,
        underthinking_rate_ci_high=under_ci_high,
        underthinking_threshold=underthinking_threshold,
        overthinking_rate=overthinking_rate,
        overthinking_rate_ci_low=over_ci_low,
        overthinking_rate_ci_high=over_ci_high,
        overthinking_threshold=overthinking_threshold,
        pearson_difficulty_length=pearson_val,
        pearson_p_value=pearson_p,
        n_samples=n,
        n_correct=n_correct,
        raw=results,
    )


def load_reference_thresholds(
    report_path: str,
    under_pct: int = 10,
    over_pct: int = 75,
    min_samples: int = 4,
) -> dict[str, dict]:
    """Derive fixed over/under-thinking thresholds from a reference eval report.

    Reads `report_path` (a run's eval_report.json) and, for each split that
    carries a `samples` series, returns the P`under_pct`/P`over_pct` of its token
    counts. Thresholds are kept PER SPLIT on purpose: GSM-8K (id_split) and MATH
    (near_ood) have legitimately different verbosity, so a single global
    threshold would mislabel one of them. Feed the e0 accuracy-only baseline here
    so every run's thinking rates are measured against the same yardstick.

    Returns `{split: {"underthinking_threshold": float, "overthinking_threshold": float}}`,
    skipping splits with fewer than `min_samples` samples.
    """
    with open(report_path) as f:
        report = json.load(f)
    out: dict[str, dict] = {}
    for split, metrics in (report.get("results") or {}).items():
        samples = (metrics or {}).get("samples")
        if not samples or len(samples) < min_samples:
            continue
        tokens = np.array([s["n_tokens"] for s in samples], dtype=float)
        out[split] = {
            "underthinking_threshold": float(np.percentile(tokens, under_pct)),
            "overthinking_threshold": float(np.percentile(tokens, over_pct)),
        }
    return out
