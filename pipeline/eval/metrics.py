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
    # None when no correct samples — distinguishes "0% underthinking" (real
    # signal) from "no correct answers to measure" (no signal). Treat 0.0
    # only as a genuinely measured rate.
    underthinking_rate: float | None = None
    underthinking_rate_ci_low: float | None = None
    underthinking_rate_ci_high: float | None = None
    # Fraction of correct completions whose token count exceeds the per-split
    # P75 (configurable). Captures "correct answer with wasted reasoning" —
    # the inverse failure mode to underthinking. None when there are no
    # correct samples or too few total samples for a stable percentile.
    overthinking_rate: float | None = None
    overthinking_rate_ci_low: float | None = None
    overthinking_rate_ci_high: float | None = None
    overthinking_threshold: float | None = None  # the absolute token threshold used (P75 of all samples)
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
    underthinking_threshold: int = 50,
    overthinking_percentile: int = 75,
    n_bootstrap: int = 2000,
) -> EvalMetrics:
    """
    Compute all thesis metrics from per-sample results.

    underthinking_rate: fraction of CORRECT completions with <= threshold tokens.
    Flags cases where the model produced a correct answer with minimal reasoning -
    likely lucky pattern-matching rather than genuine derivation.

    overthinking_rate: fraction of CORRECT completions whose token count exceeds
    the per-split percentile (default P75) of ALL completions' token counts.
    Captures wasted reasoning — the inverse failure mode to underthinking.
    The percentile is computed over all samples (correct + incorrect) so the
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
    underthinking_rate = None
    under_ci_low: float | None = None
    under_ci_high: float | None = None
    if correct_results:
        under_flags = np.array(
            [1.0 if r.n_tokens <= underthinking_threshold else 0.0 for r in correct_results]
        )
        underthinking_rate = float(under_flags.mean())
        under_ci_low, under_ci_high = _bootstrap_ci(under_flags, n_bootstrap=n_bootstrap)

    # Overthinking: per-split percentile of all token counts (population-level
    # threshold), then count correct samples above it. Threshold is computed
    # once on the observed sample and held fixed during the bootstrap — this
    # is a CI on the rate conditional on the observed threshold, which is the
    # statistic the report quotes.
    overthinking_rate = None
    overthinking_threshold = None
    over_ci_low: float | None = None
    over_ci_high: float | None = None
    if correct_results and n >= 4:
        overthinking_threshold = float(np.percentile(all_tokens, overthinking_percentile))
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
