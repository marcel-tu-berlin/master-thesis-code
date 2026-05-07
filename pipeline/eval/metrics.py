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
    # None when no correct samples — distinguishes "0% underthinking" (real
    # signal) from "no correct answers to measure" (no signal). Treat 0.0
    # only as a genuinely measured rate.
    underthinking_rate: float | None = None
    pearson_difficulty_length: float | None = None
    pearson_p_value: float | None = None
    n_samples: int = 0
    n_correct: int = 0
    raw: list[SampleResult] = field(default_factory=list)


def _bootstrap_ci(corrects: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    n = len(corrects)
    rng = np.random.default_rng(42)
    boot_accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(corrects, size=n, replace=True)
        boot_accs[i] = sample.mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_accs, 100 * alpha)), float(np.percentile(boot_accs, 100 * (1 - alpha)))


def compute_metrics(
    results: list[SampleResult],
    underthinking_threshold: int = 50,
    n_bootstrap: int = 2000,
) -> EvalMetrics:
    """
    Compute all thesis metrics from per-sample results.

    underthinking_rate: fraction of CORRECT completions with <= threshold tokens.
    Flags cases where the model produced a correct answer with minimal reasoning -
    likely lucky pattern-matching rather than genuine derivation.

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

    mean_tokens = sum(r.n_tokens for r in results) / n

    correct_results = [r for r in results if r.correct]
    if correct_results:
        underthinking_rate = (
            sum(1 for r in correct_results if r.n_tokens <= underthinking_threshold)
            / len(correct_results)
        )
    else:
        underthinking_rate = None

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
        underthinking_rate=underthinking_rate,
        pearson_difficulty_length=pearson_val,
        pearson_p_value=pearson_p,
        n_samples=n,
        n_correct=n_correct,
        raw=results,
    )
