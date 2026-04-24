from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SampleResult:
    correct: bool
    n_tokens: int
    difficulty: float | None = None


@dataclass
class EvalMetrics:
    accuracy: float = 0.0
    mean_token_count: float = 0.0
    underthinking_rate: float = 0.0
    pearson_difficulty_length: float | None = None
    n_samples: int = 0
    n_correct: int = 0
    raw: list[SampleResult] = field(default_factory=list)


def compute_metrics(results: list[SampleResult], underthinking_threshold: int = 50) -> EvalMetrics:
    """
    Compute all thesis metrics from per-sample results.

    underthinking_rate: fraction of CORRECT completions with <= threshold tokens.
    Flags cases where the model produced a correct answer with minimal reasoning —
    likely lucky pattern-matching rather than genuine derivation.

    pearson_difficulty_length: Pearson r between difficulty score and mean token count.
    Positive r means the model spends more tokens on harder problems — desired behaviour.
    Requires difficulty labels (MATH levels 1-5); returns None for GSM-8K / DAPO.
    """
    if not results:
        return EvalMetrics()

    n = len(results)
    n_correct = sum(r.correct for r in results)
    accuracy = n_correct / n

    mean_tokens = sum(r.n_tokens for r in results) / n

    # Underthinking: correct AND short
    correct_results = [r for r in results if r.correct]
    underthinking_rate = (
        sum(1 for r in correct_results if r.n_tokens <= underthinking_threshold)
        / max(len(correct_results), 1)
    )

    # Pearson(difficulty, token_count) — only if difficulty labels available
    pearson = None
    with_difficulty = [(r.difficulty, r.n_tokens) for r in results if r.difficulty is not None]
    if len(with_difficulty) >= 10:
        difficulties = [d for d, _ in with_difficulty]
        lengths = [l for _, l in with_difficulty]
        pearson = _pearson(difficulties, lengths)

    return EvalMetrics(
        accuracy=accuracy,
        mean_token_count=mean_tokens,
        underthinking_rate=underthinking_rate,
        pearson_difficulty_length=pearson,
        n_samples=n,
        n_correct=n_correct,
        raw=results,
    )


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(
        sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)
    )
    return num / den if den > 1e-12 else 0.0
