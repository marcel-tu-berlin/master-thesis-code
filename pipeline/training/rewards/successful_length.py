"""Bounded length costs on successful episodes (decision 0019).

These return negative costs; the composer adds the independently weighted task
reward. The relative variant follows Arora and Zanette (2025), with an explicit
one-token denominator floor. Neither variant changes the historical token ruler.
"""

import math
import statistics

from domains.env_base import CORRECT_REWARD_THRESHOLD
from training.rewards.compose import _group_indices
from training.rewards.utils import model_token_count


def successful_length_costs(
    lengths: list[int],
    correct: list[bool],
    kind: str,
    group_size: int,
    max_len: int = 4096,
) -> list[float]:
    if kind not in {"linear", "relative"}:
        raise ValueError(f"Unknown successful length cost: {kind}")
    if max_len <= 0 or len(lengths) != len(correct):
        raise ValueError("Positive max_len and aligned correctness/lengths required")
    if any(type(n) is not int or n < 0 for n in lengths):
        raise ValueError("Lengths must be nonnegative integer token counts")
    groups = _group_indices(len(lengths), group_size)
    costs = [0.0] * len(lengths)
    for group in groups:
        successful = [i for i in group if correct[i]]
        if not successful:
            continue
        if kind == "linear":
            for i in successful:
                costs[i] = min(lengths[i] / max_len, 1.0)
        else:
            values = [lengths[i] for i in successful]
            mean = statistics.mean(values)
            scale = max(statistics.pstdev(values), 1.0)
            for i in successful:
                z = (lengths[i] - mean) / scale
                # Stable sigmoid, including deliberately extreme test fixtures.
                exp = math.exp(-abs(z))
                costs[i] = 1 / (1 + exp) if z >= 0 else exp / (1 + exp)
    return costs


class SuccessfulLengthPenalty:
    def __init__(self, tokenizer, kind: str, num_generations: int, max_len: int = 4096):
        self.tokenizer = tokenizer
        self.kind = kind
        self.num_generations = num_generations
        self.max_len = max_len
        successful_length_costs([], [], kind, num_generations, max_len)

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        environments = kwargs.get("environments")
        if environments is None or len(environments) != len(completions):
            raise ValueError("SuccessfulLengthPenalty requires aligned environments")
        lengths = [model_token_count(c, self.tokenizer) for c in completions]
        correct = [float(e.reward) >= CORRECT_REWARD_THRESHOLD for e in environments]
        return [
            -c
            for c in successful_length_costs(
                lengths, correct, self.kind, self.num_generations, self.max_len
            )
        ]
