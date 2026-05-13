from typing import Callable

import torch


def _group_indices(prompts) -> list[list[int]]:
    """Partition completions into prompt-groups for per-group z-scoring.

    TRL GRPOTrainer's reward function receives `prompts` of length
    batch_size * num_generations, with each prompt repeated num_generations
    times *consecutively*. We rely on that ordering and compare each prompt
    to its predecessor with `==` only — no hashing, so no chance of a
    64-bit collision merging two distinct prompts into one group.

    Returns a list of index-lists, one per group.
    """
    if not prompts:
        return []
    groups: list[list[int]] = []
    cur: list[int] = [0]
    for i in range(1, len(prompts)):
        if prompts[i] == prompts[i - 1]:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


class AdvantageWeightedComposer:
    """
    Normalizes each reward component's advantages per prompt-group before combining.

    Motivation (DIET §3.2): raw reward variance σ²≈C(1-C) differs across components
    (e.g., binary accuracy has high variance; partial-credit format has low variance).
    Naive weighted sum lets high-variance components dominate the gradient signal
    regardless of assigned weight. Normalizing each component's rewards to
    zero-mean unit-std before summing ensures weights faithfully control contribution.

    Per-group normalization: GRPO advantages are computed within each prompt-group
    (n_rollouts completions for one prompt). When `per_device_train_batch_size > 1`,
    the reward function receives multiple groups concatenated. Z-scoring across
    that concatenation conflates rewards across different prompts, distorting the
    relative ranking inside each group. We z-score per group instead.
    """

    def __init__(self, components: list[tuple[Callable, float]]) -> None:
        self.components = components
        self.__name__ = "advantage_weighted_composer"

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        n = len(completions)
        total = [0.0] * n
        groups = _group_indices(prompts)

        for fn, weight in self.components:
            raw = fn(prompts, completions, **kwargs)
            r = torch.tensor(raw, dtype=torch.float32)
            normalized = torch.zeros_like(r)
            for idx in groups:
                if not idx:
                    continue
                slice_idx = torch.tensor(idx, dtype=torch.long)
                sub = r[slice_idx]
                std = sub.std()
                if std < 1e-6:
                    # Constant signal across the group carries no advantage
                    # information (every rollout gets the same value), so it
                    # contributes 0 to the composed reward regardless of its
                    # assigned weight. This is by design — there is no GRPO
                    # gradient to extract from a zero-variance signal — but
                    # it can surprise ablations where a saturated component
                    # (e.g. format_exact at 100% match) appears to have its
                    # weight ignored.
                    continue
                normalized[slice_idx] = (sub - sub.mean()) / std
            for i, v in enumerate(normalized.tolist()):
                total[i] += weight * v

        return total


class NaiveSumComposer:
    """Plain weighted sum — ablation baseline for E3.
    Contrast with AdvantageWeightedComposer to measure DIET advantage-weighting effect.
    """

    def __init__(self, components: list[tuple[Callable, float]]) -> None:
        self.components = components
        self.__name__ = "naive_sum_composer"

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        n = len(completions)
        total = [0.0] * n

        for fn, weight in self.components:
            raw = fn(prompts, completions, **kwargs)
            for i in range(n):
                total[i] += weight * raw[i]

        return total


def build_composer(
    components: list[tuple[Callable, float]],
    method: str = "advantage_weighted",
) -> AdvantageWeightedComposer | NaiveSumComposer:
    if method == "advantage_weighted":
        return AdvantageWeightedComposer(components)
    if method == "naive_sum":
        return NaiveSumComposer(components)
    raise ValueError(f"Unknown compose_method: {method!r}. Choose advantage_weighted | naive_sum")
