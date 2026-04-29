from typing import Callable

import torch


class AdvantageWeightedComposer:
    """
    Normalizes each reward component's advantages independently before combining.

    Motivation (DIET §3.2): raw reward variance σ²≈C(1-C) differs across components
    (e.g., binary accuracy has high variance; partial-credit format has low variance).
    Naive weighted sum lets high-variance components dominate the gradient signal
    regardless of assigned weight. Normalizing each component's batch rewards to
    zero-mean unit-std before summing ensures weights faithfully control contribution.
    """

    def __init__(self, components: list[tuple[Callable, float]]) -> None:
        self.components = components
        self.__name__ = "advantage_weighted_composer"

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        n = len(completions)
        total = [0.0] * n

        for fn, weight in self.components:
            raw = fn(prompts, completions, **kwargs)
            r = torch.tensor(raw, dtype=torch.float32)
            std = r.std()
            normalized = (r - r.mean()) / (std + 1e-8)
            for i in range(n):
                total[i] += weight * normalized[i].item()

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
