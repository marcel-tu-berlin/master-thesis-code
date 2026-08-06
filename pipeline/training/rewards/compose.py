from typing import Callable

# torch is imported lazily inside AdvantageWeightedComposer.__call__ so this
# module can be imported (and the torch-free NaiveSumComposer / metric draining
# exercised) without pulling the full deep-learning stack.


def _group_indices(n: int, group_size: int) -> list[list[int]]:
    """Positional prompt-groups for per-group z-scoring.

    TRL groups advantages positionally - rewards.view(-1, num_generations)
    (grpo_trainer.py) - so the composer must cut the same consecutive blocks.
    Grouping by adjacent prompt equality matched that only while neighbouring
    groups never shared their prompt text; a domain whose reset observation is
    seed-invariant (Wordle: the target word is hidden, every prompt in the
    batch is byte-identical) merged the whole batch into one z-scoring group
    and normalized each component across different games.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if n % group_size:
        raise ValueError(
            f"{n} completions do not divide into groups of {group_size}; "
            f"num_generations is out of sync with the trainer"
        )
    return [list(range(i, i + group_size)) for i in range(0, n, group_size)]


def _drain_step_metrics(composer) -> dict:
    """Average a composer's buffered per-step component metrics across this
    step's reward-fn calls, then clear the buffer. Shared by both composers.
    """
    buf = composer._step_metrics
    if not buf:
        return {}
    agg: dict[str, float] = {}
    for key in buf[0]:
        vals = [m[key] for m in buf if key in m]
        agg[key] = sum(vals) / len(vals)
    composer._step_metrics = []
    return agg


class AdvantageWeightedComposer:
    """
    Normalizes each reward component's advantages per prompt-group before combining.

    Motivation (DIET §3.2): raw reward variance σ²≈C(1-C) differs across components
    (e.g., binary accuracy has high variance; partial-credit format has low variance).
    Naive weighted sum lets high-variance components dominate the gradient signal
    regardless of assigned weight. Normalizing each component's rewards to
    zero-mean unit-std before summing ensures weights faithfully control contribution.

    Per-group normalization: GRPO advantages are computed within each prompt-group
    (n_rollouts completions for one prompt). With batch_size > 1 the reward
    function receives multiple groups concatenated - consecutive blocks of
    `num_generations` completions, the same positional cut TRL applies to the
    advantages. Z-scoring across the concatenation conflates rewards across
    different prompts, distorting the relative ranking inside each group.
    """

    def __init__(self, components: list[tuple[Callable, float]],
                 num_generations: int) -> None:
        self.components = components
        self.num_generations = int(num_generations)
        self.__name__ = "advantage_weighted_composer"
        # Per-step diagnostics, drained by a TrainerCallback (T2.1). Purely
        # observational — never read back into the composed reward.
        self._step_metrics: list[dict] = []

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        import torch

        n = len(completions)
        total = [0.0] * n
        groups = _group_indices(n, self.num_generations)

        call_metrics: dict[str, float] = {}
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
            # weight * normalized is exactly the old `weight * v` per element —
            # the composed reward is unchanged; we only also record diagnostics.
            contribution = weight * normalized
            for i, v in enumerate(contribution.tolist()):
                total[i] += v
            name = type(fn).__name__
            call_metrics[f"reward/{name}/raw_mean"] = float(r.mean())
            call_metrics[f"reward/{name}/raw_std"] = float(r.std(unbiased=False))
            call_metrics[f"reward/{name}/contrib_l1"] = float(contribution.abs().mean())

        self._step_metrics.append(call_metrics)
        return total

    def pop_step_metrics(self) -> dict:
        return _drain_step_metrics(self)


class NaiveSumComposer:
    """Plain weighted sum — ablation baseline for E3.
    Contrast with AdvantageWeightedComposer to measure DIET advantage-weighting effect.
    """

    def __init__(self, components: list[tuple[Callable, float]]) -> None:
        self.components = components
        self.__name__ = "naive_sum_composer"
        self._step_metrics: list[dict] = []

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        n = len(completions)
        total = [0.0] * n

        call_metrics: dict[str, float] = {}
        for fn, weight in self.components:
            raw = fn(prompts, completions, **kwargs)
            for i in range(n):
                total[i] += weight * raw[i]
            if n:
                mean = sum(raw) / n
                var = sum((x - mean) ** 2 for x in raw) / n
                name = type(fn).__name__
                call_metrics[f"reward/{name}/raw_mean"] = float(mean)
                call_metrics[f"reward/{name}/raw_std"] = float(var ** 0.5)
                call_metrics[f"reward/{name}/contrib_l1"] = float(sum(abs(weight * x) for x in raw) / n)

        self._step_metrics.append(call_metrics)
        return total

    def pop_step_metrics(self) -> dict:
        return _drain_step_metrics(self)


def build_composer(
    components: list[tuple[Callable, float]],
    method: str,
    num_generations: int,
) -> AdvantageWeightedComposer | NaiveSumComposer:
    if method == "advantage_weighted":
        return AdvantageWeightedComposer(components, num_generations)
    if method == "naive_sum":
        return NaiveSumComposer(components)
    raise ValueError(f"Unknown compose_method: {method!r}. Choose advantage_weighted | naive_sum")
