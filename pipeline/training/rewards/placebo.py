import random

from training.rewards.compose import _group_indices


def maybe_placebo(component, reward_cfg: dict, training_cfg: dict, seed: int):
    """Wrap `component` when its config says `placebo: true`, else return it."""
    if not (reward_cfg or {}).get("placebo"):
        return component
    from training.config_schema import DEFAULT_N_ROLLOUTS

    return WithinGroupShuffle(
        component,
        num_generations=int((training_cfg or {}).get("n_rollouts", DEFAULT_N_ROLLOUTS)),
        seed=int(seed),
    )


class WithinGroupShuffle:
    """Shuffle a shaped component's rollout assignments within each group.

    Calls the wrapped component, then permutes its values within each positional
    prompt-group - the same consecutive blocks of `num_generations` completions
    TRL cuts advantages on. Every group keeps the exact multiset of values the
    real component produced: same scale, same within-group variance, same share
    of non-zero entries. Only which rollout gets which value is random. For any
    fixed group position, the expected assigned value is the group mean, so its
    expected centered contribution is zero.

    The shuffle breaks the observed pairing between each rollout and its cost,
    but it also changes the component's covariance with the task reward. It
    therefore does not guarantee the same total-reward variance or gradient
    noise as the real shaped arm. A difference between arms supports a role for
    the cost assignment. Similar endpoints do not identify a shared mechanism.

    Seeded, so a placebo arm is reproducible from its frozen config. Metrics the
    composer logs are keyed by this class name, which marks the arm in the logs.

    Every placebo component of a run is seeded from the same run seed and
    called on the same batches, so an E4 placebo permutes both costs with one
    permutation: a rollout's (length, budget) cost pair moves together and the
    combined cost keeps its joint structure. test_placebo pins this.
    """

    def __init__(self, inner, num_generations: int, seed: int) -> None:
        self.inner = inner
        self.num_generations = int(num_generations)
        self._rng = random.Random(int(seed))

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        raw = list(self.inner(prompts, completions, **kwargs))
        for idx in _group_indices(len(raw), self.num_generations):
            vals = [raw[i] for i in idx]
            self._rng.shuffle(vals)
            for i, v in zip(idx, vals, strict=True):
                raw[i] = v
        return raw
