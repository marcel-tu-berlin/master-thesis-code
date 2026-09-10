"""Placebo arm: the shaped term uniformly shuffled within each prompt-group.

What must hold: every group keeps exactly the values the real component
produced, groups never exchange values, the assignment is random but
reproducible from the seed, and the `placebo: true` config key wraps the
component while the task reward can never be shuffled. These invariants do not
claim matched total-reward variance or gradient noise.
"""

import pytest

from training.rewards.placebo import WithinGroupShuffle


class _Ramp:
    """Distinct value per completion so a permutation is visible."""

    def __call__(self, prompts, completions, **kwargs):
        return [float(i) for i in range(len(completions))]


def _run(seed=0, n=8, group=4):
    return WithinGroupShuffle(_Ramp(), num_generations=group, seed=seed)(
        ["p"] * n, ["c"] * n
    )


def test_each_group_keeps_its_own_multiset():
    out = _run()
    assert sorted(out[:4]) == [0.0, 1.0, 2.0, 3.0]
    assert sorted(out[4:]) == [4.0, 5.0, 6.0, 7.0]


def test_assignment_is_actually_permuted():
    # Over a few seeds at least one group must leave the identity order.
    assert any(_run(seed=s) != [float(i) for i in range(8)] for s in range(5))


def test_same_seed_reproduces_and_calls_advance_the_stream():
    a = WithinGroupShuffle(_Ramp(), 4, seed=7)
    b = WithinGroupShuffle(_Ramp(), 4, seed=7)
    first_a, first_b = a(["p"] * 8, ["c"] * 8), b(["p"] * 8, ["c"] * 8)
    assert first_a == first_b
    # Successive calls draw fresh permutations rather than replaying one.
    seconds = [a(["p"] * 8, ["c"] * 8) for _ in range(5)]
    assert any(s != first_a for s in seconds)


def test_components_sharing_a_seed_shuffle_jointly():
    """E4 placebo: both costs get the same permutation, so a rollout's cost
    pair moves together and the combined cost keeps its joint structure."""

    class _Tens:
        def __call__(self, prompts, completions, **kwargs):
            return [10.0 * i for i in range(len(completions))]

    a = WithinGroupShuffle(_Ramp(), 4, seed=11)
    b = WithinGroupShuffle(_Tens(), 4, seed=11)
    for _ in range(3):
        out_a, out_b = a(["p"] * 8, ["c"] * 8), b(["p"] * 8, ["c"] * 8)
        assert out_b == [10.0 * v for v in out_a]


def test_batch_must_divide_into_groups():
    with pytest.raises(ValueError):
        WithinGroupShuffle(_Ramp(), 4, seed=0)(["p"] * 6, ["c"] * 6)


def test_placebo_key_wraps_the_shaped_component_only():
    from training.config_schema import validate_config
    from training.rewards import REWARD_REGISTRY
    from training.rewards.non_termination import NonTerminationPenalty
    from training.rewards.placebo import maybe_placebo

    class _Runner:
        def completion_budget(self):
            return 1024

    config = {
        "experiment_id": "x",
        "seed": 3,
        "model": {"slug": "qwen3-1.7b"},
        "training": {
            "mode": "agentic",
            "env": "browsergym",
            "n_rollouts": 4,
            "scale_rewards": "none",
        },
        "rewards": {
            "compose_method": "naive_sum",
            "env_reward": {"enabled": True, "weight": 1.0},
            "non_termination": {"enabled": True, "weight": 0.5, "placebo": True},
        },
    }
    validate_config(config)
    tr, rw = config["training"], config["rewards"]
    build = {k: v[2] for k, v in REWARD_REGISTRY.items()}
    env_fn = maybe_placebo(
        build["env_reward"](None, _Runner(), tr, rw["env_reward"]),
        rw["env_reward"],
        tr,
        config["seed"],
    )
    e3_fn = maybe_placebo(
        build["non_termination"](None, _Runner(), tr, rw["non_termination"]),
        rw["non_termination"],
        tr,
        config["seed"],
    )
    assert type(env_fn).__name__ == "EnvReward"
    assert isinstance(e3_fn, WithinGroupShuffle)
    assert isinstance(e3_fn.inner, NonTerminationPenalty)
    assert e3_fn.num_generations == 4


def test_schema_rejects_placebo_on_the_task_reward():
    from training.config_schema import validate_config

    with pytest.raises(ValueError, match="env_reward"):
        validate_config(
            {
                "experiment_id": "x",
                "model": {"slug": "qwen3-1.7b"},
                "training": {"mode": "agentic", "env": "browsergym"},
                "rewards": {"env_reward": {"enabled": True, "placebo": True}},
            }
        )
