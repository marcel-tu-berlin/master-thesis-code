"""Declared successful-response cost contracts; old cosine remains unchanged."""

import copy
import math
import random
from types import SimpleNamespace

import pytest

from training.rewards.successful_length import (
    SuccessfulLengthPenalty,
    successful_length_costs,
)


def test_registry_schema_and_old_readiness_do_not_misclassify_new_reward():
    from pathlib import Path

    import yaml
    from probes.readiness import _condition

    from training.config_schema import validate_config
    from training.rewards import REWARD_REGISTRY

    config = yaml.safe_load(
        Path(
            "configs/archive/development-2026-09-24/readiness/g3-feedback-v2-e1.yaml"
        ).read_text()
    )
    config["rewards"]["successful_length"] = {
        "enabled": True,
        "weight": 0.1,
        "kind": "relative",
        "max_len": 4096,
    }
    config["model"]["initial_adapter"] = "runs/example/checkpoint-final"
    validate_config(config)
    assert _condition(config) is None
    runner = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text, **_: list(text))
    )
    reward = REWARD_REGISTRY["successful_length"][2](
        None, runner, config["training"], config["rewards"]["successful_length"]
    )
    assert reward.kind == "relative" and reward.num_generations == 8
    for key, bad in (("kind", "rank"), ("max_len", 0), ("weight", 1.0)):
        changed = copy.deepcopy(config)
        changed["rewards"]["successful_length"][key] = bad
        with pytest.raises(ValueError, match="successful_length"):
            validate_config(changed)
    config["training"]["scale_rewards"] = "group"
    with pytest.raises(ValueError, match="scale_rewards"):
        validate_config(config)


@pytest.mark.parametrize("kind", ["linear", "relative"])
def test_order_bounds_group_isolation_and_dose(kind):
    lengths = [600, 800, 0, 4096, 10, 10, 10, 10]
    correct = [True, True, False, True, False, False, False, False]
    costs = successful_length_costs(lengths, correct, kind, 4, 4096)
    assert 0 <= costs[0] < costs[1] < costs[3] <= 1
    assert costs[2] == 0 and costs[4:] == [0] * 4
    assert costs[:4] == successful_length_costs(lengths[:4], correct[:4], kind, 4, 4096)
    for weight in (0, 0.1, 0.2, 0.4):
        rewards = [
            float(y) - weight * c for y, c in zip(correct[:4], costs[:4], strict=True)
        ]
        assert all(
            1 - weight <= r <= 1 for r, y in zip(rewards, correct[:4], strict=True) if y
        )
        assert rewards[2] == 0
        centered = [r - sum(rewards) / 4 for r in rewards]
        base = [float(y) - 0.75 for y in correct[:4]]
        expected = [-weight * (c - sum(costs[:4]) / 4) for c in costs[:4]]
        assert [a - b for a, b in zip(centered, base, strict=True)] == pytest.approx(
            expected
        )


def test_relative_edge_cases_and_independent_formula():
    assert successful_length_costs([0, 4096], [False, False], "relative", 2) == [0, 0]
    assert successful_length_costs([1, 4096], [True, False], "relative", 2) == [0.5, 0]
    assert successful_length_costs([20, 20], [True, True], "relative", 2) == [0.5, 0.5]
    assert successful_length_costs(
        [600, 800], [True, True], "relative", 2
    ) == pytest.approx([1 / (1 + math.exp(1)), 1 / (1 + math.exp(-1))])
    assert successful_length_costs([1, 1, 1, 10**9], [True] * 4, "relative", 4)[-1] < 1
    assert successful_length_costs([4096, 8192], [True, True], "linear", 2) == [1, 1]


@pytest.mark.parametrize("kind", ["linear", "relative"])
def test_permutation_and_observation_transparency(kind):
    lengths, correct = [600, 800, 400, 1000], [True, True, False, True]
    before, rng = copy.deepcopy((lengths, correct)), random.getstate()
    costs = successful_length_costs(lengths, correct, kind, 4)
    order = [3, 0, 2, 1]
    assert successful_length_costs(
        [lengths[i] for i in order], [correct[i] for i in order], kind, 4
    ) == pytest.approx([costs[i] for i in order])
    assert (lengths, correct) == before and random.getstate() == rng


def test_real_reward_surface_counts_reasoning_and_ignores_tool_feedback():
    tokenizer = SimpleNamespace(encode=lambda text, **_: list(text))
    reward = SuccessfulLengthPenalty(tokenizer, "linear", 2, 100)
    completions = [
        [{"role": "assistant", "content": "x", "reasoning_content": "xxx"}],
        [{"role": "assistant", "content": "xx"}, {"role": "tool", "content": "z" * 90}],
    ]
    environments = [SimpleNamespace(reward=1), SimpleNamespace(reward=0)]
    assert reward(["p"] * 2, completions, environments=environments) == [-0.04, 0]
    with pytest.raises(ValueError, match="environments"):
        reward(["p"] * 2, completions)
    with pytest.raises(ValueError, match="environments"):
        reward(["p"] * 2, completions, environments=environments[:1])


@pytest.mark.parametrize(
    "lengths,correct,kind,size,cap",
    [
        ([1], [], "linear", 1, 4096),
        ([-1], [True], "linear", 1, 4096),
        ([1], [True], "unknown", 1, 4096),
        ([1], [True], "linear", 0, 4096),
        ([1], [True], "linear", 2, 4096),
        ([1], [True], "linear", 1, 0),
    ],
)
def test_invalid_inputs_fail_loudly(lengths, correct, kind, size, cap):
    with pytest.raises(ValueError):
        successful_length_costs(lengths, correct, kind, size, cap)
