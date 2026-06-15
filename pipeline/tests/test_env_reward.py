import pytest

from training.rewards.env_reward import EnvReward


def test_passthrough():
    assert EnvReward()(["p", "p"], ["c1", "c2"], env_reward=[1.0, 0.0]) == [1.0, 0.0]


def test_missing_raises():
    with pytest.raises(ValueError):
        EnvReward()(["p"], ["c"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        EnvReward()(["p"], ["c1", "c2"], env_reward=[1.0])


def test_registry_has_env_reward():
    from training.rewards import REWARD_REGISTRY

    assert "env_reward" in REWARD_REGISTRY
    enabled, weight, _builder = REWARD_REGISTRY["env_reward"]
    assert enabled is False and weight == 1.0
