import pytest

from training.rewards.env_reward import EnvReward


class _FakeEnv:
    """Stands in for a TRL environment instance with a stored episode reward."""

    def __init__(self, reward):
        self.reward = reward


def test_reads_environments():
    envs = [_FakeEnv(1.0), _FakeEnv(0.0)]
    assert EnvReward()(["p", "p"], ["c1", "c2"], environments=envs) == [1.0, 0.0]


def test_coerces_to_float():
    envs = [_FakeEnv(1), _FakeEnv(0)]
    out = EnvReward()(["p", "p"], ["c1", "c2"], environments=envs)
    assert out == [1.0, 0.0] and all(isinstance(x, float) for x in out)


def test_missing_environments_raises():
    with pytest.raises(ValueError):
        EnvReward()(["p"], ["c"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        EnvReward()(["p"], ["c1", "c2"], environments=[_FakeEnv(1.0)])


def test_registry_has_env_reward():
    from training.rewards import REWARD_REGISTRY

    assert "env_reward" in REWARD_REGISTRY
    enabled, weight, _builder = REWARD_REGISTRY["env_reward"]
    assert enabled is False and weight == 1.0


def test_dataset_only_rewards_default_off_in_agentic():
    from training.rewards import default_enabled

    for k in ("format_exact", "format_approx", "accuracy", "numeric"):
        assert default_enabled(k, agentic=True) is False   # need answers/tags -> off
        assert default_enabled(k, agentic=False) is True   # registry default in dataset mode


def test_env_and_efficiency_defaults_mode_independent():
    from training.rewards import default_enabled

    for k in ("env_reward", "token_length", "token_entropy"):
        assert default_enabled(k, agentic=True) == default_enabled(k, agentic=False)
