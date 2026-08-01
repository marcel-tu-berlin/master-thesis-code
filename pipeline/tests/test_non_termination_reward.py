"""E3 non-termination penalty.

The load-bearing property is the SIGN: a config author writes `weight: 4.0`
meaning lambda = 4, so the component itself must be negative for the bad case.
A flipped sign here trains the agent to never finish, and the run would look
healthy right up to the eval.
"""
import pytest

from training.rewards.non_termination import NonTerminationPenalty


class _FakeEnv:
    def __init__(self, done):
        self.done = done


def test_penalizes_non_terminated_episode():
    out = NonTerminationPenalty()(["p", "p"], ["c1", "c2"],
                                  environments=[_FakeEnv(True), _FakeEnv(False)])
    assert out == [0.0, -1.0]


def test_penalty_is_negative_so_positive_weight_is_a_penalty():
    # Composed as weight * value, so lambda > 0 must reduce the total reward.
    out = NonTerminationPenalty()(["p"], ["c"], environments=[_FakeEnv(False)])
    assert 4.0 * out[0] < 0


def test_env_without_done_counts_as_not_terminated():
    class _NoFlag:
        pass

    assert NonTerminationPenalty()(["p"], ["c"], environments=[_NoFlag()]) == [-1.0]


def test_missing_environments_raises():
    with pytest.raises(ValueError):
        NonTerminationPenalty()(["p"], ["c"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        NonTerminationPenalty()(["p"], ["c1", "c2"], environments=[_FakeEnv(True)])


def test_registry_entry():
    from training.rewards import REWARD_REGISTRY

    assert "non_termination" in REWARD_REGISTRY
    enabled, weight, builder = REWARD_REGISTRY["non_termination"]
    assert enabled is False and weight == 1.0
    assert isinstance(builder(None, None, {}, {}), NonTerminationPenalty)


def test_schema_accepts_the_key():
    from training.config_schema import validate_config

    validate_config({
        "experiment_id": "x", "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "agentic", "env": "reasoning_gym"},
        "rewards": {"compose_method": "naive_sum",
                    "non_termination": {"enabled": True, "weight": 4.0}},
    })


def test_warns_that_advantage_weighted_silences_the_penalty():
    from training.config_schema import warn_inert_scalars

    cfg = {"non_termination": {"enabled": True, "weight": 4.0}}
    assert any("naive_sum" in w for w in warn_inert_scalars(cfg, "advantage_weighted"))
    assert warn_inert_scalars(cfg, "naive_sum") == []
