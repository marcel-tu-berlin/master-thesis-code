"""E3 budget-exhaustion penalty.

Two load-bearing properties. The SIGN: a config author writes `weight: 4.0`
meaning lambda = 4, so the component itself must be negative for the bad case.
The LINE: an episode the model ended on its own (no tool call) is NOT the bad
case - premature stopping is the substitute the thesis predicts and the panel
measures, so the cost must leave it alone. A cost that penalized every not-done
episode would make that substitution impossible by construction.
"""
import pytest

from training.rewards.non_termination import NonTerminationPenalty


class _FakeEnv:
    def __init__(self, done):
        self.done = done


def _assistant(with_call):
    msg = {"role": "assistant", "content": "" if with_call else "I am done."}
    if with_call:
        msg["tool_calls"] = [{"type": "function",
                              "function": {"name": "click", "arguments": {"bid": "1"}}}]
    return msg


_TOOL = {"role": "tool", "name": "click", "content": "Page now: ..."}
CAP = 100


def _ids(n):
    return [0] * n


def test_finished_episode_is_not_penalized_however_it_ended():
    out = NonTerminationPenalty(CAP)(["p", "p"], [[_assistant(True), _TOOL], []],
                                     environments=[_FakeEnv(True), _FakeEnv(True)],
                                     completion_ids=[_ids(10), _ids(CAP)])
    assert out == [0.0, 0.0]


def test_turn_cap_is_penalized():
    # Trajectory ends after a tool result: the loop ran out of turns, env not done.
    out = NonTerminationPenalty(CAP)(["p"], [[_assistant(True), _TOOL]],
                                     environments=[_FakeEnv(False)],
                                     completion_ids=[_ids(10)])
    assert out == [-1.0]


def test_undispatched_final_calls_count_as_the_turn_cap():
    out = NonTerminationPenalty(CAP)(["p"], [[_assistant(True), _TOOL, _assistant(True)]],
                                     environments=[_FakeEnv(False)],
                                     completion_ids=[_ids(10)])
    assert out == [-1.0]


def test_completion_budget_is_penalized_even_if_the_last_turn_has_no_call():
    # A completion cut off mid-turn parses as an assistant message without a
    # call; the token count, not the message shape, says it was the cap.
    out = NonTerminationPenalty(CAP)(["p"], [[_assistant(True), _TOOL, _assistant(False)]],
                                     environments=[_FakeEnv(False)],
                                     completion_ids=[_ids(CAP)])
    assert out == [-1.0]


def test_stopping_on_its_own_is_not_penalized():
    # The substitute the thesis predicts: env not done, budget left, no call.
    out = NonTerminationPenalty(CAP)(["p"], [[_assistant(True), _TOOL, _assistant(False)]],
                                     environments=[_FakeEnv(False)],
                                     completion_ids=[_ids(10)])
    assert out == [0.0]


def test_penalty_is_negative_so_positive_weight_is_a_penalty():
    out = NonTerminationPenalty(CAP)(["p"], [[_assistant(True), _TOOL]],
                                     environments=[_FakeEnv(False)],
                                     completion_ids=[_ids(1)])
    assert 4.0 * out[0] < 0


def test_missing_kwargs_raise():
    with pytest.raises(ValueError):
        NonTerminationPenalty(CAP)(["p"], [[]], environments=[_FakeEnv(False)])
    with pytest.raises(ValueError):
        NonTerminationPenalty(CAP)(["p"], [[]], completion_ids=[_ids(1)])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        NonTerminationPenalty(CAP)(["p"], [[], []], environments=[_FakeEnv(True)],
                                   completion_ids=[_ids(1), _ids(1)])


def test_string_completion_is_the_dataset_path_and_raises():
    with pytest.raises(ValueError):
        NonTerminationPenalty(CAP)(["p"], ["plain text"], environments=[_FakeEnv(False)],
                                   completion_ids=[_ids(1)])


def test_registry_entry_reads_the_budget_off_the_runner():
    from training.rewards import REWARD_REGISTRY

    class _Runner:
        def completion_budget(self):
            return 1024

    assert "non_termination" in REWARD_REGISTRY
    enabled, weight, builder = REWARD_REGISTRY["non_termination"]
    assert enabled is False and weight == 1.0
    fn = builder(None, _Runner(), {}, {})
    assert isinstance(fn, NonTerminationPenalty) and fn.max_completion_tokens == 1024


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


def test_warns_that_group_scaling_cancels_the_weight_under_naive_sum():
    from training.config_schema import warn_inert_scalars

    cfg = {"non_termination": {"enabled": True, "weight": 4.0}}
    assert any("scale_rewards" in w for w in warn_inert_scalars(cfg, "naive_sum", "group"))
    assert warn_inert_scalars(cfg, "naive_sum", "none") == []
    assert warn_inert_scalars(cfg, "naive_sum", "batch") == []
    # Task reward alone has nothing to cancel.
    assert warn_inert_scalars({"env_reward": {"enabled": True}}, "naive_sum", "group") == []
