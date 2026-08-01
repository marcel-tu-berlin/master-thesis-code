import inspect

from domains.reasoning_gym.domain import ReasoningGymDomain
from domains.reasoning_gym.adapter import ReasoningGymEnvAdapter


class _FakeStep:
    """Stands in for an OpenEnv StepResult (eval-side reward reading)."""

    def __init__(self, reward):
        self.reward = reward


class _Obs:
    def __init__(self, question="Q?"):
        self.question = question


class _Result:
    def __init__(self):
        self.observation = _Obs()


class _FakeClient:
    def reset(self, **kwargs):
        return _Result()


def test_episode_reward_reads_step_result():
    assert ReasoningGymDomain().episode_reward(_FakeStep(1.0)) == 1.0


def test_is_correct_from_reward_sign():
    d = ReasoningGymDomain()
    assert d.is_correct(_FakeStep(1.0)) is True
    assert d.is_correct(_FakeStep(0.0)) is False


def test_is_correct_rejects_graded_partial_credit():
    # reasoning_gym scorers are graded: countdown gives 0.05 to a near-miss and
    # 0.01 to garbage; polynomial_equations decays to tiny-but-positive for
    # wrong roots. None of these count as solved.
    d = ReasoningGymDomain()
    assert d.is_correct(_FakeStep(0.01)) is False
    assert d.is_correct(_FakeStep(0.05)) is False
    assert d.is_correct(_FakeStep(4.5e-5)) is False
    assert d.is_correct(_FakeStep(0.5)) is True


def test_make_env_factory_is_zero_arg_and_builds_adapter():
    d = ReasoningGymDomain()
    factory = d.make_env_factory(
        "http://x", {"dataset": "chain_sum"}, client_factory=_FakeClient
    )
    assert callable(factory) and len(inspect.signature(factory).parameters) == 0
    env = factory()
    assert isinstance(env, ReasoningGymEnvAdapter)
    assert env.reset(seed=1) == "Q?"


def test_build_seed_dataset_distinct_seeds_and_prompt():
    d = ReasoningGymDomain()
    ds = d.build_seed_dataset({"dataset": "chain_sum"}, n=4, seed_base=10)
    assert len(ds) == 4
    assert [r["seed"] for r in ds] == [10, 11, 12, 13]
    assert all(r["prompt"][0]["role"] == "user" for r in ds)


def test_base_defaults_server_env_empty_and_single_turn():
    from domains.env_base import EnvDomain
    d = EnvDomain()
    assert d.server_env({"anything": 1}) == {}
    assert d.multi_turn is False


def test_reasoning_gym_eval_tools_is_answer():
    d = ReasoningGymDomain()
    factory = d.make_env_factory("http://x", {"dataset": "chain_sum"}, client_factory=_FakeClient)
    env = factory()
    tools = d.eval_tools(env)
    assert tools == [env.answer]
    assert d.multi_turn is False


# --- one turn cap across training, eval, and each server (max_turns) ---
# TRL treats an unset max_tool_calling_iterations as sys.maxsize, so a domain
# whose server cap lives under its own key leaves the training tool loop
# unbounded. Every multi-turn domain reads the same `max_turns`.

def test_finqa_server_env_maps_max_turns_to_its_step_cap():
    from domains.finqa import FinQADomain

    env = FinQADomain().server_env({"max_turns": 12})
    assert env["FINQA_MAX_STEPS"] == "12"


def test_repl_server_env_maps_max_turns_to_its_iteration_cap():
    from domains.repl import REPLDomain

    env = REPLDomain().server_env({"max_turns": 12})
    assert env["REPL_MAX_ITERATIONS"] == "12"


def test_schema_rejects_the_old_per_domain_cap_aliases():
    import pytest
    from training.config_schema import validate_config

    for stale in ("max_steps", "max_iterations"):
        with pytest.raises(ValueError, match="env_config"):
            validate_config({
                "experiment_id": "x", "model": {"slug": "qwen3-1.7b"},
                "training": {"mode": "agentic", "env": "finqa",
                             "env_config": {stale: 12}},
            })
