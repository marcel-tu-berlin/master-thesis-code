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
