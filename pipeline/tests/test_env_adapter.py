import inspect

from domains.reasoning_gym.adapter import ReasoningGymEnvAdapter


class _Obs:
    def __init__(self, question=None, score=None, correct_answer=None):
        self.question = question
        self.score = score
        self.correct_answer = correct_answer


class _Result:
    def __init__(self, obs):
        self.observation = obs


class _FakeAction:
    def __init__(self, answer):
        self.answer = answer


class _FakeClient:
    """Records reset/step calls; returns canned observations (no server)."""

    def __init__(self, question="Q?", score=1.0, correct_answer="7"):
        self._question = question
        self._score = score
        self._correct = correct_answer
        self.reset_calls = []
        self.step_calls = []

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return _Result(_Obs(question=self._question))

    def step(self, action):
        self.step_calls.append(action)
        return _Result(_Obs(score=self._score, correct_answer=self._correct))


def _adapter(client=None, env_config=None):
    return ReasoningGymEnvAdapter(
        base_url="http://x",
        env_config=env_config or {"dataset": "chain_sum"},
        client=client or _FakeClient(),
        action_cls=_FakeAction,
    )


def test_reset_returns_question_string():
    a = _adapter()
    out = a.reset(seed=42, prompt=[{"role": "user", "content": ""}])
    assert out == "Q?" and a.reward == 0.0


def test_reset_passes_seed_dataset_and_size_one():
    c = _FakeClient()
    a = _adapter(client=c, env_config={"dataset": "chain_sum"})
    a.reset(seed=7)
    call = c.reset_calls[0]
    assert call["seed"] == 7 and call["dataset_name"] == "chain_sum" and call["size"] == 1


def test_reset_forwards_dataset_config():
    c = _FakeClient()
    a = _adapter(client=c, env_config={"dataset": "chain_sum", "dataset_config": {"min_value": 1}})
    a.reset(seed=1)
    assert c.reset_calls[0]["dataset_config"] == {"min_value": 1}


def test_answer_sets_reward_from_score():
    c = _FakeClient(score=1.0)
    a = _adapter(client=c)
    a.reset(seed=1)
    msg = a.answer("7")
    assert a.reward == 1.0 and isinstance(msg, str)
    assert c.step_calls[0].answer == "7"


def test_answer_wrong_gives_zero_reward():
    a = _adapter(client=_FakeClient(score=0.0))
    a.reset(seed=1)
    a.answer("-1")
    assert a.reward == 0.0


def test_reset_ignores_unknown_row_keys():
    # TRL passes the whole dataset row as reset kwargs (incl. prompt + extras).
    a = _adapter()
    out = a.reset(seed=3, prompt=[{"role": "user", "content": ""}], extra="ignored")
    assert out == "Q?"


def test_minimal_public_surface_only_reset_and_answer():
    # Critical: TRL turns every public method (except reset) into a tool the
    # model can call. The adapter must expose exactly {reset, answer}.
    a = _adapter()
    public = {n for n, _ in inspect.getmembers(a, predicate=inspect.ismethod)
              if not n.startswith("_")}
    assert public == {"reset", "answer"}
