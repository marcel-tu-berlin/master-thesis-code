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


def test_answer_tool_schema_is_generatable():
    # TRL builds the tool spec from the answer() docstring via transformers'
    # get_json_schema, which raises unless every parameter has a Google-style
    # Args description. Guards against silently dropping that docstring section.
    try:
        from transformers.utils.chat_template_utils import get_json_schema
    except Exception:
        import pytest
        pytest.skip("transformers not available in this venv")
    fn = get_json_schema(_adapter().answer)["function"]
    assert fn["name"] == "answer"
    assert fn["parameters"]["properties"]["answer"]["description"]


# --- done-guard: the reward is the FIRST answer, not the last ---
# Without it a second `answer` call re-stepped the env and overwrote self.reward,
# so a rollout that answered correctly and then answered again trained as WRONG
# while eval - which reads the first call - scored it correct. Two plausible
# numbers, disagreeing about the same trajectory. Every sibling adapter
# (textarena.move, repl.execute, browsergym._act) guards.

class _ScoringClient(_FakeClient):
    """Scores each submitted answer against a map, so order is observable."""

    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def step(self, action):
        self.step_calls.append(action)
        return _Result(_Obs(score=self.scores.get(action.answer, 0.0)))


def test_second_answer_does_not_overwrite_the_reward():
    client = _ScoringClient({"x=3": 1.0, "x=4": 0.0})
    a = _adapter(client=client)
    a.reset(seed=1)
    a.answer("x=3")
    assert a.reward == 1.0
    a.answer("x=4")                      # the model second-guesses itself
    assert a.reward == 1.0               # first answer stands
    assert len(client.step_calls) == 1   # the env was never re-stepped


def test_second_answer_returns_a_terminated_notice():
    a = _adapter(client=_ScoringClient({"x=3": 1.0}))
    a.reset(seed=1)
    a.answer("x=3")
    assert "already finished" in a.answer("x=4")


def test_reset_reopens_the_episode():
    client = _ScoringClient({"x=3": 1.0, "x=9": 1.0})
    a = _adapter(client=client)
    a.reset(seed=1)
    a.answer("x=3")
    a.reset(seed=2)                      # next episode
    assert a.done is False and a.reward == 0.0
    a.answer("x=9")
    assert a.reward == 1.0
    assert len(client.step_calls) == 2


def test_wrong_first_answer_also_stands():
    # The guard must not be a "keep the best score" rule - it keeps the FIRST,
    # which is what eval scores.
    client = _ScoringClient({"x=3": 0.0, "x=4": 1.0})
    a = _adapter(client=client)
    a.reset(seed=1)
    a.answer("x=3")
    a.answer("x=4")
    assert a.reward == 0.0
