"""BrowserGym adapter: the seed->task contract and the terminal-reward guard.

The two things that must not silently break are (a) reset(seed=N) picking a
deterministic (task, page) pair - every rollout slot in a GRPO group resets with
the same seed and must land on the same page, and (b) the terminal reward
surviving any tool call the model makes after the episode ends.
"""

import inspect
from types import SimpleNamespace

import pytest

from domains.browsergym.adapter import BrowserGymEnvAdapter
from domains.browsergym.domain import BrowserGymDomain


class _Obs:
    def __init__(self, goal="", axtree_txt="", error=""):
        self.goal = goal
        self.axtree_txt = axtree_txt
        self.error = error


class _Result:
    def __init__(self, obs, reward=0.0, done=False):
        self.observation = obs
        self.reward = reward
        self.done = done


class _FakeClient:
    """Records reset kwargs and pops a canned step result per action."""

    def __init__(self, steps=None, goal="Select cp and click Submit."):
        self._goal = goal
        self._steps = list(steps or [])
        self.reset_calls = []
        self.actions = []

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return _Result(_Obs(goal=self._goal, axtree_txt="[24] radio 'cp'"))

    def step(self, action):
        self.actions.append(action)
        return self._steps.pop(0) if self._steps else _Result(_Obs())


def _adapter(client, **cfg):
    a = BrowserGymEnvAdapter("http://x", cfg, client=client)
    # The real action builder lives in the OpenEnv clone; the tests run without it.
    a._make_action = lambda tool, args: (tool, args)
    return a


# --- seed -> task is deterministic and cycles the mix ---


def test_reset_forwards_seed_and_derives_task_from_it():
    c = _FakeClient()
    a = _adapter(c, tasks=["click-option", "click-checkboxes"])
    a.reset(seed=4)
    assert c.reset_calls == [{"seed": 4, "task_name": "click-option"}]


def test_seed_cycles_the_task_mix_evenly():
    c = _FakeClient()
    a = _adapter(c, tasks=["click-option", "click-checkboxes"])
    for s in range(4):
        a.reset(seed=s)
    assert [k["task_name"] for k in c.reset_calls] == [
        "click-option",
        "click-checkboxes",
        "click-option",
        "click-checkboxes",
    ]


def test_same_seed_is_the_same_task():
    # GRPO repeats one prompt across rollout slots; a group must share a page.
    c = _FakeClient()
    a = _adapter(c, tasks=["click-option", "click-checkboxes"])
    a.reset(seed=7)
    a.reset(seed=7)
    assert c.reset_calls[0] == c.reset_calls[1]


def test_default_task_mix_pairs_headroom_with_axis_separation():
    # click-option gives base-model headroom; click-checkboxes is the family whose
    # success sits below its termination rate, which is what decouples task
    # performance from the off-target axis.
    c = _FakeClient()
    a = _adapter(c)
    a.reset(seed=0)
    a.reset(seed=1)
    assert {k["task_name"] for k in c.reset_calls} == {
        "click-option",
        "click-checkboxes",
    }


def test_empty_task_list_is_rejected():
    with pytest.raises(ValueError, match="at least one task"):
        BrowserGymEnvAdapter("http://x", {"tasks": []}, client=_FakeClient())


# --- reset payload ---


def test_reset_returns_goal_and_page():
    a = _adapter(_FakeClient(goal="Select cp and click Submit."))
    text = a.reset(seed=0)
    assert "Select cp and click Submit." in text
    assert "[24] radio 'cp'" in text
    assert a.reward == 0.0 and a.done is False


# --- reward and the done guard ---


def test_click_records_terminal_reward_and_done():
    c = _FakeClient(
        steps=[_Result(_Obs(axtree_txt="submitted"), reward=1.0, done=True)]
    )
    a = _adapter(c)
    a.reset(seed=0)
    a.click(bid="24")
    assert a.reward == 1.0 and a.done is True


def test_calls_after_done_cannot_overwrite_the_reward():
    c = _FakeClient(
        steps=[
            _Result(_Obs(), reward=1.0, done=True),
            _Result(_Obs(), reward=0.0, done=True),  # must never be reached
        ]
    )
    a = _adapter(c)
    a.reset(seed=0)
    a.click(bid="24")
    out = a.click(bid="99")
    assert a.reward == 1.0
    assert "already finished" in out
    assert len(c.actions) == 1


def test_reset_clears_reward_and_done_between_episodes():
    c = _FakeClient(steps=[_Result(_Obs(), reward=1.0, done=True)])
    a = _adapter(c)
    a.reset(seed=0)
    a.click(bid="24")
    a.reset(seed=1)
    assert a.reward == 0.0 and a.done is False


def test_env_error_is_surfaced_as_feedback_not_raised():
    c = _FakeClient(steps=[_Result(_Obs(axtree_txt="page", error="bid not found"))])
    a = _adapter(c)
    a.reset(seed=0)
    out = a.click(bid="999")
    assert "bid not found" in out and "page" in out


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("via_eval", [False, True])
def test_native_action_error_reaches_training_and_eval_tools(enabled, via_eval):
    observation = SimpleNamespace(
        axtree_txt="unchanged page",
        error="",
        last_action_error=True,
        metadata={"browsergym_obs": {"last_action_error": "Element is not editable"}},
    )
    client = _FakeClient(steps=[_Result(observation)])
    domain = BrowserGymDomain()
    env = domain.make_env_factory(
        "http://x", {"enable_fill": enabled}, client_factory=lambda: client
    )()
    env._make_action = lambda tool, args: (tool, args)
    name = "fill" if enabled else "click"
    tool = (
        next(tool for tool in domain.eval_tools(env) if tool.__name__ == name)
        if via_eval
        else getattr(env, name)
    )
    arguments = {"bid": "17", "text": "value"} if enabled else {"bid": "999"}

    assert tool(**arguments) == (
        "Action error: Element is not editable\nPage now:\nunchanged page"
    )
    assert env.reward == 0 and not env.done
    assert client.actions == [(name, arguments)]


def test_native_error_flag_without_details_is_not_silent():
    observation = SimpleNamespace(
        axtree_txt="page", error="", last_action_error=True, metadata=None
    )
    env = _adapter(_FakeClient(steps=[_Result(observation)]))

    assert env.click(bid="999") == (
        "Action error: Action failed (no error details provided).\nPage now:\npage"
    )


def test_explicit_error_keeps_precedence_over_native_metadata():
    observation = SimpleNamespace(
        axtree_txt="page",
        error="wrapper error",
        last_action_error=True,
        metadata={"browsergym_obs": {"last_action_error": "native error"}},
    )
    env = _adapter(_FakeClient(steps=[_Result(observation)]))

    assert env.click(bid="999") == "Action error: wrapper error\nPage now:\npage"


def test_long_observation_is_truncated():
    c = _FakeClient(steps=[_Result(_Obs(axtree_txt="x" * 5000))])
    a = _adapter(c)
    a.reset(seed=0)
    out = a.click(bid="1")
    assert "[truncated]" in out and len(out) < 2500


# --- the tool surface TRL will expose ---


def test_public_surface_is_exactly_reset_click_noop():
    # TRL turns every public method except reset into a tool, so an accidental
    # public helper would silently become a tool the model can call.
    public = {
        n
        for n, _ in inspect.getmembers(BrowserGymEnvAdapter, inspect.isfunction)
        if not n.startswith("_")
    }
    assert public == {"reset", "click", "noop"}


def test_click_docstring_has_the_args_block_the_tool_schema_needs():
    # transformers' get_json_schema raises without a Google-style Args entry.
    assert "Args:" in BrowserGymEnvAdapter.click.__doc__
    assert "bid:" in BrowserGymEnvAdapter.click.__doc__


def test_noop_is_a_countable_stall_not_an_absence():
    c = _FakeClient(steps=[_Result(_Obs(axtree_txt="unchanged"))])
    a = _adapter(c)
    a.reset(seed=0)
    a.noop()
    assert len(c.actions) == 1 and a.done is False


@pytest.mark.parametrize("enabled", [False, True])
def test_fill_opt_in_preserves_training_eval_tool_parity(enabled):
    client = _FakeClient(steps=[_Result(_Obs(), reward=1.0, done=True)])
    domain = BrowserGymDomain()
    env = domain.make_env_factory(
        "http://x", {"enable_fill": enabled}, client_factory=lambda: client
    )()
    training_tools = {
        name
        for name, _ in inspect.getmembers(env, inspect.ismethod)
        if not name.startswith("_") and name != "reset"
    }
    expected = {"click", "noop", "fill"} if enabled else {"click", "noop"}
    assert training_tools == {t.__name__ for t in domain.eval_tools(env)} == expected
    if enabled:
        env._make_action = lambda tool, args: (tool, args)
        env.fill(bid="33", text="O'Brien\\value")
        assert client.actions == [("fill", {"bid": "33", "text": "O'Brien\\value"})]
        assert env.reward == 1.0 and env.done
        env.fill(bid="36", text="later")
        assert len(client.actions) == 1 and env.reward == 1.0


def test_fill_factory_rejects_truthy_non_boolean():
    with pytest.raises(ValueError, match=r"enable_fill.*bool"):
        BrowserGymDomain().make_env_factory("http://x", {"enable_fill": "false"})
