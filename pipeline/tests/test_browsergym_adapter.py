"""BrowserGym adapter: the seed->task contract and the terminal-reward guard.

The two things that must not silently break are (a) reset(seed=N) picking a
deterministic (task, page) pair - every rollout slot in a GRPO group resets with
the same seed and must land on the same page, and (b) the terminal reward
surviving any tool call the model makes after the episode ends.
"""
import inspect

import pytest

from domains.browsergym.adapter import BrowserGymEnvAdapter


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
        "click-option", "click-checkboxes", "click-option", "click-checkboxes"]


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
    assert {k["task_name"] for k in c.reset_calls} == {"click-option", "click-checkboxes"}


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
    c = _FakeClient(steps=[_Result(_Obs(axtree_txt="submitted"), reward=1.0, done=True)])
    a = _adapter(c)
    a.reset(seed=0)
    a.click(bid="24")
    assert a.reward == 1.0 and a.done is True


def test_calls_after_done_cannot_overwrite_the_reward():
    c = _FakeClient(steps=[
        _Result(_Obs(), reward=1.0, done=True),
        _Result(_Obs(), reward=0.0, done=True),   # must never be reached
    ])
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
    public = {n for n, _ in inspect.getmembers(BrowserGymEnvAdapter, inspect.isfunction)
              if not n.startswith("_")}
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
