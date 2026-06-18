import inspect

from domains.textarena.adapter import TextArenaEnvAdapter


class _Obs:
    def __init__(self, prompt=None, reward=0.0, done=False, messages=None):
        self.prompt = prompt
        self.reward = reward
        self.done = done
        self.messages = messages


class _Result:
    def __init__(self, obs):
        self.observation = obs


class _FakeAction:
    def __init__(self, message):
        self.message = message


class _FakeClient:
    """Scripts a short game: reset -> opening prompt; each step pops a canned obs."""

    def __init__(self, open_prompt="Guess a 5-letter word.", steps=None):
        self._open = open_prompt
        self._steps = list(steps or [])
        self.reset_calls = []
        self.step_calls = []

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return _Result(_Obs(prompt=self._open))

    def step(self, action):
        self.step_calls.append(action)
        return _Result(self._steps.pop(0))


def _adapter(client=None, env_config=None):
    return TextArenaEnvAdapter(
        base_url="http://x",
        env_config=env_config or {"env_id": "Wordle-v0"},
        client=client or _FakeClient(),
        action_cls=_FakeAction,
    )


def test_reset_returns_prompt_and_zeroes_state():
    a = _adapter()
    out = a.reset(seed=42, prompt=[{"role": "user", "content": ""}])
    assert out == "Guess a 5-letter word." and a.reward == 0.0 and a.done is False


def test_reset_passes_seed():
    c = _FakeClient()
    a = _adapter(client=c)
    a.reset(seed=7)
    assert c.reset_calls[0].get("seed") == 7


def test_move_steps_sets_reward_done_and_returns_feedback():
    c = _FakeClient(steps=[_Obs(messages=[{"role": "user", "content": "wrong, try again"}],
                                reward=0.0, done=False)])
    a = _adapter(client=c)
    a.reset(seed=1)
    fb = a.move("crane")
    assert fb == "wrong, try again" and a.reward == 0.0 and a.done is False
    assert c.step_calls[0].message == "crane"


class _Msg:
    """Mimics a TextArenaMessage (pydantic): carries .content, is NOT a dict."""
    def __init__(self, content):
        self.content = content


def test_move_feedback_reads_message_content_attribute():
    # The live env returns messages as TextArenaMessage objects (.content attr),
    # not dicts - _feedback_text must read the attribute, not str() the object.
    c = _FakeClient(steps=[_Obs(messages=[_Msg("guess feedback here")], reward=0.0, done=False)])
    a = _adapter(client=c)
    a.reset(seed=1)
    assert a.move("crane") == "guess feedback here"


def test_move_terminal_sets_reward_and_done():
    c = _FakeClient(steps=[_Obs(prompt="You win!", reward=1.0, done=True)])
    a = _adapter(client=c)
    a.reset(seed=1)
    a.move("slate")
    assert a.reward == 1.0 and a.done is True


def test_move_done_guard_blocks_after_terminal():
    c = _FakeClient(steps=[_Obs(prompt="You win!", reward=1.0, done=True)])
    a = _adapter(client=c)
    a.reset(seed=1)
    a.move("slate")                       # terminal
    assert len(c.step_calls) == 1
    msg = a.move("crane")                  # blocked: no further step
    assert "over" in msg.lower() and len(c.step_calls) == 1
    assert a.reward == 1.0 and a.done is True


def test_minimal_public_surface_only_reset_and_move():
    a = _adapter()
    public = {n for n, _ in inspect.getmembers(a, predicate=inspect.ismethod)
              if not n.startswith("_")}
    assert public == {"reset", "move"}


def test_move_tool_schema_is_generatable():
    try:
        from transformers.utils.chat_template_utils import get_json_schema
    except Exception:
        import pytest
        pytest.skip("transformers not available in this venv")
    fn = get_json_schema(_adapter().move)["function"]
    assert fn["name"] == "move"
    assert fn["parameters"]["properties"]["message"]["description"]
