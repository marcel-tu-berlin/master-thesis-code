from eval.agentic_eval import _parse_tool_call, _run_multiturn_episodes


# --- _parse_tool_call: first valid tool call of any name ---

def test_parse_tool_call_returns_name_and_args():
    t = '<tool_call>{"name": "move", "arguments": {"message": "crane"}}</tool_call>'
    assert _parse_tool_call(t) == ("move", {"message": "crane"})


def test_parse_tool_call_with_think_prefix():
    t = '<think>hmm</think><tool_call>{"name":"move","arguments":{"message":"slate"}}</tool_call>'
    assert _parse_tool_call(t) == ("move", {"message": "slate"})


def test_parse_tool_call_none_without_call():
    assert _parse_tool_call("no tool call here") is None


def test_parse_tool_call_none_on_malformed_json():
    assert _parse_tool_call("<tool_call>{not json}</tool_call>") is None


# --- _run_multiturn_episodes: drive a scripted game with an injected turn fn ---

class _FakeGameEnv:
    """Solves when move == solution; self-dones on solve or after fail_after moves."""

    def __init__(self, solution, fail_after=6):
        self.solution = solution
        self.fail_after = fail_after
        self.reward = 0.0
        self.done = False
        self.moves = 0
        self.resets = []

    def reset(self, seed=None, **_):
        self.resets.append(seed)
        self.reward = 0.0
        self.done = False
        self.moves = 0
        return "start"

    def move(self, message):
        self.moves += 1
        if message == self.solution:
            self.reward = 1.0
            self.done = True
            return "correct"
        if self.moves >= self.fail_after:
            self.done = True
            return "out of guesses"
        return "wrong, try again"


def _msgs(o):
    return [{"role": "user", "content": o}]


def test_multiturn_solves_in_two_turns():
    env = _FakeGameEnv("slate")
    scripted = iter([("move", {"message": "crane"}, 10), ("move", {"message": "slate"}, 8)])
    rs = _run_multiturn_episodes(env, 1, 0, lambda m: next(scripted),
                                 max_turns=6, make_messages=_msgs)
    assert rs[0].correct is True and rs[0].n_steps == 2 and rs[0].n_tokens == 18
    assert env.resets == [0]


def test_multiturn_stops_when_model_stops_calling_move():
    env = _FakeGameEnv("slate")
    scripted = iter([("move", {"message": "crane"}, 5), (None, None, 3)])
    rs = _run_multiturn_episodes(env, 1, 0, lambda m: next(scripted),
                                 max_turns=6, make_messages=_msgs)
    # Turn 1 is a move (counted, stepped); turn 2 is no-move -> counted then stop.
    assert rs[0].n_steps == 1 and rs[0].n_tokens == 8 and rs[0].correct is False


def test_multiturn_caps_at_max_turns():
    env = _FakeGameEnv("zzzzz", fail_after=99)  # never solves, never early-dones
    scripted = iter([("move", {"message": "aaaaa"}, 4)] * 10)
    rs = _run_multiturn_episodes(env, 1, 0, lambda m: next(scripted),
                                 max_turns=3, make_messages=_msgs)
    assert rs[0].n_steps == 3 and rs[0].n_tokens == 12 and rs[0].correct is False


def test_multiturn_appends_assistant_and_tool_messages():
    env = _FakeGameEnv("slate")
    seen = []

    def turn_fn(messages):
        seen.append([m["role"] for m in messages])
        return ("move", {"message": "slate"}, 7)

    _run_multiturn_episodes(env, 1, 0, turn_fn, max_turns=6, make_messages=_msgs)
    # First turn sees just the user lead-in; episode ends on the solving move.
    assert seen == [["user"]]
