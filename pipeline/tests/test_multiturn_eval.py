from eval.agentic_eval import _first_tool_call, _run_multiturn_episodes


# --- _first_tool_call: first call of any name, off a parsed assistant message ---
# The message shape is what trl.chat_template_utils.parse_response returns - the
# same function TRL applies to a rollout during training. Eval used to re-parse
# the decoded text with its own regex, which is how the two ended up disagreeing.

def _assistant(name=None, args=None, content="", reasoning=None):
    """A parse_response-shaped assistant message."""
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    if name is not None:
        msg["tool_calls"] = [{"type": "function",
                              "function": {"name": name, "arguments": args or {}}}]
    return msg


def test_first_tool_call_returns_name_and_args():
    msg = _assistant("move", {"message": "crane"})
    assert _first_tool_call(msg) == ("move", {"message": "crane"})


def test_first_tool_call_ignores_reasoning():
    # A JSON object quoted inside the think block is reasoning, not a call.
    msg = _assistant("move", {"message": "slate"},
                     reasoning='maybe {"name": "answer", "arguments": {"answer": "0"}}')
    assert _first_tool_call(msg) == ("move", {"message": "slate"})


def test_first_tool_call_none_without_call():
    assert _first_tool_call(_assistant(content="no tool call here")) is None


def test_first_tool_call_none_on_empty_tool_calls():
    assert _first_tool_call({"role": "assistant", "tool_calls": []}) is None


def test_first_tool_call_non_dict_args_read_as_empty():
    msg = {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "move", "arguments": "prestringified"}}]}
    assert _first_tool_call(msg) == ("move", {})


def test_first_tool_call_takes_the_first_of_several():
    msg = _assistant("move", {"message": "a"})
    msg["tool_calls"].append(
        {"type": "function", "function": {"name": "quit", "arguments": {}}})
    assert _first_tool_call(msg) == ("move", {"message": "a"})


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


def _turn(name, args, n_tok, **kw):
    """A scripted turn_fn result: (message, tool name, args, tokens)."""
    return _assistant(name, args, **kw), name, args, n_tok


def _scripted(*turns):
    """turn_fn that replays `turns`, ignoring messages and budget."""
    it = iter(turns)
    return lambda messages, budget: next(it)


def test_multiturn_solves_in_two_turns():
    env = _FakeGameEnv("slate")
    rs = _run_multiturn_episodes(
        env, 1, 0, _scripted(_turn("move", {"message": "crane"}, 10),
                             _turn("move", {"message": "slate"}, 8)),
        max_turns=6, make_messages=_msgs, tool_names={"move"})
    assert rs[0].correct is True and rs[0].n_steps == 2 and rs[0].n_tokens == 18
    assert env.resets == [0]


def test_multiturn_stops_when_model_stops_calling_move():
    env = _FakeGameEnv("slate")
    rs = _run_multiturn_episodes(
        env, 1, 0, _scripted(_turn("move", {"message": "crane"}, 5),
                             _turn(None, None, 3)),
        max_turns=6, make_messages=_msgs, tool_names={"move"})
    # Turn 1 is a move (counted, stepped); turn 2 is no-move -> counted then stop.
    assert rs[0].n_steps == 1 and rs[0].n_tokens == 8 and rs[0].correct is False


def test_multiturn_caps_at_max_turns():
    env = _FakeGameEnv("zzzzz", fail_after=99)  # never solves, never early-dones
    rs = _run_multiturn_episodes(
        env, 1, 0, _scripted(*[_turn("move", {"message": "aaaaa"}, 4)] * 10),
        max_turns=3, make_messages=_msgs, tool_names={"move"})
    assert rs[0].n_steps == 3 and rs[0].n_tokens == 12 and rs[0].correct is False


def test_multiturn_appends_assistant_and_tool_messages():
    env = _FakeGameEnv("slate")
    seen = []

    def turn_fn(messages, budget):
        seen.append([m["role"] for m in messages])
        return _turn("move", {"message": "slate"}, 7)

    _run_multiturn_episodes(env, 1, 0, turn_fn, max_turns=6,
                            make_messages=_msgs, tool_names={"move"})
    # First turn sees just the user lead-in; episode ends on the solving move.
    assert seen == [["user"]]


def test_multiturn_keeps_the_models_own_reasoning_in_context():
    # The loop used to rebuild the turn as {"content": ""} plus a tool_calls
    # stub, so turn N+1 ran without the reasoning that produced turn N - a
    # context training never generates, since TRL keeps every prior turn's text.
    env = _FakeGameEnv("slate")
    seen = []

    def turn_fn(messages, budget):
        seen.append(list(messages))
        if len(seen) == 1:
            return _turn("move", {"message": "crane"}, 5, reasoning="crane splits vowels")
        return _turn("move", {"message": "slate"}, 5)

    _run_multiturn_episodes(env, 1, 0, turn_fn, max_turns=6,
                            make_messages=_msgs, tool_names={"move"})
    assert len(seen) == 2
    assistant = [m for m in seen[1] if m.get("role") == "assistant"]
    assert assistant and assistant[0].get("reasoning_content") == "crane splits vowels"


# --- the generation budget covers the whole trajectory, not each turn ---
# Training caps the full completion at max_completion_length. An eval that
# renewed the budget every turn let an episode generate max_turns times what the
# policy trained under, and hit_generation_cap could never fire for the real cap.

def test_budget_is_spent_across_turns_not_renewed():
    env = _FakeGameEnv("zzzzz", fail_after=99)
    budgets = []

    def turn_fn(messages, budget):
        budgets.append(budget)
        return _turn("move", {"message": "aaaaa"}, 300)

    rs = _run_multiturn_episodes(env, 1, 0, turn_fn, max_turns=8,
                                 make_messages=_msgs, tool_names={"move"},
                                 gen_cap=1000)
    assert budgets == [1000, 700, 400, 100]          # never renewed
    assert rs[0].stop_reason == "hit_generation_cap"
    assert rs[0].n_tokens == 1200                    # the 4th turn overruns, then stop


def test_no_budget_means_no_cap():
    env = _FakeGameEnv("zzzzz", fail_after=99)
    budgets = []

    def turn_fn(messages, budget):
        budgets.append(budget)
        return _turn("move", {"message": "aaaaa"}, 300)

    rs = _run_multiturn_episodes(env, 1, 0, turn_fn, max_turns=3,
                                 make_messages=_msgs, tool_names={"move"})
    assert budgets == [None, None, None]
    assert rs[0].stop_reason == "max_turns"


def test_turn_that_exhausts_the_budget_without_a_call_is_a_cap_hit():
    env = _FakeGameEnv("zzzzz", fail_after=99)
    rs = _run_multiturn_episodes(
        env, 1, 0, _scripted(_turn(None, None, 500)),
        max_turns=4, make_messages=_msgs, tool_names={"move"}, gen_cap=500)
    assert rs[0].stop_reason == "hit_generation_cap"


def test_short_turn_without_a_call_is_not_a_cap_hit():
    env = _FakeGameEnv("zzzzz", fail_after=99)
    rs = _run_multiturn_episodes(
        env, 1, 0, _scripted(_turn(None, None, 3)),
        max_turns=4, make_messages=_msgs, tool_names={"move"}, gen_cap=500)
    assert rs[0].stop_reason == "no_tool_call"


# --- an env-side error must not discard the split ---
# Only TypeError used to be caught, so an HTTP error / reset connection /
# malformed observation propagated out and killed the process before any
# episode record was written.

class _ExplodingEnv:
    """Env whose tool raises something that is not a TypeError."""

    def __init__(self, exc=RuntimeError("env server gone")):
        self.reward = 0.0
        self.done = False
        self.exc = exc

    def reset(self, seed=None, **_):
        self.reward = 0.0
        self.done = False
        return "obs"

    def move(self, action):
        raise self.exc


def test_env_error_becomes_feedback_not_a_crash():
    env = _ExplodingEnv()
    rs = _run_multiturn_episodes(
        env, 2, 0, _scripted(*[_turn("move", {"action": "x"}, 5)] * 4),
        max_turns=2, make_messages=_msgs, tool_names={"move"})
    assert len(rs) == 2                       # both episodes survived
    assert all(r.stop_reason == "max_turns" for r in rs)


def test_failed_dispatch_is_not_counted_as_a_step():
    # calls.append used to run even when the call never reached the env, which
    # inflated n_steps and mean_verification_depth in the RQ2 panel.
    env = _ExplodingEnv()
    rs = _run_multiturn_episodes(
        env, 1, 0, _scripted(*[_turn("move", {"action": "x"}, 5)] * 3),
        max_turns=3, make_messages=_msgs, tool_names={"move"})
    assert rs[0].n_steps == 0
    assert rs[0].tool_calls == []


def test_on_result_fires_per_episode():
    # The durability hook: records are written as they are produced, so a crash
    # part-way through a split keeps what came before it.
    env = _ExplodingEnv()
    seen = []
    _run_multiturn_episodes(
        env, 3, 100, _scripted(*[_turn(None, None, 1)] * 3),
        max_turns=1, make_messages=_msgs, tool_names={"move"},
        on_result=lambda i, r: seen.append(i))
    assert seen == [0, 1, 2]
