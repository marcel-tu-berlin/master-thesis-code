from eval.agentic_eval import _run_multiturn_episodes, _tool_calls


# --- _tool_calls: every call, in order, off a parsed assistant message ---
# The message shape is what trl.chat_template_utils.parse_response returns - the
# same function TRL applies to a rollout during training. Eval used to re-parse
# the decoded text with its own regex, which is how the two ended up disagreeing.

def _assistant(name=None, args=None, content="", reasoning=None, extra=()):
    """A parse_response-shaped assistant message.

    `extra` appends further (name, args) calls, for the multi-call turns that
    browsergym produces routinely.
    """
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    if name is not None:
        msg["tool_calls"] = [{"type": "function",
                              "function": {"name": name, "arguments": args or {}}}]
        msg["tool_calls"] += [{"type": "function",
                               "function": {"name": n, "arguments": a or {}}}
                              for n, a in extra]
    return msg


def test_tool_calls_returns_name_and_args():
    msg = _assistant("move", {"message": "crane"})
    assert _tool_calls(msg) == [("move", {"message": "crane"})]


def test_tool_calls_ignores_reasoning():
    # A JSON object quoted inside the think block is reasoning, not a call.
    msg = _assistant("move", {"message": "slate"},
                     reasoning='maybe {"name": "answer", "arguments": {"answer": "0"}}')
    assert _tool_calls(msg) == [("move", {"message": "slate"})]


def test_tool_calls_empty_without_call():
    assert _tool_calls(_assistant(content="no tool call here")) == []


def test_tool_calls_empty_on_empty_tool_calls():
    assert _tool_calls({"role": "assistant", "tool_calls": []}) == []


def test_tool_calls_non_dict_args_read_as_empty():
    msg = {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "move", "arguments": "prestringified"}}]}
    assert _tool_calls(msg) == [("move", {})]


def test_tool_calls_returns_every_call_in_order():
    # Not just the first. TRL executes all of them during training, and a turn
    # requesting four clicks is the normal shape on browsergym.
    msg = _assistant("click", {"bid": "30"},
                     extra=[("click", {"bid": "24"}), ("click", {"bid": "36"})])
    assert _tool_calls(msg) == [("click", {"bid": "30"}),
                                ("click", {"bid": "24"}),
                                ("click", {"bid": "36"})]


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
    """A scripted turn_fn result: (message, [(name, args), ...], tokens)."""
    msg = _assistant(name, args, **kw)
    return msg, _tool_calls(msg), n_tok


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


# --- a turn requesting several calls dispatches all of them ---
# The failure this guards against, measured on seed 200051 of e27's shifted
# split: the model emitted four clicks in one turn, the loop dispatched the
# first and appended the message advertising all four, and the next turn saw
# four tool calls answered by one tool response. The model read the three
# unanswered ones as having succeeded, announced the task complete, and stopped.
# 19 of 100 shifted episodes were lost that way, every one a loss and none a
# gain. TRL dispatches every call during training (grpo_trainer.py, "Call the
# tools, and build the new prompt"), so this is also what the policy trained on.

class _ClickEnv:
    """Done once every required bid has been clicked, in any order."""

    def __init__(self, required):
        self.required, self.clicked, self.done = set(required), set(), False

    def reset(self, seed=0):
        self.clicked, self.done = set(), False
        return "page"

    def click(self, bid):
        self.clicked.add(bid)
        self.done = self.required <= self.clicked
        return f"clicked {bid}"

    @property
    def reward(self):
        return 1.0 if self.done else 0.0


def test_every_call_in_a_turn_is_dispatched():
    env = _ClickEnv({"30", "24", "36"})
    rs = _run_multiturn_episodes(
        env, 1, 0,
        _scripted(_turn("click", {"bid": "30"}, 40,
                        extra=[("click", {"bid": "24"}), ("click", {"bid": "36"})])),
        max_turns=8, make_messages=_msgs, tool_names={"click"})
    assert env.clicked == {"30", "24", "36"}
    assert rs[0].correct is True and rs[0].stop_reason == "env_done"
    assert rs[0].n_steps == 3          # three calls reached the env, in one turn
    assert rs[0].tool_calls == ["click", "click", "click"]


def test_each_dispatched_call_gets_its_own_tool_response():
    # N calls advertised, N responses. The transcript the next turn sees must not
    # leave a call unanswered, or the model treats it as having succeeded.
    env = _ClickEnv({"30", "24", "99"})   # never completes, so the loop runs on
    seen = []

    def turn_fn(messages, budget):
        seen.append(list(messages))
        return _turn("click", {"bid": "30"}, 10, extra=[("click", {"bid": "24"})])

    _run_multiturn_episodes(env, 1, 0, turn_fn, max_turns=2,
                            make_messages=_msgs, tool_names={"click"})
    roles = [m["role"] for m in seen[1]]
    assert roles == ["user", "assistant", "tool", "tool"]


def test_calls_after_the_env_is_done_are_not_dispatched():
    # A turn may request more actions than the episode needs. Once the env is
    # done the rest must not run into a finished env.
    env = _ClickEnv({"30"})
    rs = _run_multiturn_episodes(
        env, 1, 0,
        _scripted(_turn("click", {"bid": "30"}, 12, extra=[("click", {"bid": "99"})])),
        max_turns=8, make_messages=_msgs, tool_names={"click"})
    assert env.clicked == {"30"}
    assert rs[0].n_steps == 1 and rs[0].stop_reason == "env_done"


def test_unknown_names_are_filtered_but_known_ones_still_run():
    env = _ClickEnv({"30"})
    rs = _run_multiturn_episodes(
        env, 1, 0,
        _scripted(_turn("reset", {}, 9, extra=[("click", {"bid": "30"})])),
        max_turns=8, make_messages=_msgs, tool_names={"click"})
    assert env.clicked == {"30"} and rs[0].n_steps == 1


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
