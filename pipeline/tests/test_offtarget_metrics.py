"""Off-target panel (RQ2): stop-reason classification and the derived rates.

The panel is the whole point of the reward-bias-substitution direction, so the
two things that must not silently break are (a) a truncated episode is never
labelled the same as one where the model simply stopped, and (b) a completion
claim with no supporting tool call is counted.
"""
from eval.agentic_eval import _no_call_reason, _run_episodes, _run_multiturn_episodes
from eval.metrics import SampleResult, compute_metrics


# --- _no_call_reason: budget artifact vs behavior ---

def test_no_call_reason_flags_cap_when_budget_exhausted():
    assert _no_call_reason(1024, 1024) == "hit_generation_cap"


def test_no_call_reason_is_behavior_below_cap():
    assert _no_call_reason(300, 1024) == "no_tool_call"


def test_no_call_reason_without_cap_is_behavior():
    # No budget known -> cannot claim truncation.
    assert _no_call_reason(999_999, None) == "no_tool_call"


# --- single-step: a missing tool call is not a wrong answer ---

class _FakeEnv:
    def __init__(self, scores):
        self.scores = scores
        self.reward = 0.0

    def reset(self, seed=None, **_):
        self.reward = 0.0
        return f"q{seed}"

    def answer(self, answer):
        self.reward = float(self.scores.get(answer, 0.0))


def test_single_step_answer_terminates():
    rs = _run_episodes(_FakeEnv({"7": 1.0}), n=1, seed_base=0,
                       gen_fn=lambda q: ("7", 10), gen_cap=1024)
    assert rs[0].terminated is True
    assert rs[0].stop_reason == "env_done"
    assert rs[0].tool_calls == ["answer"]


def test_single_step_no_answer_is_non_termination_not_wrong_answer():
    rs = _run_episodes(_FakeEnv({}), n=1, seed_base=0,
                       gen_fn=lambda q: (None, 42), gen_cap=1024)
    assert rs[0].correct is False          # still scored as a failure
    assert rs[0].terminated is False       # but distinguishable from a wrong answer
    assert rs[0].stop_reason == "no_tool_call"
    assert rs[0].tool_calls == []


def test_single_step_truncation_is_labelled_as_the_cap():
    rs = _run_episodes(_FakeEnv({}), n=1, seed_base=0,
                       gen_fn=lambda q: (None, 1024), gen_cap=1024)
    assert rs[0].stop_reason == "hit_generation_cap"


# --- multi-turn: exit reasons and the tool sequence ---

class _FakeToolEnv:
    """Two tools: `read` gathers evidence, `submit` ends the episode."""

    def __init__(self, correct_answer="42"):
        self.correct_answer = correct_answer
        self.reward = 0.0
        self.done = False

    def reset(self, seed=None, **_):
        self.reward = 0.0
        self.done = False
        return "task"

    def read(self):
        return "some data"

    def submit(self, answer):
        self.reward = 1.0 if answer == self.correct_answer else 0.0
        self.done = True
        return "submitted"


_TOOLS = {"read", "submit"}


def _msgs(o):
    return [{"role": "user", "content": o}]


def _turn(name, args, n_tok):
    """A turn_fn result: (parsed assistant message, [(name, args), ...], tokens)."""
    msg = {"role": "assistant", "content": ""}
    calls = []
    if name is not None:
        msg["tool_calls"] = [{"type": "function",
                              "function": {"name": name, "arguments": args or {}}}]
        calls = [(name, args or {})]
    return msg, calls, n_tok


def _run(scripted, max_turns=6, gen_cap=1024):
    turns = iter([_turn(*t) for t in scripted])
    return _run_multiturn_episodes(
        _FakeToolEnv(), 1, 0, lambda m, budget: next(turns),
        max_turns=max_turns, make_messages=_msgs, tool_names=_TOOLS, gen_cap=gen_cap,
    )


def test_multiturn_records_tool_sequence_and_env_done():
    rs = _run(iter([("read", {}, 5), ("submit", {"answer": "42"}, 7)]))
    assert rs[0].terminated is True
    assert rs[0].stop_reason == "env_done"
    assert rs[0].tool_calls == ["read", "submit"]
    assert rs[0].n_steps == 2


def test_multiturn_max_turns_is_non_termination():
    rs = _run(iter([("read", {}, 5)] * 10), max_turns=3)
    assert rs[0].terminated is False
    assert rs[0].stop_reason == "max_turns"
    assert rs[0].tool_calls == ["read", "read", "read"]


def test_multiturn_truncated_turn_is_the_cap_not_a_stop():
    rs = _run(iter([("read", {}, 5), (None, None, 1024)]))
    assert rs[0].stop_reason == "hit_generation_cap"
    assert rs[0].terminated is False


# --- the panel itself ---

def _ep(correct, terminated, stop_reason, tool_calls):
    return SampleResult(correct=correct, n_tokens=100, n_steps=len(tool_calls),
                        reward=1.0 if correct else 0.0, terminated=terminated,
                        stop_reason=stop_reason, tool_calls=tool_calls)


def test_panel_rates():
    m = compute_metrics([
        _ep(True, True, "env_done", ["read", "read", "submit"]),   # verified, depth 2
        _ep(True, True, "env_done", ["submit"]),                   # bare claim, depth 0
        _ep(False, False, "max_turns", ["read", "read"]),          # never finished
        _ep(False, False, "hit_generation_cap", []),               # truncated
    ])
    assert m.non_termination_rate == 0.5
    # Denominator is TERMINATED episodes only: an episode that ran out of turns
    # never got the chance to claim completion.
    assert m.unsupported_claim_rate == 0.5
    assert m.mean_verification_depth == 1.0
    assert m.stop_reasons == {"env_done": 2, "hit_generation_cap": 1, "max_turns": 1}
    # Wilson, so a boundary rate would still carry a real interval.
    assert m.non_termination_rate_ci_low < 0.5 < m.non_termination_rate_ci_high


def test_panel_absent_when_episodes_carry_no_termination_record():
    # An old report must not read as "zero non-termination".
    m = compute_metrics([SampleResult(True, 10, n_steps=1, reward=1.0)])
    assert m.non_termination_rate is None
    assert m.unsupported_claim_rate is None
    assert m.stop_reasons == {}


def test_panel_single_tool_domain_is_degenerate_but_honest():
    # reasoning_gym has one tool, so every terminated episode is a "bare claim".
    # The number is meaningless there by construction, not wrong.
    m = compute_metrics([_ep(True, True, "env_done", ["answer"]) for _ in range(3)])
    assert m.unsupported_claim_rate == 1.0
    assert m.mean_verification_depth == 0.0


# --- protocol splits (expose section 4.2: held-out AND shifted) ---

from eval.agentic_eval import _resolve_splits


def _cfg(splits=None, n=100):
    ag = {"n_episodes": n}
    if splits is not None:
        ag["splits"] = splits
    return {"seed": 42,
            "training": {"env_config": {"dataset": "polynomial_equations", "size": 500}},
            "eval": {"agentic": ag}}


def test_default_is_the_single_agentic_split():
    # Back-compat: every existing report, plot and batch skip predicate keys on it.
    splits = _resolve_splits(_cfg(), 100)
    assert [s["name"] for s in splits] == ["agentic"]
    assert splits[0]["env_config"]["dataset"] == "polynomial_equations"
    assert splits[0]["n_episodes"] == 100


def test_split_env_config_is_merged_over_the_training_one():
    splits = _resolve_splits(_cfg([
        {"name": "held_out"},
        {"name": "shifted", "env_config": {"dataset": "countdown"}, "n_episodes": 50},
    ]), 100)
    assert [s["name"] for s in splits] == ["held_out", "shifted"]
    # Unspecified keys survive the override.
    assert splits[1]["env_config"] == {"dataset": "countdown", "size": 500}
    assert splits[1]["n_episodes"] == 50
    assert splits[0]["n_episodes"] == 100


def test_split_seed_offset_moves_the_question_range():
    splits = _resolve_splits(_cfg([{"name": "far", "seed_offset": 200000}]), 10)
    assert splits[0]["seed_offset"] == 200000


def test_schema_rejects_unnamed_and_duplicate_splits():
    import pytest
    from training.config_schema import validate_config

    base = {"experiment_id": "x", "model": {"slug": "qwen3-1.7b"},
            "training": {"mode": "agentic", "env": "reasoning_gym"}}
    for splits, match in (
        ([{"n_episodes": 10}], "missing `name`"),
        ([{"name": "a"}, {"name": "a"}], "Duplicate"),
        ([{"name": "a", "env_config": {"datsaet": "x"}}], "env_config"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_config({**base, "eval": {"agentic": {"splits": splits}}})


def test_schema_accepts_a_valid_two_split_protocol():
    from training.config_schema import validate_config

    validate_config({
        "experiment_id": "x", "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "agentic", "env": "reasoning_gym"},
        "eval": {"agentic": {"n_episodes": 100, "splits": [
            {"name": "held_out"},
            {"name": "shifted", "env_config": {"dataset": "countdown"},
             "seed_offset": 200000},
        ]}},
    })
