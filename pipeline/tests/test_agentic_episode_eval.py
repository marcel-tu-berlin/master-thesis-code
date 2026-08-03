from eval.agentic_eval import (
    _answer_from, _run_episodes, _metrics_to_dict, _completion_budget,
)
from eval.metrics import SampleResult, compute_metrics


# --- _completion_budget: eval must match the training generation budget ---

def test_completion_budget_defaults_to_training_budget():
    # max_seq 2048, default max_prompt = 1024 -> completion budget 1024 (NOT 512).
    cfg = {"model": {"max_seq_length": 2048}, "training": {}}
    assert _completion_budget(cfg, 2048) == 1024


def test_completion_budget_respects_explicit_override():
    cfg = {"model": {"max_seq_length": 2048}, "eval": {"max_new_tokens": 700}}
    assert _completion_budget(cfg, 2048) == 700


def test_completion_budget_honors_max_prompt_length():
    cfg = {"model": {"max_seq_length": 2048}, "training": {"max_prompt_length": 256}}
    assert _completion_budget(cfg, 2048) == 1792


# --- _answer_from: the answer argument of a parsed assistant message ---
# The message shape is trl.chat_template_utils.parse_response's output, so eval
# and training agree on what counts as a call by construction. The wire-format
# handling (Hermes tags vs Llama's untagged `parameters`, truncated calls,
# JSON quoted inside a think block) lives in that function and is covered
# end-to-end in test_response_parsing.py, which needs a real tokenizer.

def _answer_msg(value, name="answer"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": name, "arguments": {"answer": value}}}]}


def test_answer_from_simple():
    assert _answer_from(_answer_msg("42")) == "42"


def test_answer_from_coerces_non_string():
    assert _answer_from(_answer_msg(42)) == "42"


def test_answer_from_takes_first_answer_call():
    msg = _answer_msg("7")
    msg["tool_calls"].append({"type": "function",
                              "function": {"name": "answer", "arguments": {"answer": "8"}}})
    assert _answer_from(msg) == "7"


def test_answer_from_none_without_tool_call():
    assert _answer_from({"role": "assistant", "content": "the answer is 42"}) is None


def test_answer_from_ignores_other_tools():
    assert _answer_from(_answer_msg("42", name="other")) is None


def test_answer_from_none_when_argument_missing():
    msg = {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "answer", "arguments": {}}}]}
    assert _answer_from(msg) is None


def test_answer_from_none_on_non_dict_arguments():
    msg = {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "answer", "arguments": "42"}}]}
    assert _answer_from(msg) is None


# --- _run_episodes: drive env reset/score with an injected generator ---

class _FakeEnv:
    def __init__(self, scores):
        self.scores = scores          # answer string -> score
        self.reward = 0.0
        self.resets = []

    def reset(self, seed=None, **_):
        self.resets.append(seed)
        self.reward = 0.0
        return f"q{seed}"

    def answer(self, answer):
        self.reward = float(self.scores.get(answer, 0.0))


def test_run_episodes_scores_and_counts():
    env = _FakeEnv({"7": 1.0})
    gen = iter([("7", 10), ("9", 20)])
    rs = _run_episodes(env, n=2, seed_base=100, gen_fn=lambda q: next(gen))
    assert env.resets == [100, 101]
    assert [r.correct for r in rs] == [True, False]
    assert [r.n_tokens for r in rs] == [10, 20]
    assert all(r.n_steps == 1 for r in rs)


def test_run_episodes_handles_none_answer():
    env = _FakeEnv({})
    rs = _run_episodes(env, n=1, seed_base=0, gen_fn=lambda q: (None, 5))
    assert rs[0].correct is False and rs[0].n_tokens == 5


# --- _metrics_to_dict: serialize EvalMetrics for the report ---

def test_metrics_to_dict_shape():
    m = compute_metrics([SampleResult(True, 10, n_steps=1, reward=1.0),
                         SampleResult(False, 20, n_steps=1, reward=0.05)])
    d = _metrics_to_dict(m)
    assert d["accuracy"] == 0.5 and d["n_samples"] == 2 and d["n_correct"] == 1
    assert d["samples"][0] == {"correct": True, "n_tokens": 10, "n_steps": 1,
                               "reward": 1.0, "terminated": None,
                               "stop_reason": None, "tool_calls": None}
    assert "mean_token_count" in d and "mean_steps" in d
