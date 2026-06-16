from eval.agentic_eval import _parse_answer, _run_episodes, _metrics_to_dict, _completion_budget
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


# --- _parse_answer: extract the answer from a model tool call ---

def test_parse_answer_simple():
    t = '<tool_call>\n{"name": "answer", "arguments": {"answer": "42"}}\n</tool_call>'
    assert _parse_answer(t) == "42"


def test_parse_answer_with_think_prefix():
    t = '<think>17+25 is 42</think>\n<tool_call>{"name": "answer", "arguments": {"answer": "42"}}</tool_call>'
    assert _parse_answer(t) == "42"


def test_parse_answer_coerces_non_string():
    t = '<tool_call>{"name": "answer", "arguments": {"answer": 42}}</tool_call>'
    assert _parse_answer(t) == "42"


def test_parse_answer_takes_first_answer_call():
    t = ('<tool_call>{"name":"answer","arguments":{"answer":"7"}}</tool_call>'
         '<tool_call>{"name":"answer","arguments":{"answer":"8"}}</tool_call>')
    assert _parse_answer(t) == "7"


def test_parse_answer_none_without_tool_call():
    assert _parse_answer("the answer is 42") is None


def test_parse_answer_none_on_malformed_json():
    assert _parse_answer("<tool_call>{not valid json}</tool_call>") is None


def test_parse_answer_ignores_other_tools():
    assert _parse_answer('<tool_call>{"name":"other","arguments":{"x":1}}</tool_call>') is None


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
    m = compute_metrics([SampleResult(True, 10, n_steps=1), SampleResult(False, 20, n_steps=1)])
    d = _metrics_to_dict(m)
    assert d["accuracy"] == 0.5 and d["n_samples"] == 2 and d["n_correct"] == 1
    assert d["samples"][0] == {"correct": True, "n_tokens": 10, "n_steps": 1}
    assert "mean_token_count" in d and "mean_steps" in d
