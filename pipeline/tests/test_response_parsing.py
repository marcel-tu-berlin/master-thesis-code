"""End-to-end wire-format parsing, against real tokenizers.

Eval scores a completion with `trl.chat_template_utils.parse_response`, the same
function TRL applies during training. These tests pin the behaviour the fix
depends on - especially that a TRUNCATED tool call is not a tool call, which is
what keeps a cap-hit episode labelled `hit_generation_cap` instead of
`env_done`. Collapsing those two is what made the e9-e21 sweep uninterpretable.

Needs transformers + trl and the tokenizers on disk, so it is skipped on the
CPU test venv and runs on the GPU box.
"""
import pytest

pytest.importorskip("transformers")
pytest.importorskip("trl")

from transformers import AutoTokenizer                       # noqa: E402
from trl.chat_template_utils import add_response_schema, parse_response  # noqa: E402

from eval.agentic_eval import _answer_from, _first_tool_call  # noqa: E402

QWEN = "Qwen/Qwen3-1.7B"
LLAMA = "unsloth/Llama-3.2-1B-Instruct"


def _tok(name):
    tok = AutoTokenizer.from_pretrained(name)
    add_response_schema(tok)
    return tok


@pytest.fixture(scope="module")
def qwen():
    return _tok(QWEN)


@pytest.fixture(scope="module")
def llama():
    return _tok(LLAMA)


def _parse(tok, text):
    return parse_response(tok, tok.encode(text, add_special_tokens=False))


# --- Qwen3 / Hermes ---

def test_qwen_complete_call_is_parsed(qwen):
    msg = _parse(qwen, '<think>17+25</think>\n<tool_call>\n'
                       '{"name": "answer", "arguments": {"answer": "42"}}\n</tool_call>')
    assert _answer_from(msg) == "42"
    assert msg.get("reasoning_content") == "17+25"


def test_qwen_truncated_call_is_not_a_call(qwen):
    # Cut after the closing brace, before </tool_call>: the episode filled its
    # generation budget. It must NOT read as a finished answer.
    msg = _parse(qwen, '<think>17+25</think>\n<tool_call>\n'
                       '{"name": "answer", "arguments": {"answer": "42"}}')
    assert _answer_from(msg) is None
    assert _first_tool_call(msg) is None


def test_qwen_json_quoted_in_think_is_not_a_call(qwen):
    msg = _parse(qwen, '<think>maybe {"name": "answer", "arguments": '
                       '{"answer": "0"}} hmm</think>\nI give up.')
    assert _answer_from(msg) is None
    assert _first_tool_call(msg) is None


def test_qwen_prose_only_is_not_a_call(qwen):
    assert _first_tool_call(_parse(qwen, "I think the answer is 42.")) is None


# --- Llama 3.x: untagged, `parameters` rather than `arguments` ---

def test_llama_untagged_call_is_parsed(llama):
    msg = _parse(llama, '{"name": "answer", "parameters": {"answer": "42"}}')
    assert _answer_from(msg) == "42"


def test_llama_multi_tool_call_is_parsed(llama):
    msg = _parse(llama, '{"name": "click", "parameters": {"bid": "17"}}')
    assert _first_tool_call(msg) == ("click", {"bid": "17"})


def test_llama_prose_only_is_not_a_call(llama):
    assert _first_tool_call(_parse(llama, "I think the answer is 42.")) is None


# --- the context the multi-turn loop rebuilds must keep the reasoning ---

def test_qwen_template_preserves_reasoning_on_a_prior_turn(qwen):
    # _run_multiturn_episodes appends the parsed message and re-renders. If the
    # template dropped reasoning_content here, turn N+1 would run on a context
    # training never produced and the fix would be cosmetic.
    rendered = qwen.apply_chat_template([
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "", "reasoning_content": "LONG REASONING",
         "tool_calls": [{"type": "function",
                         "function": {"name": "answer", "arguments": {"answer": "42"}}}]},
        {"role": "tool", "content": "Recorded 42"},
    ], add_generation_prompt=True, tokenize=False)
    assert "LONG REASONING" in rendered
