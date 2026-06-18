import pytest

from training.rewards.cosine_length import CosineLengthReward


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return text.split()  # one pseudo-token per word


class _Env:
    def __init__(self, reward):
        self.reward = reward


def test_prefers_completion_ids_over_text():
    r = CosineLengthReward(_Tok(), max_len=10)
    # completion_ids says 2 tokens; the decoded text would re-encode to 5.
    out_ids = r(["p"], ["a b c d e"], environments=[_Env(1.0)], completion_ids=[[1, 2]])
    out_txt = r(["p"], ["a b c d e"], environments=[_Env(1.0)])
    assert out_ids != out_txt  # id path (2 tokens) differs from text path (5)


def test_falls_back_to_text_without_ids():
    r = CosineLengthReward(_Tok(), max_len=10)
    out = r(["p"], ["a b c"], environments=[_Env(1.0)])
    assert len(out) == 1


def test_agentic_correctness_from_environments():
    # Correctness comes from env.reward > 0. Same length, so the correct env must
    # outrank the wrong env purely on the correctness gate.
    r = CosineLengthReward(_Tok(), max_len=10)
    out = r(
        ["p", "p"], ["a b", "a b"],
        environments=[_Env(1.0), _Env(0.0)], completion_ids=[[1, 2], [1, 2]],
    )
    assert out[0] > out[1]


def test_requires_environments():
    # No environments kwarg -> must raise (there is no answer-column fallback).
    r = CosineLengthReward(_Tok(), max_len=10)
    with pytest.raises(ValueError):
        r(["p"], ["a b"])


def test_multiturn_counts_assistant_only_not_completion_ids():
    # A multi-turn completion (assistant/tool/assistant) with a long completion_ids
    # stream that includes tool tokens. The reward must use the assistant-only
    # count (4), not len(completion_ids) (12).
    r = CosineLengthReward(_Tok(), max_len=100)
    completion = [
        {"role": "assistant", "content": "guess crane"},          # 2
        {"role": "tool", "content": "C absent R absent ..."},     # tool -> skip
        {"role": "assistant", "content": "guess slate"},          # 2
    ]
    out = r(["p"], [completion], environments=[_Env(1.0)], completion_ids=[[0] * 12])
    assert out[0] == r._reward(4, True)


def test_singleturn_one_assistant_message_uses_completion_ids():
    # Single assistant message, no tool message -> single-turn -> use the exact
    # completion_ids count (3), NOT the re-encoded text (5). Reasoning_gym parity.
    r = CosineLengthReward(_Tok(), max_len=100)
    completion = [{"role": "assistant", "content": "a b c d e"}]
    out = r(["p"], [completion], environments=[_Env(1.0)], completion_ids=[[1, 2, 3]])
    assert out[0] == r._reward(3, True)
