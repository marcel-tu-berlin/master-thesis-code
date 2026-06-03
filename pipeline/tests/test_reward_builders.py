"""The token_length builder picks linear vs cosine on the `shape` subkey."""
from training.rewards import _build_token_length
from training.rewards.cosine_length import CosineLengthReward
from training.rewards.token_length import TokenLengthReward


class StubTok:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


class StubDomain:
    def is_correct(self, completion, ground_truth):
        return False


class StubRunner:
    tokenizer = StubTok()
    config = {"model": {"max_seq_length": 2048}}


def test_default_shape_is_linear():
    fn = _build_token_length(StubDomain(), StubRunner(), {"max_steps": 500}, {})
    assert isinstance(fn, TokenLengthReward)


def test_cosine_shape_builds_cosine_reward():
    fn = _build_token_length(
        StubDomain(), StubRunner(), {"max_steps": 500},
        {"shape": "cosine", "max_len": 256},
    )
    assert isinstance(fn, CosineLengthReward)
    assert fn.max_len == 256


def test_unknown_shape_raises():
    raised = False
    try:
        _build_token_length(StubDomain(), StubRunner(), {"max_steps": 500}, {"shape": "bogus"})
    except ValueError:
        raised = True
    assert raised
