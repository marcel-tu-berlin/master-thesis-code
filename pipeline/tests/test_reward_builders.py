"""The token_length builder always returns the cosine length reward."""
from training.rewards import _build_token_length
from training.rewards.cosine_length import CosineLengthReward


class StubTok:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


class StubDomain:
    def is_correct(self, completion, ground_truth):
        return False


class StubRunner:
    tokenizer = StubTok()
    config = {"model": {"max_seq_length": 2048}}


def test_builds_cosine_reward_with_defaults():
    fn = _build_token_length(StubDomain(), StubRunner(), {"max_steps": 500}, {})
    assert isinstance(fn, CosineLengthReward)
    assert fn.max_len == 256  # builder default now matches the configs


def test_max_len_is_configurable():
    fn = _build_token_length(StubDomain(), StubRunner(), {"max_steps": 500}, {"max_len": 512})
    assert fn.max_len == 512
