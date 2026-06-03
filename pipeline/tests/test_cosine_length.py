"""Tests for the correctness-coupled cosine length reward (Wu/Yeo-2025).

The point of this reward is to fix the `e1-token-length` collapse: a *linear*
penalty z-scores to "shorter always ranks higher" (monotone) and drives output
to the format-envelope floor. The cosine reward is gated by correctness, so a
wrong-and-short completion is the *most* penalized cell and cannot win the
within-group ranking.
"""
from training.rewards.cosine_length import CosineLengthReward
from training.rewards.compose import AdvantageWeightedComposer


class StubTok:
    """Completion marker is "<n_tokens>|<C|W>"; encode returns n_tokens ids."""

    def encode(self, text, add_special_tokens=False):
        return [0] * int(text.split("|")[0])


class StubDomain:
    def is_correct(self, completion, ground_truth):
        return completion.split("|")[1] == "C"


def _mk(n, correct):
    return f"{n}|{'C' if correct else 'W'}"


def _reward(max_len=200):
    return CosineLengthReward(
        StubTok(), StubDomain(), max_len=max_len,
        r_correct_short=1.0, r_correct_long=0.5,
        r_wrong_short=-1.0, r_wrong_long=-0.5,
    )


def test_correct_prefers_shorter_wrong_prefers_longer():
    r = _reward()
    correct_short, correct_long, wrong_short, wrong_long = r(
        ["p"] * 4,
        [_mk(10, True), _mk(190, True), _mk(10, False), _mk(190, False)],
        answer=["x"] * 4,
    )
    assert correct_short > correct_long      # correct: shorter is better
    assert wrong_long > wrong_short          # wrong: longer is less penalized
    assert correct_short > wrong_short       # correctness dominates length
    assert correct_long > wrong_long


def test_zscore_no_collapse_wrong_short_is_worst():
    r = _reward(max_len=200)
    # One prompt-group of 8 rollouts, including the degenerate wrong-17-token
    # cell that the linear penalty would have rewarded as the shortest.
    comps = [
        _mk(17, False), _mk(17, False), _mk(17, True), _mk(150, True),
        _mk(150, False), _mk(120, True), _mk(30, False), _mk(200, False),
    ]
    composed = AdvantageWeightedComposer([(r, 1.0)])(["P"] * 8, comps, answer=["x"] * 8)

    top = max(range(8), key=lambda i: composed[i])
    assert comps[top].endswith("C"), f"top-ranked rollout should be correct, got {comps[top]}"

    bot = min(range(8), key=lambda i: composed[i])
    n_bot, kind_bot = comps[bot].split("|")
    assert kind_bot == "W" and int(n_bot) <= 30, f"worst cell should be wrong-and-short, got {comps[bot]}"


def test_requires_answer_column():
    r = _reward()
    raised = False
    try:
        r(["p"], [_mk(10, True)], answer=None)
    except ValueError:
        raised = True
    assert raised, "CosineLengthReward must require the 'answer' column like AnswerReward"
