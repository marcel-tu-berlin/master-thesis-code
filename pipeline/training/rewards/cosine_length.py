import math

from training.rewards.utils import _require_answers, extract_content


class CosineLengthReward:
    """Correctness-coupled cosine length reward (Wu/Yeo 2025, "Demystifying
    Long CoT"; Kimi-style length scaling).

    For a completion of `n_tokens`, let progress = min(n_tokens / max_len, 1)
    and c = cos(progress * pi) — so c = +1 for an empty completion and c = -1
    at the length cap. The reward interpolates between a short-end and a
    long-end value, with the endpoints chosen by correctness:

        correct: short -> r_correct_short, long -> r_correct_long   (short > long)
        wrong:   short -> r_wrong_short,   long -> r_wrong_long     (short < long)

    Correct answers are rewarded more when shorter (concise reasoning); wrong
    answers are penalized less when longer (don't punish the model for thinking
    before failing). Wrong-and-short is therefore the most-penalized cell.

    Under the default `advantage_weighted` composer every component is z-scored
    per prompt-group, which cancels any global positive scalar. A purely monotone
    signal (e.g. a plain length penalty) z-scores to pure rank order — "shorter
    always ranks higher" — so the model drives length to the format-envelope
    floor. This reward is non-linear in length *and* gated by correctness, so it
    survives z-scoring with real structure: a wrong 17-token completion lands in
    the worst cell instead of the best one.

    Tradeoff: the length signal now carries correctness information, partially
    overlapping the `accuracy`/`numeric` components. That is acceptable under
    independent per-component z-scoring and is the intended Wu/Yeo coupling.
    """

    def __init__(
        self,
        tokenizer,
        domain,
        max_len: int,
        r_correct_short: float = 1.0,
        r_correct_long: float = 0.5,
        r_wrong_short: float = -1.0,
        r_wrong_long: float = -0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.domain = domain
        self.max_len = max(int(max_len), 1)
        self.r_correct_short = r_correct_short
        self.r_correct_long = r_correct_long
        self.r_wrong_short = r_wrong_short
        self.r_wrong_long = r_wrong_long

    def _reward(self, n_tokens: int, correct: bool) -> float:
        progress = min(n_tokens / self.max_len, 1.0)
        c = math.cos(progress * math.pi)  # +1 at length 0, -1 at max_len
        if correct:
            lo, hi = self.r_correct_long, self.r_correct_short
        else:
            lo, hi = self.r_wrong_long, self.r_wrong_short
        return lo + 0.5 * (hi - lo) * (1.0 + c)

    def __call__(self, prompts, completions, answer=None, **kwargs) -> list[float]:
        truths = _require_answers(answer, len(completions), "CosineLengthReward")
        # Prefer the exact model-generated token ids TRL/the rollout_func provide
        # over re-encoding decoded text. In the agentic loop this is what makes
        # the length signal count model tokens only. Falls back to re-encoding.
        provided_ids = kwargs.get("completion_ids")
        scores = []
        for i, (completion, truth) in enumerate(zip(completions, truths)):
            text = extract_content(completion)
            if provided_ids is not None and i < len(provided_ids):
                n_tokens = len(provided_ids[i])
            else:
                n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            correct = self.domain.is_correct(text, truth)
            scores.append(self._reward(n_tokens, correct))
        return scores
