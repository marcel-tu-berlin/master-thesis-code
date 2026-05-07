from domains.base import Domain
from training.rewards.utils import extract_content


class FormatExactReward:
    """
    +3.0 if full tag structure is present in the correct order.
    Regex is built from the domain's tag config so reward is template-agnostic.
    """

    def __init__(self, domain: Domain, reward: float = 3.0) -> None:
        self.domain = domain
        self.reward = reward

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        scores = []
        for completion in completions:
            text = extract_content(completion)
            match = self.domain._solution_re.search(text)
            scores.append(self.reward if match is not None else 0.0)
        return scores


class FormatApproxReward:
    """
    Partial credit per tag: +per_tag if exactly one occurrence, -penalty if zero or multiple.
    Penalizes repetition (model hallucinating extra tags) and rewards partial progress.
    reasoning_start excluded — it is always prepended by add_generation_prompt=True.

    reasoning_end is counted on the full text. solution_start / solution_end are counted
    on the suffix following the first reasoning_end so prose mentioning the tags inside
    the reasoning chain doesn't inflate the score.
    """

    def __init__(
        self,
        domain: Domain,
        per_tag: float = 0.5,
        penalty: float = -1.0,
        missing_penalty: float | None = None,
    ) -> None:
        self.domain = domain
        self.per_tag = per_tag
        self.penalty = penalty
        # Missing tag is a stronger failure than duplicated tag (no
        # answer at all vs. malformed-but-attempted). Default the
        # missing penalty to 1.5x the duplicate penalty so the gradient
        # signal can distinguish the two failure modes.
        self.missing_penalty = missing_penalty if missing_penalty is not None else penalty * 1.5

    def _score_tag(self, count: int) -> float:
        if count == 1:
            return self.per_tag
        if count == 0:
            return self.missing_penalty
        return self.penalty

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        re_end = self.domain.reasoning_end
        sol_start = self.domain.solution_start
        sol_end = self.domain.solution_end
        scores = []
        for completion in completions:
            text = extract_content(completion)
            score = self._score_tag(text.count(re_end))
            suffix = text.split(re_end, 1)[1] if re_end in text else ""
            score += self._score_tag(suffix.count(sol_start))
            score += self._score_tag(suffix.count(sol_end))
            scores.append(score)
        return scores
