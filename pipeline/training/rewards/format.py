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
    Partial credit per tag: +0.5 if exactly one occurrence, -1.0 if zero or multiple.
    Penalizes repetition (model hallucinating extra tags) and rewards partial progress.
    reasoning_start excluded — it is always prepended by add_generation_prompt=True.
    """

    def __init__(self, domain: Domain, per_tag: float = 0.5, penalty: float = -1.0) -> None:
        self.domain = domain
        self.per_tag = per_tag
        self.penalty = penalty
        self._tags = [domain.reasoning_end, domain.solution_start, domain.solution_end]

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        scores = []
        for completion in completions:
            text = extract_content(completion)
            after_reasoning = text.split(self.domain.reasoning_end, 1)[-1] if self.domain.reasoning_end in text else text
            score = sum(
                self.per_tag if after_reasoning.count(tag) == 1 else self.penalty
                for tag in self._tags
            )
            scores.append(score)
        return scores
