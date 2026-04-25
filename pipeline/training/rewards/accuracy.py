from __future__ import annotations

from domains.base import Domain
from training.rewards.utils import extract_content


class AnswerReward:
    """Graded answer-correctness reward. Calls domain.score_answer on extracted solution."""

    def __init__(self, domain: Domain) -> None:
        self.domain = domain

    def __call__(self, prompts, completions, answer: list[str], **kwargs) -> list[float]:
        scores = []
        for completion, truth in zip(completions, answer):
            text = extract_content(completion)
            extracted = self.domain.extract_answer(text)
            scores.append(self.domain.score_answer(extracted, truth))
        return scores


class NumericReward:
    """Strict float-equality reward — catches answers like '123,456' vs '123456'."""

    def __init__(self, domain: Domain) -> None:
        self.domain = domain

    def __call__(self, prompts, completions, answer: list[str], **kwargs) -> list[float]:
        scores = []
        for completion, truth in zip(completions, answer):
            text = extract_content(completion)
            extracted = self.domain.extract_number(text) if hasattr(self.domain, "extract_number") else None
            scores.append(self.domain.score_numbers(extracted, truth))
        return scores
