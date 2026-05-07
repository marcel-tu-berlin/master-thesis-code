from domains.base import Domain
from training.rewards.utils import extract_content


def _require_answers(answer, n: int, reward_name: str) -> list[str]:
    if answer is None:
        raise ValueError(
            f"{reward_name} requires the dataset to expose an 'answer' column. "
            "TRL passes dataset columns as keyword args to reward functions; "
            "ensure the loader's map() output includes 'answer'."
        )
    if len(answer) != n:
        raise ValueError(
            f"{reward_name}: len(answer)={len(answer)} does not match len(completions)={n}"
        )
    return answer


class AnswerReward:
    """Graded answer-correctness reward. Calls domain.score_answer on extracted solution."""

    def __init__(self, domain: Domain) -> None:
        self.domain = domain

    def __call__(self, prompts, completions, answer=None, **kwargs) -> list[float]:
        truths = _require_answers(answer, len(completions), "AnswerReward")
        scores = []
        for completion, truth in zip(completions, truths):
            text = extract_content(completion)
            extracted = self.domain.extract_answer(text)
            scores.append(self.domain.score_answer(extracted, truth))
        return scores


class NumericReward:
    """Strict float-equality reward — catches answers like '123,456' vs '123456'."""

    def __init__(self, domain: Domain) -> None:
        self.domain = domain

    def __call__(self, prompts, completions, answer=None, **kwargs) -> list[float]:
        truths = _require_answers(answer, len(completions), "NumericReward")
        scores = []
        for completion, truth in zip(completions, truths):
            text = extract_content(completion)
            extracted = self.domain.extract_number(text) if hasattr(self.domain, "extract_number") else None
            scores.append(self.domain.score_numbers(extracted, truth))
        return scores
