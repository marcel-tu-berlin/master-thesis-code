import math

from training.rewards.utils import extract_content


class TokenLengthReward:
    """
    Negative length penalty — discourages token-budget waste.

    Two modes:
    - constant: reward = -alpha * num_tokens
    - cosine:   alpha annealed from 0 → alpha_max over training, so the model
                learns to reason first and compresses later (DIET schedule).
    """

    def __init__(
        self,
        tokenizer,
        alpha: float = 0.001,
        mode: str = "constant",
        total_steps: int = 500,
    ) -> None:
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.mode = mode
        self.total_steps = total_steps
        self._step = 0

    def _current_alpha(self) -> float:
        if self.mode == "cosine":
            progress = min(self._step / max(self.total_steps, 1), 1.0)
            return self.alpha * (1 - math.cos(math.pi * progress)) / 2
        return self.alpha

    def step(self) -> None:
        self._step += 1

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        alpha = self._current_alpha()
        scores = []
        for completion in completions:
            text = extract_content(completion)
            n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            scores.append(-alpha * n_tokens)
        return scores
