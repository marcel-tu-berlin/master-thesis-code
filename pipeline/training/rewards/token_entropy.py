from __future__ import annotations

import torch
import torch.nn.functional as F


class TokenEntropyReward:
    """
    Mean per-token Shannon entropy H_t = -∑ p log p from model logits.

    Requires a forward pass on the completion (one extra inference step per batch).
    Rewards completions where the model faced high-uncertainty "fork" tokens,
    incentivising genuine deliberation over low-entropy pattern matching.

    fork_mask_top_pct: if >0, average only over tokens in the top-X% by entropy
    (wang-2025-high-entropy-tokens). Focuses reward on actual decision points.
    """

    def __init__(
        self,
        model,
        tokenizer,
        reward_scale: float = 0.1,
        fork_mask_top_pct: float = 0.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_scale = reward_scale
        self.fork_mask_top_pct = fork_mask_top_pct

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        scores = []
        for completion in completions:
            text = completion[0]["content"]
            entropy = self._compute_mean_entropy(text)
            scores.append(self.reward_scale * entropy)
        return scores

    @torch.no_grad()
    def _compute_mean_entropy(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if inputs["input_ids"].shape[1] < 2:
            return 0.0

        logits = self.model(**inputs).logits  # (1, T, V)
        probs = F.softmax(logits[0], dim=-1)  # (T, V)
        # H_t = -∑ p * log(p+ε)
        H = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (T,)

        if self.fork_mask_top_pct > 0.0:
            threshold = torch.quantile(H, 1.0 - self.fork_mask_top_pct)
            H = H[H >= threshold]

        return H.mean().item() if H.numel() > 0 else 0.0
