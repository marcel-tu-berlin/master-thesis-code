import torch
import torch.nn.functional as F

from training.rewards.utils import extract_content


class TokenEntropyReward:
    """
    Mean per-token Shannon entropy H_t = -sum p log p from model logits.

    Batched forward pass over all completions in one call.
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
        texts = [extract_content(c) for c in completions]
        if not texts:
            return []

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        if inputs["input_ids"].shape[1] < 2:
            return [0.0] * len(completions)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (B, T, V)

        probs = F.softmax(logits, dim=-1)  # (B, T, V)
        H = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (B, T)

        pad_mask = inputs["attention_mask"].bool()  # (B, T)

        scores = []
        for i in range(len(completions)):
            h_i = H[i][pad_mask[i]]  # only real tokens

            if self.fork_mask_top_pct > 0.0 and h_i.numel() > 0:
                threshold = torch.quantile(h_i.float(), 1.0 - self.fork_mask_top_pct)
                h_i = h_i[h_i >= threshold]

            mean_entropy = h_i.mean().item() if h_i.numel() > 0 else 0.0
            scores.append(self.reward_scale * mean_entropy)

        return scores
