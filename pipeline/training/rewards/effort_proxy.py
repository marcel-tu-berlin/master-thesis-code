from training.rewards.utils import extract_content


class EffortProxyReward:
    """
    Penalizes compute effort per rollout. Three metrics:

    token_count — count completion tokens; most reliable, directly proportional
                  to decoding cost.
    flops — heuristic: 2 * T * L * D² (T=tokens, L=layers, D=hidden_dim).
    Returns effort in GFLOPs (divided by 1e9) so the magnitude is comparable
    to token_count but reflects architecture differences.
    gpu_time    — wall-clock timing is not available inside the reward function
                  (generation already finished); falls back to token_count.

    The reward is negative (penalises effort). Higher weight → stronger pressure
    to produce shorter chains of thought.
    """

    def __init__(
        self,
        tokenizer,
        metric: str = "token_count",
        alpha: float = 0.001,
        model_config: dict | None = None,
    ) -> None:
        if metric not in {"token_count", "flops", "gpu_time"}:
            raise ValueError(f"Unknown metric: {metric!r}")
        self.tokenizer = tokenizer
        self.metric = metric
        self.alpha = alpha
        self._flop_scale = self._build_flop_scale(model_config)

    def _build_flop_scale(self, cfg: dict | None) -> float:
        if cfg is None:
            return 1.0
        hidden = cfg.get("hidden_size", 4096)
        layers = cfg.get("num_hidden_layers", 32)
        # Approximate FLOPs per token: 2 * D² * L (dominant attention + FFN term)
        return 2.0 * (hidden ** 2) * layers

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        scores = []
        for completion in completions:
            text = extract_content(completion)
            n_tokens = len(self.tokenizer.encode(text))

            if self.metric == "flops":
                effort = n_tokens * self._flop_scale / 1e9
            else:
                effort = float(n_tokens)

            scores.append(-self.alpha * effort)
        return scores
