"""
Reward registry. Maps a config key (under `rewards:`) to a default weight and
a builder. `train.build_reward_components` iterates this registry instead of
hard-coding seven branches; new signals only require a builder + entry here
plus the matching key in `config_schema._KNOWN_REWARD_KEYS`.

Each builder has the signature:
    build(domain, runner, training_cfg, reward_cfg) -> Callable
where reward_cfg is the dict under `rewards.<key>` from the YAML.
"""
from training.rewards.accuracy import AnswerReward, NumericReward
from training.rewards.effort_proxy import EffortProxyReward
from training.rewards.format import FormatApproxReward, FormatExactReward
from training.rewards.token_entropy import TokenEntropyReward
from training.rewards.token_length import TokenLengthReward


def _build_format_exact(domain, runner, training_cfg, cfg):
    return FormatExactReward(domain)


def _build_format_approx(domain, runner, training_cfg, cfg):
    return FormatApproxReward(
        domain,
        per_tag=cfg.get("per_tag", 0.5),
        penalty=cfg.get("penalty", -1.0),
        missing_penalty=cfg.get("missing_penalty"),
    )


def _build_accuracy(domain, runner, training_cfg, cfg):
    return AnswerReward(domain)


def _build_numeric(domain, runner, training_cfg, cfg):
    return NumericReward(domain)


def _build_token_length(domain, runner, training_cfg, cfg):
    return TokenLengthReward(
        runner.tokenizer,
        alpha=cfg.get("alpha", 0.001),
        mode=cfg.get("schedule", "constant"),
        total_steps=int(training_cfg.get("max_steps", 500)),
    )


def _build_token_entropy(domain, runner, training_cfg, cfg):
    # `fork_mask_top_pct` is deprecated; prefer `fork_mask_top_frac`. Keep
    # the old key working for now so existing configs don't break, but
    # emit a one-line warning so they get migrated.
    if "fork_mask_top_pct" in cfg and "fork_mask_top_frac" not in cfg:
        print(
            "Warning: rewards.token_entropy.fork_mask_top_pct is deprecated; "
            "rename to fork_mask_top_frac (value range [0, 1])."
        )
    frac = cfg.get("fork_mask_top_frac", cfg.get("fork_mask_top_pct", 0.0))
    # Honour explicit max_seq_length on the model config so the entropy
    # forward pass doesn't silently exceed the configured context.
    model_cfg = getattr(runner, "config", {}).get("model", {}) if hasattr(runner, "config") else {}
    max_seq = model_cfg.get("max_seq_length") or training_cfg.get("max_seq_length")
    return TokenEntropyReward(
        runner.model,
        runner.tokenizer,
        reward_scale=cfg.get("reward_scale", 0.1),
        fork_mask_top_frac=frac,
        max_seq_length=int(max_seq) if max_seq is not None else None,
    )


def _build_effort_proxy(domain, runner, training_cfg, cfg):
    model_cfg = None
    inner = getattr(runner.model, "config", None)
    if inner is not None and hasattr(inner, "to_dict"):
        try:
            model_cfg = inner.to_dict()
        except (AttributeError, TypeError):
            model_cfg = None
    return EffortProxyReward(
        runner.tokenizer,
        metric=cfg.get("metric", "token_count"),
        alpha=cfg.get("alpha", 0.001),
        model_config=model_cfg,
    )


# key -> (default_enabled, default_weight, builder)
REWARD_REGISTRY: dict[str, tuple[bool, float, callable]] = {
    "format_exact":  (True,  1.0, _build_format_exact),
    "format_approx": (True,  0.5, _build_format_approx),
    "accuracy":      (True,  1.0, _build_accuracy),
    "numeric":       (True,  1.0, _build_numeric),
    "token_length":  (False, 1.0, _build_token_length),
    "token_entropy": (False, 1.0, _build_token_entropy),
    "effort_proxy":  (False, 1.0, _build_effort_proxy),
}
