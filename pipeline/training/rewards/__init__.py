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
    return FormatApproxReward(domain)


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
    return TokenEntropyReward(
        runner.model,
        runner.tokenizer,
        reward_scale=cfg.get("reward_scale", 0.1),
        fork_mask_top_pct=cfg.get("fork_mask_top_pct", 0.0),
    )


def _build_effort_proxy(domain, runner, training_cfg, cfg):
    model_cfg = None
    inner = getattr(runner.model, "config", None)
    if inner is not None and hasattr(inner, "to_dict"):
        try:
            model_cfg = inner.to_dict()
        except Exception:
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
