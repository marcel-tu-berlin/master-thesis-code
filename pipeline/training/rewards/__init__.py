"""
Reward registry. Maps a config key (under `rewards:`) to a default weight and
a builder. `train.build_reward_components` iterates this registry instead of
hard-coding branches; new signals only require a builder + entry here plus the
matching key in `config_schema._KNOWN_REWARD_KEYS`.

Each builder has the signature:
    build(domain, runner, training_cfg, reward_cfg) -> Callable
where reward_cfg is the dict under `rewards.<key>` from the YAML.

Agentic-only: `env_reward` is the task-success signal; `token_length` and
`token_entropy` are the token-efficiency signals. All three default off and are
enabled explicitly per config.
"""
from training.rewards.cosine_length import CosineLengthReward
from training.rewards.token_entropy import TokenEntropyReward
from training.rewards.env_reward import EnvReward


def _build_token_length(domain, runner, training_cfg, cfg):
    # Cosine length reward (Wu/Yeo 2025): correct -> prefer shorter, wrong ->
    # prefer longer. Non-linear and correctness-gated (correctness comes from the
    # env), so it survives the advantage_weighted per-group z-scoring.
    return CosineLengthReward(
        runner.tokenizer,
        max_len=int(cfg.get("max_len", 256)),
        r_correct_short=cfg.get("r_correct_short", 1.0),
        r_correct_long=cfg.get("r_correct_long", 0.5),
        r_wrong_short=cfg.get("r_wrong_short", -1.0),
        r_wrong_long=cfg.get("r_wrong_long", -0.5),
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
    # chunk_size default: vLLM co-resident -> 1 (tight VRAM, see token_entropy
    # OOM); otherwise 4 (ample headroom on 24 GB). Override per-config via
    # `rewards.token_entropy.chunk_size`.
    default_chunk = 1 if model_cfg.get("use_vllm") else 4
    chunk_size = int(cfg.get("chunk_size", default_chunk))
    return TokenEntropyReward(
        runner.model,
        runner.tokenizer,
        reward_scale=cfg.get("reward_scale", 0.1),
        fork_mask_top_frac=frac,
        max_seq_length=int(max_seq) if max_seq is not None else None,
        chunk_size=chunk_size,
    )


def _build_env_reward(domain, runner, training_cfg, cfg):
    # Task-success reward from the OpenEnv environment. TRL's environment_factory
    # path passes the live env instances as kwargs['environments']; EnvReward
    # reads env.reward off each.
    return EnvReward()


# key -> (default_enabled, default_weight, builder). All default off; agentic
# configs enable env_reward + the efficiency signals explicitly.
REWARD_REGISTRY: dict[str, tuple[bool, float, callable]] = {
    "token_length":  (False, 1.0, _build_token_length),
    "token_entropy": (False, 1.0, _build_token_entropy),
    "env_reward":    (False, 1.0, _build_env_reward),
}
