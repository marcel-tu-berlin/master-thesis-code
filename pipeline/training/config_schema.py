_REQUIRED_KEYS = {
    "experiment_id": "experiment_id (str)",
    "model.slug": "model.slug (str) — must match a key in training/registry.py",
    "training.dataset": "training.dataset (str) — HuggingFace dataset id",
}

_KNOWN_TOP_LEVEL_KEYS = {
    "experiment_id",
    "description",
    "seed",
    "baseline_id",
    "model",
    "training",
    "rewards",
    "eval",
    # Internal: smoke override marker propagated from train.py to eval. Allowed
    # but stripped before the frozen config is written.
    "_smoke",
}

_KNOWN_REWARD_KEYS = {
    "compose_method",
    "format_exact",
    "format_approx",
    "accuracy",
    "numeric",
    "token_length",
    "token_entropy",
    "effort_proxy",
}

# Whitelist of allowed sub-keys per reward. Catches typos in YAML (e.g.
# `fork_mask_top_pct` after the rename to `fork_mask_top_frac`) that would
# otherwise pass through silently and use the default.
_COMMON_REWARD_SUBKEYS = {"enabled", "weight"}
_KNOWN_REWARD_SUBKEYS: dict[str, set[str]] = {
    "format_exact":  _COMMON_REWARD_SUBKEYS | {"reward"},
    "format_approx": _COMMON_REWARD_SUBKEYS | {"per_tag", "penalty", "missing_penalty"},
    "accuracy":      _COMMON_REWARD_SUBKEYS,
    "numeric":       _COMMON_REWARD_SUBKEYS,
    "token_length":  _COMMON_REWARD_SUBKEYS | {"alpha", "schedule"},
    "token_entropy": _COMMON_REWARD_SUBKEYS | {
        "reward_scale", "fork_mask_top_frac",
        # Deprecated alias: still accepted, builder warns.
        "fork_mask_top_pct",
    },
    "effort_proxy":  _COMMON_REWARD_SUBKEYS | {"metric", "alpha"},
}

_NUMERIC_COERCIONS = {
    "model.lora_r": (1, 256),
    "model.max_seq_length": (64, 131072),
    "training.max_steps": (1, 100_000),
    "training.learning_rate": (1e-8, 1e-2),
    "training.kl_beta": (0.0, 1.0),
    "training.temperature": (0.0, 10.0),
    "training.weight_decay": (0.0, 1.0),
    "training.warmup_ratio": (0.0, 1.0),
    "training.batch_size": (1, 1024),
    "training.gradient_accumulation_steps": (1, 1024),
    "training.n_rollouts": (1, 256),
    "training.save_steps": (1, 100_000),
    "training.max_prompt_length": (1, 131072),
    "training.dataset_size_limit": (1, 1_000_000),
}


def _get_nested(d: dict, key: str):
    parts = key.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def validate_config(config: dict) -> None:
    """Validate config in-place-free: never mutates the input dict.

    A previous version coerced int fields to floats via _set_nested, which
    leaked floats (e.g. max_steps: 500.0) into the frozen runs/<exp>/config.yaml.
    Range checks now operate on a local float copy only.
    """
    errors = []

    for key, label in _REQUIRED_KEYS.items():
        if _get_nested(config, key) is None:
            errors.append(f"Missing required field: {label}")

    for key, (lo, hi) in _NUMERIC_COERCIONS.items():
        val = _get_nested(config, key)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            errors.append(f"Field {key}={val!r} is not numeric")
            continue
        if not (lo <= fval <= hi):
            errors.append(f"Field {key}={val} out of range [{lo}, {hi}]")

    slug = _get_nested(config, "model.slug")
    if slug is not None:
        from training.registry import MODEL_REGISTRY
        if slug not in MODEL_REGISTRY:
            errors.append(
                f"model.slug={slug!r} not in registry. Available: {list(MODEL_REGISTRY)}"
            )
        else:
            # Cross-check lora_r against the registry's max for this model;
            # _NUMERIC_COERCIONS upper bound (256) is permissive across all models.
            lora_r = _get_nested(config, "model.lora_r")
            max_rank = MODEL_REGISTRY[slug].get("max_lora_rank")
            if lora_r is not None and max_rank is not None:
                try:
                    if int(lora_r) > int(max_rank):
                        errors.append(
                            f"model.lora_r={lora_r} exceeds registry max_lora_rank={max_rank} for slug={slug!r}"
                        )
                except (TypeError, ValueError):
                    pass

    compose = _get_nested(config, "rewards.compose_method")
    if compose is not None and compose not in ("advantage_weighted", "naive_sum"):
        errors.append(
            f"rewards.compose_method={compose!r} must be 'advantage_weighted' or 'naive_sum'"
        )

    rewards = config.get("rewards") or {}
    unknown_rewards = set(rewards.keys()) - _KNOWN_REWARD_KEYS
    if unknown_rewards:
        errors.append(
            f"Unknown rewards keys: {sorted(unknown_rewards)}. Known: {sorted(_KNOWN_REWARD_KEYS)}"
        )

    for reward_name, allowed in _KNOWN_REWARD_SUBKEYS.items():
        sub = rewards.get(reward_name)
        if not isinstance(sub, dict):
            continue
        unknown_sub = set(sub.keys()) - allowed
        if unknown_sub:
            errors.append(
                f"Unknown sub-keys under rewards.{reward_name}: {sorted(unknown_sub)}. "
                f"Allowed: {sorted(allowed)}"
            )

    unknown_top = set(config.keys()) - _KNOWN_TOP_LEVEL_KEYS
    if unknown_top:
        errors.append(
            f"Unknown top-level keys: {sorted(unknown_top)}. Known: {sorted(_KNOWN_TOP_LEVEL_KEYS)}"
        )

    if errors:
        msg = "Config validation failed:\n" + "\n".join(f" - {e}" for e in errors)
        raise ValueError(msg)
