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
    "token_length":  _COMMON_REWARD_SUBKEYS | {
        # Linear-penalty knobs (used when shape=linear, e.g. under naive_sum):
        "alpha", "schedule",
        # Cosine-shape knobs (Wu/Yeo correctness-coupled length reward):
        "shape", "max_len",
        "r_correct_short", "r_correct_long", "r_wrong_short", "r_wrong_long",
    },
    "token_entropy": _COMMON_REWARD_SUBKEYS | {
        "reward_scale", "fork_mask_top_frac", "chunk_size",
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


def warn_inert_scalars(rewards_cfg: dict, compose_method: str) -> list[str]:
    """Return warnings for reward knobs that do nothing as configured.

    Two independent sources of inertness:

    * **Wrong shape.** `token_length` resolves `shape` the same way the registry
      builder does (default `cosine`). The cosine length reward uses
      correctness-coupled endpoints and never reads `alpha`/`schedule`, so those
      linear-only knobs are dead under *any* composer — a set value signals a
      mistake (you probably meant `shape: linear`). This check is
      composer-independent.
    * **Z-scoring.** Under `advantage_weighted`, per-group z-scoring is invariant
      to any global positive scalar, so the *linear* `token_length.alpha`/cosine
      `schedule`, `token_entropy.reward_scale`, `effort_proxy.alpha`, and a
      non-token_count effort `metric` all cancel. These are live under
      `naive_sum`, so they stay quiet there.

    Default-valued scalars are not flagged (boilerplate); we warn only when a
    value signals intent to tune (non-default alpha/reward_scale, an explicit
    cosine schedule, or a flops/gpu_time metric). Disabled rewards are skipped.
    """
    rc = rewards_cfg or {}
    warnings: list[str] = []
    lever = "Use `weight`, the signal shape, or compose_method: naive_sum instead."

    # token_length — resolve shape exactly as the builder does (default cosine)
    # so the warning matches what actually runs.
    tl = rc.get("token_length") or {}
    if tl.get("enabled"):
        shape = tl.get("shape", "cosine")
        alpha_set = "alpha" in tl and tl["alpha"] != 0.001
        cosine_schedule = tl.get("schedule") == "cosine"
        if shape == "cosine":
            # Composer-independent: the cosine reward never reads these knobs.
            if alpha_set:
                warnings.append(
                    f"rewards.token_length.alpha={tl['alpha']} is ignored under shape: cosine "
                    "(the cosine length reward uses correctness-coupled endpoints, not alpha). "
                    "Set shape: linear if you meant the linear penalty."
                )
            if cosine_schedule:
                warnings.append(
                    "rewards.token_length.schedule: cosine is ignored under shape: cosine "
                    "(schedule only modulates the linear penalty). Remove it, or set shape: linear."
                )
        elif shape == "linear" and compose_method == "advantage_weighted":
            # Linear penalty: per-group z-scoring cancels the global scalars.
            if alpha_set:
                warnings.append(
                    f"rewards.token_length.alpha={tl['alpha']} is inert under advantage_weighted "
                    f"(per-group z-scoring cancels global scalars). {lever}"
                )
            if cosine_schedule:
                warnings.append(
                    "rewards.token_length.schedule: cosine is inert under advantage_weighted "
                    "(the schedule is a per-step global scalar that z-scoring cancels). "
                    "Anneal `weight`, or use shape: cosine / naive_sum."
                )

    # The remaining knobs are inert only under advantage_weighted (z-scoring);
    # under naive_sum they are live.
    if compose_method == "advantage_weighted":
        te = rc.get("token_entropy") or {}
        if te.get("enabled") and "reward_scale" in te and te["reward_scale"] != 0.1:
            warnings.append(
                f"rewards.token_entropy.reward_scale={te['reward_scale']} is inert under "
                f"advantage_weighted (z-scoring cancels global scalars). {lever}"
            )

        ep = rc.get("effort_proxy") or {}
        if ep.get("enabled"):
            if "alpha" in ep and ep["alpha"] != 0.001:
                warnings.append(
                    f"rewards.effort_proxy.alpha={ep['alpha']} is inert under advantage_weighted "
                    f"(z-scoring cancels global scalars). {lever}"
                )
            if ep.get("metric") in ("flops", "gpu_time"):
                warnings.append(
                    f"rewards.effort_proxy.metric={ep['metric']!r} has no effect under advantage_weighted: "
                    "flops/gpu_time differ from token_count only by a global scalar that z-scoring cancels, "
                    "so all effort metrics reduce to z-scored token count (identical to token_length)."
                )

    return warnings


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
