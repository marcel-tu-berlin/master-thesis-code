_REQUIRED_KEYS = {
    "experiment_id": "experiment_id (str)",
    "model.slug": "model.slug (str) — must match a key in training/registry.py",
    "training.dataset": "training.dataset (str) — HuggingFace dataset id",
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


def _set_nested(d: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def validate_config(config: dict) -> None:
    errors = []

    for key, label in _REQUIRED_KEYS.items():
        if _get_nested(config, key) is None:
            errors.append(f"Missing required field: {label}")

    for key, (lo, hi) in _NUMERIC_COERCIONS.items():
        val = _get_nested(config, key)
        if val is not None:
            try:
                val = float(val)
            except (TypeError, ValueError):
                errors.append(f"Field {key}={val!r} is not numeric")
                continue
            if not (lo <= val <= hi):
                errors.append(f"Field {key}={val} out of range [{lo}, {hi}]")
            _set_nested(config, key, val)

    slug = _get_nested(config, "model.slug")
    if slug is not None:
        from training.registry import MODEL_REGISTRY
        if slug not in MODEL_REGISTRY:
            errors.append(
                f"model.slug={slug!r} not in registry. Available: {list(MODEL_REGISTRY)}"
            )

    compose = _get_nested(config, "rewards.compose_method")
    if compose is not None and compose not in ("advantage_weighted", "naive_sum"):
        errors.append(
            f"rewards.compose_method={compose!r} must be 'advantage_weighted' or 'naive_sum'"
        )

    if errors:
        msg = "Config validation failed:\n" + "\n".join(f" - {e}" for e in errors)
        raise ValueError(msg)
