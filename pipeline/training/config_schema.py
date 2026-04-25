from __future__ import annotations

_REQUIRED_KEYS = {
    "experiment_id": "experiment_id (str)",
    "model.slug": "model.slug (str) — must match a key in training/registry.py",
    "training.dataset": "training.dataset (str) — HuggingFace dataset id",
}

_NUMERIC_RANGES = {
    "model.lora_r": (1, 256),
    "training.max_steps": (1, 100_000),
    "training.learning_rate": (1e-8, 1e-2),
    "training.kl_beta": (0.0, 1.0),
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
    errors = []

    for key, label in _REQUIRED_KEYS.items():
        if _get_nested(config, key) is None:
            errors.append(f"Missing required field: {label}")

    for key, (lo, hi) in _NUMERIC_RANGES.items():
        val = _get_nested(config, key)
        if val is not None and not (lo <= val <= hi):
            errors.append(f"Field {key}={val} out of range [{lo}, {hi}]")

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
        msg = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)
