_REQUIRED_KEYS = {
    "experiment_id": "experiment_id (str)",
    "model.slug": "model.slug (str) — must match a key in training/registry.py",
    # training.env (OpenEnv environment id) is required; see validate_config.
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
    "token_length",
    "env_reward",
    "non_termination",
}

# Known keys under `training`. This was the last block with no whitelist, and it
# is the one where a typo is most expensive: every key below has a default in
# grpo_runner._grpo_config, so `max_prompt_lenght: 6144` validated fine, was
# ignored, and the run trained at max_seq//2 while its own frozen config claimed
# otherwise. The same silent fallback applied to n_rollouts (the GRPO group size,
# hence the advantage denominator), micro_batch_size, kl_beta and save_steps.
#
# Deliberately absent: `gradient_accumulation_steps` and `dataset_size_limit`.
# Both used to be range-checked here, and neither is read anywhere - grad accum
# is derived from batch_size / micro_batch_size, and the dataset size comes from
# env_config.size. Setting either now fails loudly instead of doing nothing.
_KNOWN_TRAINING_KEYS = {
    "mode", "env", "env_config", "env_server",
    "max_prompt_length", "max_steps", "save_steps",
    "n_rollouts", "batch_size", "micro_batch_size",
    "learning_rate", "kl_beta", "temperature", "weight_decay", "warmup_ratio",
}

_KNOWN_ENV_SERVER_KEYS = {"repo_path", "port"}

# Known sub-keys under training.env_config (union across env types - catches
# typos like `datsaet` that would otherwise pass through and use the default).
_KNOWN_ENV_CONFIG_KEYS = {
    # reasoning_gym
    "dataset", "dataset_name", "dataset_config", "size",
    # textarena
    "env_id", "num_players",
    # finqa
    "data_path",
    # browsergym: `tasks` is the MiniWoB family list the seed cycles through
    # (`tasks[seed % len]`), so it also sets the training task mix. `miniwob_url`
    # points at the served miniwob-plusplus HTML, which browsergym-miniwob does
    # not ship - without it the env raises "core is not defined" at first reset.
    "tasks", "benchmark", "miniwob_url",
    # Every multi-turn domain: the ONE turn cap. Read by training
    # (max_tool_calling_iterations), by the eval loop, and mapped to each
    # server's own env var (TEXTARENA_MAX_TURNS / FINQA_MAX_STEPS /
    # REPL_MAX_ITERATIONS). finqa's `max_steps` and repl's `max_iterations`
    # were per-domain aliases that let the three caps drift apart.
    "max_turns",
}

# Known eval keys. Closes the silent-passthrough gap that let a dead `ood_probes`
# block and a mistyped value slip through unnoticed.
# `reference_report` pins the over/under-thinking thresholds to another run's
# token distribution (normally the E0 base-model report). Without it each run
# uses its own P10/P75, which makes the rates invariant to a uniform change in
# length - so they cannot detect the compression E2 exists to produce, and two
# arms' rates are measured against two different yardsticks.
_KNOWN_EVAL_KEYS = {"temperature", "do_sample", "max_new_tokens", "agentic",
                    "reference_report"}
_KNOWN_EVAL_AGENTIC_KEYS = {"n_episodes", "splits"}
# Per-split keys. `env_config` is merged over training.env_config, so a split
# overrides only what shifts; `seed_offset` moves the split to a disjoint region
# of the seed -> question mapping.
_KNOWN_EVAL_SPLIT_KEYS = {"name", "n_episodes", "env_config", "seed_offset"}

# Whitelist of allowed sub-keys per reward. Catches typos in YAML that would
# otherwise pass through silently and leave the reward on its default.
_COMMON_REWARD_SUBKEYS = {"enabled", "weight"}
_KNOWN_REWARD_SUBKEYS: dict[str, set[str]] = {
    "token_length":  _COMMON_REWARD_SUBKEYS | {
        "max_len",
        "r_correct_short", "r_correct_long", "r_wrong_short", "r_wrong_long",
    },
    "env_reward":    _COMMON_REWARD_SUBKEYS,
    # E3 has no knobs: lambda is `weight`, the signal is the env's done flag.
    "non_termination": _COMMON_REWARD_SUBKEYS,
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
    "training.n_rollouts": (1, 256),
    "training.save_steps": (1, 100_000),
    "training.max_prompt_length": (1, 131072),
}


def warn_inert_scalars(rewards_cfg: dict, compose_method: str) -> list[str]:
    """Return warnings for reward knobs that do nothing as configured.

    Under `advantage_weighted` every component is z-scored per prompt-group, so
    a component with no within-group variance contributes exactly 0 no matter
    what weight it carries. Disabled rewards are skipped.
    """
    rc = rewards_cfg or {}
    warnings: list[str] = []

    if compose_method == "advantage_weighted":
        # E3 is a binary flag, so it has zero within-group variance in any group
        # where every rollout terminates - and there it contributes exactly 0.
        # The penalty then goes silent precisely where behavior is already good,
        # which is not the shaped reward the expose specifies.
        if (rc.get("non_termination") or {}).get("enabled"):
            warnings.append(
                "rewards.non_termination is binary, so advantage_weighted silences it in "
                "every prompt-group where all rollouts terminate (zero within-group "
                "variance contributes 0). Use compose_method: naive_sum for the lambda "
                "sweep, where weight is the penalty coefficient the expose defines."
            )

    return warnings


def _split_errors(splits) -> list[str]:
    """Validate eval.agentic.splits.

    Split names key the report and the per-split episodes file, so a missing or
    duplicate name silently overwrites another split's results - checked here
    rather than discovered after a 2h eval.
    """
    if splits is None:
        return []
    if not isinstance(splits, list):
        return [f"eval.agentic.splits must be a list, got {type(splits).__name__}"]
    errors = []
    seen = set()
    for i, s in enumerate(splits):
        if not isinstance(s, dict):
            errors.append(f"eval.agentic.splits[{i}] must be a mapping, got {s!r}")
            continue
        unknown = set(s) - _KNOWN_EVAL_SPLIT_KEYS
        if unknown:
            errors.append(
                f"Unknown eval.agentic.splits[{i}] keys: {sorted(unknown)}. "
                f"Known: {sorted(_KNOWN_EVAL_SPLIT_KEYS)}"
            )
        name = s.get("name")
        if not name:
            errors.append(f"eval.agentic.splits[{i}] is missing `name`")
        elif name in seen:
            errors.append(f"Duplicate eval.agentic.splits name: {name!r}")
        else:
            seen.add(name)
        env_cfg = s.get("env_config")
        if isinstance(env_cfg, dict):
            unknown_ec = set(env_cfg) - _KNOWN_ENV_CONFIG_KEYS
            if unknown_ec:
                errors.append(
                    f"Unknown eval.agentic.splits[{i}].env_config keys: "
                    f"{sorted(unknown_ec)}. Known: {sorted(_KNOWN_ENV_CONFIG_KEYS)}"
                )
    return errors


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

    # Agentic-only: every config runs against an OpenEnv environment.
    mode = (config.get("training") or {}).get("mode", "agentic")
    if mode != "agentic":
        errors.append(f"training.mode={mode!r}: only 'agentic' is supported")
    if _get_nested(config, "training.env") is None:
        errors.append("Missing required field: training.env (str) - OpenEnv environment id")

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

    for reward_name in _KNOWN_REWARD_SUBKEYS:        # excludes compose_method (a string)
        val = rewards.get(reward_name)
        if val is not None and not isinstance(val, dict):
            errors.append(
                f"rewards.{reward_name} must be a mapping (e.g. {{enabled: false}}), "
                f"got {type(val).__name__}: {val!r}"
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

    training = config.get("training")
    if isinstance(training, dict):
        unknown_tr = set(training) - _KNOWN_TRAINING_KEYS
        if unknown_tr:
            errors.append(
                f"Unknown training keys: {sorted(unknown_tr)}. "
                f"Known: {sorted(_KNOWN_TRAINING_KEYS)}"
            )
        env_server = training.get("env_server")
        if isinstance(env_server, dict):
            unknown_es = set(env_server) - _KNOWN_ENV_SERVER_KEYS
            if unknown_es:
                errors.append(
                    f"Unknown training.env_server keys: {sorted(unknown_es)}. "
                    f"Known: {sorted(_KNOWN_ENV_SERVER_KEYS)}"
                )

    env_config = (config.get("training") or {}).get("env_config")
    if isinstance(env_config, dict):
        unknown_ec = set(env_config) - _KNOWN_ENV_CONFIG_KEYS
        if unknown_ec:
            errors.append(
                f"Unknown training.env_config keys: {sorted(unknown_ec)}. "
                f"Known: {sorted(_KNOWN_ENV_CONFIG_KEYS)}"
            )

    eval_cfg = config.get("eval")
    if isinstance(eval_cfg, dict):
        unknown_eval = set(eval_cfg) - _KNOWN_EVAL_KEYS
        if unknown_eval:
            errors.append(
                f"Unknown eval keys: {sorted(unknown_eval)}. Known: {sorted(_KNOWN_EVAL_KEYS)}"
            )
        agentic = eval_cfg.get("agentic")
        if isinstance(agentic, dict):
            unknown_ag = set(agentic) - _KNOWN_EVAL_AGENTIC_KEYS
            if unknown_ag:
                errors.append(
                    f"Unknown eval.agentic keys: {sorted(unknown_ag)}. "
                    f"Known: {sorted(_KNOWN_EVAL_AGENTIC_KEYS)}"
                )
            errors.extend(_split_errors(agentic.get("splits")))

    unknown_top = set(config.keys()) - _KNOWN_TOP_LEVEL_KEYS
    if unknown_top:
        errors.append(
            f"Unknown top-level keys: {sorted(unknown_top)}. Known: {sorted(_KNOWN_TOP_LEVEL_KEYS)}"
        )

    if errors:
        msg = "Config validation failed:\n" + "\n".join(f" - {e}" for e in errors)
        raise ValueError(msg)
