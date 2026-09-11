"""validate_config rejects malformed reward values (agentic configs)."""

import re

import pytest

from training.config_schema import validate_config


def _base():
    return {
        "experiment_id": "x",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "agentic", "env": "reasoning_gym"},
        "rewards": {},
    }


def test_rejects_bool_reward_value():
    cfg = _base()
    cfg["rewards"]["env_reward"] = False
    with pytest.raises(ValueError, match="env_reward"):
        validate_config(cfg)


def test_accepts_dict_reward_value():
    cfg = _base()
    cfg["rewards"]["env_reward"] = {"enabled": True}
    validate_config(cfg)  # must not raise


def test_compose_method_string_still_ok():
    cfg = _base()
    cfg["rewards"]["compose_method"] = "naive_sum"
    validate_config(cfg)  # compose_method is a string, not a reward dict


def test_requires_env():
    cfg = _base()
    del cfg["training"]["env"]
    with pytest.raises(ValueError, match=re.escape("training.env")):
        validate_config(cfg)


def test_rejects_non_agentic_mode():
    cfg = _base()
    cfg["training"]["mode"] = "dataset"
    with pytest.raises(ValueError, match="agentic"):
        validate_config(cfg)


def _agentic_base():
    return {
        "experiment_id": "t",
        "model": {"slug": "qwen3-1.7b"},
        "training": {
            "mode": "agentic",
            "env": "reasoning_gym",
            "env_config": {"dataset": "chain_sum", "size": 8},
        },
        "rewards": {"env_reward": {"enabled": True}},
    }


def test_accepts_browsergym_env_config_keys():
    cfg = _agentic_base()
    cfg["training"]["env"] = "browsergym"
    cfg["training"]["env_config"] = {
        "tasks": ["click-option"],
        "benchmark": "miniwob",
        "miniwob_url": "http://localhost:8080/miniwob/",
        "max_turns": 6,
    }
    validate_config(cfg)  # must not raise


def test_rejects_browsergym_seed_block_above_uint32():
    cfg = _agentic_base()
    cfg["training"]["env"] = "browsergym"
    cfg["seed"] = 4294
    with pytest.raises(ValueError, match="uint32"):
        validate_config(cfg)


def test_accepts_last_complete_browsergym_seed_block():
    cfg = _agentic_base()
    cfg["training"]["env"] = "browsergym"
    cfg["seed"] = 4293
    validate_config(cfg)


def test_rejects_unknown_env_config_key():
    cfg = _agentic_base()
    cfg["training"]["env_config"]["datsaet"] = "typo"  # misspelled
    with pytest.raises(ValueError, match="env_config"):
        validate_config(cfg)


def test_accepts_known_eval_keys():
    cfg = _agentic_base()
    cfg["eval"] = {
        "temperature": 0.0,
        "do_sample": False,
        "agentic": {"n_episodes": 100},
    }
    validate_config(cfg)  # must not raise


def test_rejects_unknown_eval_key():
    cfg = _agentic_base()
    cfg["eval"] = {"ood_probes": {"far": "mmlu"}}  # the gap that slipped through before
    with pytest.raises(ValueError, match="eval"):
        validate_config(cfg)


def test_rejects_unknown_eval_agentic_key():
    cfg = _agentic_base()
    cfg["eval"] = {"agentic": {"n_epsiodes": 100}}  # typo
    with pytest.raises(ValueError, match=re.escape("eval.agentic")):
        validate_config(cfg)


# --- training block: the last config section with no key whitelist ---
# Every training key has a default in grpo_runner._grpo_config, so a typo used to
# validate fine, be ignored, and leave the run training at a geometry its own
# frozen config contradicted.


def test_rejects_unknown_training_key():
    cfg = _agentic_base()
    cfg["training"]["max_prompt_lenght"] = 6144  # the real typo shape
    with pytest.raises(ValueError, match="training keys"):
        validate_config(cfg)


def test_rejects_training_key_that_is_read_nowhere():
    # Both were range-checked but never read: grad accum is derived from
    # batch_size / micro_batch_size, and dataset size comes from env_config.size.
    for dead in ("gradient_accumulation_steps", "dataset_size_limit"):
        cfg = _agentic_base()
        cfg["training"][dead] = 8
        with pytest.raises(ValueError, match="training keys"):
            validate_config(cfg)


def test_accepts_every_training_key_the_code_reads():
    cfg = _agentic_base()
    cfg["training"].update(
        max_prompt_length=1024,
        max_steps=300,
        save_steps=100,
        n_rollouts=8,
        batch_size=1,
        micro_batch_size=2,
        learning_rate=5e-6,
        kl_beta=0.001,
        temperature=1.0,
        weight_decay=0.1,
        warmup_ratio=0.1,
        vllm_importance_sampling_mode="token_truncate",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        num_iterations=2,
        use_liger_kernel=True,
        loss_type="dapo",
    )
    validate_config(cfg)  # must not raise


def test_rejects_zero_num_iterations():
    cfg = _agentic_base()
    cfg["training"]["num_iterations"] = 0
    with pytest.raises(ValueError, match="num_iterations"):
        validate_config(cfg)


def test_rejects_unknown_model_key():
    # `model` was the last block without a whitelist: a typo here fell back to
    # the runner default while the frozen config recorded the intended value.
    cfg = _agentic_base()
    cfg["model"]["lora_rnk"] = 8
    with pytest.raises(ValueError, match="model keys"):
        validate_config(cfg)


def test_accepts_every_model_key_the_code_reads():
    cfg = _agentic_base()
    cfg["model"].update(
        lora_r=16,
        lora_alpha=32,
        load_in_4bit=False,
        max_seq_length=8192,
        use_vllm=True,
        gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
    )
    validate_config(cfg)  # must not raise


def test_accepts_every_vllm_importance_sampling_mode():
    # The four TRL 1.6 modes plus "off" (correction disabled). Absent is also
    # legal and means TRL's default, so pre-fix frozen configs keep their
    # recorded semantics.
    for mode in (
        "token_truncate",
        "token_mask",
        "sequence_truncate",
        "sequence_mask",
        "off",
    ):
        cfg = _agentic_base()
        cfg["training"]["vllm_importance_sampling_mode"] = mode
        validate_config(cfg)  # must not raise


def test_rejects_unknown_vllm_importance_sampling_mode():
    # A typo here would silently fall back to TRL's sequence_mask default,
    # which is the exact filter the key exists to turn off.
    cfg = _agentic_base()
    cfg["training"]["vllm_importance_sampling_mode"] = "token_trunacte"
    with pytest.raises(ValueError, match="vllm_importance_sampling_mode"):
        validate_config(cfg)


def test_rejects_unknown_env_server_key():
    cfg = _agentic_base()
    cfg["training"]["env_server"] = {"repo_pth": "/workspace/OpenEnv/envs"}
    with pytest.raises(ValueError, match="env_server"):
        validate_config(cfg)


# --- split seed_offset must stay inside one seed's question block ---
# The eval seed base is seed * SEED_BLOCK + offset, so an offset at or above
# SEED_BLOCK lands inside the NEXT seed's block: seed 42's shifted split would
# evaluate on exactly the questions the seed-43 replicate trained on, silently.


def _cfg_with_offset(offset):
    cfg = _agentic_base()
    cfg["eval"] = {"agentic": {"splits": [{"name": "s", "seed_offset": offset}]}}
    return cfg


def test_accepts_split_seed_offset_below_block():
    validate_config(_cfg_with_offset(200_000))  # must not raise


def test_rejects_split_seed_offset_at_or_above_seed_block():
    with pytest.raises(ValueError, match="seed_offset"):
        validate_config(_cfg_with_offset(1_000_000))


def test_rejects_negative_split_seed_offset():
    with pytest.raises(ValueError, match="seed_offset"):
        validate_config(_cfg_with_offset(-1))


def test_rejects_non_int_split_seed_offset():
    with pytest.raises(ValueError, match="seed_offset"):
        validate_config(_cfg_with_offset("far"))


# --- the whole split range must dodge training questions and the block edge ---
# Bounding only the offset let two contaminations validate: an offset below
# env_config.size replays this seed's own training questions, and an offset
# near SEED_BLOCK runs its last episodes in the next seed's block, whose
# bottom is that seed's training range. Both inflate "held-out" accuracy.


def test_rejects_split_range_crossing_the_next_seed_block():
    # 999_950 < SEED_BLOCK, but the default 100 episodes run through 1_000_049.
    with pytest.raises(ValueError, match="crosses SEED_BLOCK"):
        validate_config(_cfg_with_offset(999_950))


def test_accepts_split_range_ending_exactly_at_the_block_edge():
    cfg = _agentic_base()
    cfg["eval"] = {
        "agentic": {"splits": [{"name": "s", "seed_offset": 999_950, "n_episodes": 50}]}
    }
    validate_config(cfg)  # 999_950 + 50 == SEED_BLOCK: last episode is 999_999


def test_rejects_split_offset_inside_the_training_range():
    # _agentic_base trains on env_config.size == 8 questions, offsets [0, 8).
    with pytest.raises(ValueError, match="training questions"):
        validate_config(_cfg_with_offset(4))


def test_accepts_split_offset_at_the_training_range_end():
    validate_config(_cfg_with_offset(8))  # first offset past the trained ones


# --- an explicit max_turns the resolver would silently rewrite is rejected ---
# resolve_max_turns coerces 0 and negatives to 1, so a config stating
# max_turns: 0 used to train and eval as a 1-turn episode while its frozen
# config recorded 0 - the recorded cap and the executed cap disagreed.


def test_rejects_explicit_zero_or_negative_max_turns():
    for bad in (0, -3):
        cfg = _agentic_base()
        cfg["training"]["env_config"]["max_turns"] = bad
        with pytest.raises(ValueError, match="max_turns"):
            validate_config(cfg)


def test_rejects_non_int_max_turns():
    cfg = _agentic_base()
    cfg["training"]["env_config"]["max_turns"] = "six"
    with pytest.raises(ValueError, match="max_turns"):
        validate_config(cfg)


def test_rejects_zero_max_turns_in_a_split_env_config():
    cfg = _agentic_base()
    cfg["eval"] = {
        "agentic": {"splits": [{"name": "s", "env_config": {"max_turns": 0}}]}
    }
    with pytest.raises(ValueError, match="max_turns"):
        validate_config(cfg)


# --- one turn-cap resolution for training and eval ---


def test_resolve_max_turns_defaults_to_one_on_both_sides():
    # Unset max_turns must mean the same episode process in training and eval.
    # The old pair of defaults - TRL capped at 1, eval looping 8 - measured a
    # policy under an episode length it never trained with.
    from training.config_schema import resolve_max_turns

    assert resolve_max_turns(None) == 1
    assert resolve_max_turns({}) == 1
    assert resolve_max_turns({"max_turns": 0}) == 1
    assert resolve_max_turns({"max_turns": 6}) == 6


# --- training.scale_rewards: the std TRL divides advantages by ---


def _scale_base():
    return {
        "experiment_id": "x",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "agentic", "env": "reasoning_gym"},
    }


def test_scale_rewards_accepts_trl_modes():
    from training.config_schema import validate_config

    for mode in ("group", "batch", "none"):
        cfg = _scale_base()
        cfg["training"]["scale_rewards"] = mode
        validate_config(cfg)


def test_scale_rewards_rejects_unknown_mode():
    import pytest

    from training.config_schema import validate_config

    cfg = _scale_base()
    cfg["training"]["scale_rewards"] = "grouped"
    with pytest.raises(ValueError, match="scale_rewards"):
        validate_config(cfg)


def test_loss_type_is_explicitly_validated():
    cfg = _scale_base()
    cfg["training"]["loss_type"] = "dapo"
    validate_config(cfg)

    cfg["training"]["loss_type"] = "token_mean"
    with pytest.raises(ValueError, match="loss_type"):
        validate_config(cfg)
