"""validate_config rejects malformed reward values (agentic configs)."""
import pytest
from training.config_schema import validate_config


def _base():
    return {"experiment_id": "x", "model": {"slug": "qwen3-1.7b"},
            "training": {"mode": "agentic", "env": "reasoning_gym"}, "rewards": {}}


def test_rejects_bool_reward_value():
    cfg = _base(); cfg["rewards"]["env_reward"] = False
    with pytest.raises(ValueError, match="env_reward"):
        validate_config(cfg)


def test_accepts_dict_reward_value():
    cfg = _base(); cfg["rewards"]["env_reward"] = {"enabled": True}
    validate_config(cfg)  # must not raise


def test_compose_method_string_still_ok():
    cfg = _base(); cfg["rewards"]["compose_method"] = "naive_sum"
    validate_config(cfg)  # compose_method is a string, not a reward dict


def test_requires_env():
    cfg = _base(); del cfg["training"]["env"]
    with pytest.raises(ValueError, match="training.env"):
        validate_config(cfg)


def test_rejects_non_agentic_mode():
    cfg = _base(); cfg["training"]["mode"] = "dataset"
    with pytest.raises(ValueError, match="agentic"):
        validate_config(cfg)


def _agentic_base():
    return {
        "experiment_id": "t",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "agentic", "env": "reasoning_gym",
                     "env_config": {"dataset": "chain_sum", "size": 8}},
        "rewards": {"env_reward": {"enabled": True}},
    }


def test_accepts_textarena_env_config_keys():
    cfg = _agentic_base()
    cfg["training"]["env"] = "textarena"
    cfg["training"]["env_config"] = {"env_id": "Wordle-v0", "num_players": 1,
                                     "max_turns": 6, "size": 8}
    validate_config(cfg)  # must not raise


def test_rejects_unknown_env_config_key():
    cfg = _agentic_base()
    cfg["training"]["env_config"]["datsaet"] = "typo"   # misspelled
    with pytest.raises(ValueError, match="env_config"):
        validate_config(cfg)


def test_accepts_known_eval_keys():
    cfg = _agentic_base()
    cfg["eval"] = {"temperature": 0.0, "do_sample": False, "agentic": {"n_episodes": 100}}
    validate_config(cfg)  # must not raise


def test_rejects_unknown_eval_key():
    cfg = _agentic_base()
    cfg["eval"] = {"ood_probes": {"far": "mmlu"}}   # the gap that slipped through before
    with pytest.raises(ValueError, match="eval"):
        validate_config(cfg)


def test_rejects_unknown_eval_agentic_key():
    cfg = _agentic_base()
    cfg["eval"] = {"agentic": {"n_epsiodes": 100}}  # typo
    with pytest.raises(ValueError, match="eval.agentic"):
        validate_config(cfg)


# --- training block: the last config section with no key whitelist ---
# Every training key has a default in grpo_runner._grpo_config, so a typo used to
# validate fine, be ignored, and leave the run training at a geometry its own
# frozen config contradicted.

def test_rejects_unknown_training_key():
    cfg = _agentic_base()
    cfg["training"]["max_prompt_lenght"] = 6144      # the real typo shape
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
        max_prompt_length=1024, max_steps=300, save_steps=100,
        n_rollouts=8, batch_size=1, micro_batch_size=2,
        learning_rate=5e-6, kl_beta=0.001, temperature=1.0,
        weight_decay=0.1, warmup_ratio=0.1,
    )
    validate_config(cfg)  # must not raise


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
