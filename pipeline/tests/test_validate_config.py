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
