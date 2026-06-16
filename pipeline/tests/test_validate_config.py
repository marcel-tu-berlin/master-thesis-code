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
