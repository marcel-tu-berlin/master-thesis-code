import pytest

from training.config_schema import validate_config


def _agentic():
    return {
        "experiment_id": "e5-agentic-rg",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "agentic", "env": "reasoning_gym"},
        "rewards": {"env_reward": {"enabled": True, "weight": 1.0}},
    }


def test_agentic_valid_without_dataset():
    validate_config(_agentic())  # must not raise


def test_agentic_requires_env():
    cfg = _agentic()
    del cfg["training"]["env"]
    with pytest.raises(ValueError, match="env"):
        validate_config(cfg)


def test_dataset_mode_still_requires_dataset():
    cfg = {"experiment_id": "e0", "model": {"slug": "qwen3-1.7b"}, "training": {"mode": "dataset"}}
    with pytest.raises(ValueError, match="dataset"):
        validate_config(cfg)


def test_env_reward_is_known_key():
    validate_config(_agentic())  # env_reward must not be rejected as an unknown key


def test_bad_mode_rejected():
    cfg = _agentic()
    cfg["training"]["mode"] = "bogus"
    with pytest.raises(ValueError, match="mode"):
        validate_config(cfg)
