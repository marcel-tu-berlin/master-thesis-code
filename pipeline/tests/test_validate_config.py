"""validate_config rejects malformed reward values."""
import pytest
from training.config_schema import validate_config


def _base():
    return {"experiment_id": "x", "model": {"slug": "qwen-7b"},
            "training": {"dataset": "openai/gsm8k"}, "rewards": {}}


def test_rejects_bool_reward_value():
    cfg = _base(); cfg["rewards"]["numeric"] = False
    with pytest.raises(ValueError, match="numeric"):
        validate_config(cfg)


def test_accepts_dict_reward_value():
    cfg = _base(); cfg["rewards"]["numeric"] = {"enabled": False}
    validate_config(cfg)  # must not raise


def test_compose_method_string_still_ok():
    cfg = _base(); cfg["rewards"]["compose_method"] = "naive_sum"
    validate_config(cfg)  # compose_method is a string, not a reward dict
