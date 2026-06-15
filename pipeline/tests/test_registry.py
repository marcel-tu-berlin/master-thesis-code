import pytest

from training.registry import MODEL_REGISTRY, get_model_config


def test_qwen3_1_7b_registered():
    cfg = get_model_config("qwen3-1.7b")
    assert cfg["model_name"] == "Qwen/Qwen3-1.7B"
    assert cfg["max_seq_length"] == 2048


def test_no_unsloth_prefixed_model_names():
    for slug, cfg in MODEL_REGISTRY.items():
        assert not cfg["model_name"].startswith("unsloth/"), slug


def test_unknown_slug_raises():
    with pytest.raises(KeyError):
        get_model_config("does-not-exist")
