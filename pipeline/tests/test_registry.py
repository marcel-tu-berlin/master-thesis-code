import pytest

from training.registry import MODEL_REGISTRY, get_model_config


def test_qwen3_1_7b_registered():
    cfg = get_model_config("qwen3-1.7b")
    assert cfg["model_name"] == "Qwen/Qwen3-1.7B"
    assert cfg["max_seq_length"] == 2048


# Mirror names stay banned by default: the pipeline dropped the Unsloth loader
# for plain `AutoModelForCausalLM` in cf7f0db, and a re-uploaded checkpoint under
# a stock loader is a silent confound. Pinned exceptions are allowed only where
# the official repo is unreachable and the mirror has been checked - see the
# registry comment for what was verified on this one.
ALLOWED_MIRRORS = {"unsloth/Llama-3.2-1B-Instruct"}


def test_no_unsloth_prefixed_model_names():
    for slug, cfg in MODEL_REGISTRY.items():
        name = cfg["model_name"]
        if name in ALLOWED_MIRRORS:
            continue
        assert not name.startswith("unsloth/"), slug


def test_unknown_slug_raises():
    with pytest.raises(KeyError):
        get_model_config("does-not-exist")
