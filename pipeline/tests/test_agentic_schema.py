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


# build_domain lives in train.py, which imports the GPU stack (trl) at module
# load, so these run on the GPU box / at the L4 smoke and skip cleanly on CPU.

def test_build_domain_dispatches_textarena():
    pytest.importorskip("trl")
    from training.train import build_domain
    from domains.textarena import TextArenaDomain
    d = build_domain({"training": {"env": "textarena"}})
    assert isinstance(d, TextArenaDomain)


def test_build_domain_dispatches_reasoning_gym():
    pytest.importorskip("trl")
    from training.train import build_domain
    from domains.reasoning_gym import ReasoningGymDomain
    d = build_domain({"training": {"env": "reasoning_gym"}})
    assert isinstance(d, ReasoningGymDomain)


def test_build_domain_rejects_unknown_env():
    pytest.importorskip("trl")
    from training.train import build_domain
    with pytest.raises(NotImplementedError):
        build_domain({"training": {"env": "nope"}})
