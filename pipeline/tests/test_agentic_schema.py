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


# build_domain moved out of train.py (which imports trl at module load) into
# domains/, so these no longer need the GPU stack and run on CPU.

from domains import build_domain


def test_build_domain_dispatches_reasoning_gym():
    from domains.reasoning_gym import ReasoningGymDomain
    d = build_domain({"training": {"env": "reasoning_gym"}})
    assert isinstance(d, ReasoningGymDomain)


def test_build_domain_dispatches_browsergym():
    # The domain added last, which is the case the duplicated dispatch used to
    # break on: the stale copy in eval/runner.py never learned the later ones,
    # so `python -m eval.runner` raised NotImplementedError on exactly this one.
    from domains.browsergym import BrowserGymDomain
    d = build_domain({"training": {"env": "browsergym"}})
    assert isinstance(d, BrowserGymDomain)


def test_build_domain_rejects_unknown_env():
    with pytest.raises(NotImplementedError):
        build_domain({"training": {"env": "nope"}})
