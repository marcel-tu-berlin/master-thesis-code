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
    cfg = {
        "experiment_id": "e0",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "dataset"},
    }
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


@pytest.mark.parametrize("split", [False, True])
def test_enable_fill_is_an_explicit_boolean_in_training_and_splits(split):
    cfg = _agentic()
    cfg["training"]["env"] = "browsergym"
    env_cfg = {"enable_fill": True}
    if split:
        cfg["eval"] = {
            "agentic": {
                "splits": [{"name": "typing", "n_episodes": 3, "env_config": env_cfg}]
            }
        }
    else:
        cfg["training"]["env_config"] = env_cfg
    validate_config(cfg)
    env_cfg["enable_fill"] = "false"
    with pytest.raises(ValueError, match=r"enable_fill.*bool"):
        validate_config(cfg)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("tasks", ["read-table-2", [], [None], [42], [""], [" "]])
def test_tasks_require_a_nonempty_list_of_family_names(split, tasks):
    cfg = _agentic()
    cfg["training"]["env"] = "browsergym"
    env_cfg = {"tasks": ["read-table-2"]}
    if split:
        cfg["eval"] = {
            "agentic": {
                "splits": [{"name": "table", "n_episodes": 3, "env_config": env_cfg}]
            }
        }
    else:
        cfg["training"]["env_config"] = env_cfg
    validate_config(cfg)
    env_cfg["tasks"] = None
    validate_config(cfg)
    del env_cfg["tasks"]
    validate_config(cfg)
    env_cfg["tasks"] = tasks
    with pytest.raises(ValueError, match=r"env_config.tasks.*nonempty list"):
        validate_config(cfg)
