"""Keep active experiments on the retained seed-4016 comparison's limits."""

from pathlib import Path

import pytest
import yaml

from eval.agentic_eval import _completion_budget
from training.config_schema import resolve_max_turns

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_active_experiment_budgets_match_retained_trio(path):
    config = yaml.safe_load(path.read_text())
    # Explicit keys are required so frozen configurations retain the contract.
    assert config["model"]["max_seq_length"] == 8192
    assert config["training"]["max_prompt_length"] == 4096
    assert config["eval"]["max_new_tokens"] == 4096
    assert (
        config["model"]["max_seq_length"] - config["training"]["max_prompt_length"]
        == 4096
    )
    assert _completion_budget(config, config["model"]["max_seq_length"]) == 4096
    assert config["rewards"]["successful_length"]["max_len"] == 4096
    assert config["training"]["env_config"]["max_turns"] == 8
    assert resolve_max_turns(config["training"]["env_config"]) == 8
