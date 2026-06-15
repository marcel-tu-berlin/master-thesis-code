import pytest

from training.mode import select_mode


def test_default_dataset():
    assert select_mode({"training": {}}) == "dataset"
    assert select_mode({}) == "dataset"


def test_agentic():
    assert select_mode({"training": {"mode": "agentic", "env": "reasoning_gym"}}) == "agentic"


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        select_mode({"training": {"mode": "bogus"}})
