"""Guardrail: warn when a token_entropy knob does nothing for the resolved composer.

token_entropy.reward_scale is inert under advantage_weighted (per-group z-scoring
cancels the global scalar) but live under naive_sum.
"""
from training.config_schema import warn_inert_scalars


def test_warns_reward_scale_under_advantage_weighted():
    # token_entropy.reward_scale is inert under advantage_weighted (z-scoring
    # cancels the global scalar).
    cfg = {"token_entropy": {"enabled": True, "reward_scale": 0.5}}
    w = warn_inert_scalars(cfg, "advantage_weighted")
    assert any("reward_scale" in s for s in w)


def test_entropy_quiet_under_naive_sum():
    # token_entropy.reward_scale is live under naive_sum -> no warning.
    cfg = {"token_entropy": {"enabled": True, "reward_scale": 0.5}}
    assert warn_inert_scalars(cfg, "naive_sum") == []
