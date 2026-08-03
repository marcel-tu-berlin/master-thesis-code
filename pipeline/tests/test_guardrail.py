"""Guardrail: warn when a reward does nothing for the resolved composer.

`advantage_weighted` z-scores each component per prompt-group, so a component
with zero within-group variance contributes exactly 0 regardless of its weight.
E3 (non_termination) is binary, so that is precisely what happens in every group
where all rollouts terminate - the penalty goes silent exactly where behaviour
is already good, which is not the shaped reward the expose specifies. The lambda
sweep therefore has to run under naive_sum.
"""
from training.config_schema import warn_inert_scalars


def test_warns_non_termination_under_advantage_weighted():
    cfg = {"non_termination": {"enabled": True, "weight": 1.0}}
    w = warn_inert_scalars(cfg, "advantage_weighted")
    assert any("non_termination" in s and "naive_sum" in s for s in w)


def test_non_termination_quiet_under_naive_sum():
    cfg = {"non_termination": {"enabled": True, "weight": 1.0}}
    assert warn_inert_scalars(cfg, "naive_sum") == []


def test_disabled_reward_is_not_flagged():
    cfg = {"non_termination": {"enabled": False, "weight": 1.0}}
    assert warn_inert_scalars(cfg, "advantage_weighted") == []


def test_no_rewards_configured_is_quiet():
    assert warn_inert_scalars({}, "advantage_weighted") == []
    assert warn_inert_scalars(None, "advantage_weighted") == []
