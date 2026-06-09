"""Guardrail: warn when a token_length knob does nothing for the resolved shape.

The guardrail resolves `shape` the same way the registry builder does (default
cosine), so a warning always matches what actually gets built. Two failure modes:

1. shape: cosine (the default) never reads the linear-penalty knobs alpha/schedule
   — they are dead under ANY composer. Setting them signals a mistake (you
   probably meant shape: linear), so warn.
2. shape: linear under advantage_weighted: per-group z-scoring cancels the global
   alpha/schedule scalars, so they are inert. Under naive_sum they are live, quiet.

token_entropy.reward_scale and effort_proxy knobs are inert only under
advantage_weighted (z-scoring), so they stay gated on that composer.
"""
from training.config_schema import warn_inert_scalars


def test_cosine_default_warns_on_stray_linear_knobs():
    # No shape -> resolves to cosine (same default as the builder). alpha/schedule
    # are linear-only knobs the cosine reward ignores -> warn under ANY composer.
    cfg = {"token_length": {"enabled": True, "alpha": 0.05, "schedule": "cosine"}}
    for method in ("advantage_weighted", "naive_sum"):
        w = warn_inert_scalars(cfg, method)
        assert any("token_length.alpha" in s for s in w), method
        assert any("schedule" in s for s in w), method


def test_explicit_cosine_warns_on_stray_alpha():
    # Explicit cosine + non-default alpha -> ignored by the cosine reward -> warn.
    cfg = {"token_length": {"enabled": True, "shape": "cosine", "alpha": 0.05}}
    w = warn_inert_scalars(cfg, "advantage_weighted")
    assert any("token_length.alpha" in s for s in w)


def test_cosine_default_alpha_is_quiet():
    # Default alpha (0.001), no cosine schedule -> boilerplate, not tuning intent.
    cfg = {"token_length": {"enabled": True, "alpha": 0.001}}
    assert warn_inert_scalars(cfg, "advantage_weighted") == []
    assert warn_inert_scalars(cfg, "naive_sum") == []


def test_linear_inert_under_advantage_weighted():
    # shape: linear under advantage_weighted -> z-scoring cancels alpha/schedule.
    cfg = {"token_length": {"enabled": True, "shape": "linear", "alpha": 0.05, "schedule": "cosine"}}
    w = warn_inert_scalars(cfg, "advantage_weighted")
    assert any("token_length.alpha" in s for s in w)
    assert any("schedule" in s for s in w)


def test_linear_live_under_naive_sum():
    # shape: linear under naive_sum -> alpha/schedule are live -> quiet.
    cfg = {"token_length": {"enabled": True, "shape": "linear", "alpha": 0.05, "schedule": "cosine"}}
    assert warn_inert_scalars(cfg, "naive_sum") == []


def test_warns_reward_scale_and_effort_metric():
    cfg = {
        "token_entropy": {"enabled": True, "reward_scale": 0.5},
        "effort_proxy": {"enabled": True, "metric": "flops"},
    }
    w = warn_inert_scalars(cfg, "advantage_weighted")
    assert any("reward_scale" in s for s in w)
    assert any("metric" in s for s in w)


def test_entropy_and_effort_quiet_under_naive_sum():
    # These knobs are live under naive_sum -> no warning.
    cfg = {
        "token_entropy": {"enabled": True, "reward_scale": 0.5},
        "effort_proxy": {"enabled": True, "metric": "flops"},
    }
    assert warn_inert_scalars(cfg, "naive_sum") == []


def test_disabled_rewards_not_warned():
    cfg = {"token_length": {"enabled": False, "shape": "linear", "alpha": 0.05, "schedule": "cosine"}}
    assert warn_inert_scalars(cfg, "advantage_weighted") == []
