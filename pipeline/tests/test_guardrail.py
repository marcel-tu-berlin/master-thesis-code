"""Guardrail: warn when a scalar knob is inert under advantage_weighted.

Per-group z-scoring (compose.py) cancels any global scalar, so token_length.alpha,
the cosine *schedule*, token_entropy.reward_scale, effort_proxy.alpha, and the
effort metric (flops/gpu_time differ from token_count only by a constant) do
nothing under advantage_weighted. The writeup wasted a run on exactly this. The
guardrail surfaces it; under naive_sum these knobs are live, so it stays quiet.
"""
from training.config_schema import warn_inert_scalars


def test_quiet_under_naive_sum():
    cfg = {"token_length": {"enabled": True, "alpha": 0.05, "schedule": "cosine"}}
    assert warn_inert_scalars(cfg, "naive_sum") == []


def test_warns_token_length_alpha_and_schedule():
    cfg = {"token_length": {"enabled": True, "alpha": 0.05, "schedule": "cosine"}}
    w = warn_inert_scalars(cfg, "advantage_weighted")
    assert any("token_length.alpha" in s for s in w)
    assert any("schedule" in s for s in w)


def test_default_alpha_constant_schedule_is_quiet():
    cfg = {"token_length": {"enabled": True, "alpha": 0.001}}
    assert warn_inert_scalars(cfg, "advantage_weighted") == []


def test_cosine_shape_does_not_warn_on_alpha():
    # shape: cosine uses correctness-coupled endpoints, not a global scalar.
    cfg = {"token_length": {"enabled": True, "shape": "cosine", "alpha": 0.05}}
    assert warn_inert_scalars(cfg, "advantage_weighted") == []


def test_warns_reward_scale_and_effort_metric():
    cfg = {
        "token_entropy": {"enabled": True, "reward_scale": 0.5},
        "effort_proxy": {"enabled": True, "metric": "flops"},
    }
    w = warn_inert_scalars(cfg, "advantage_weighted")
    assert any("reward_scale" in s for s in w)
    assert any("metric" in s for s in w)


def test_disabled_rewards_not_warned():
    cfg = {"token_length": {"enabled": False, "alpha": 0.05, "schedule": "cosine"}}
    assert warn_inert_scalars(cfg, "advantage_weighted") == []
