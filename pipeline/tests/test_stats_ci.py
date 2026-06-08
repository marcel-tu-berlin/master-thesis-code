"""Statistical-CI upgrades (T1.4 Wilson, T2.3 threshold-in-bootstrap).

The bug being fixed: a percentile bootstrap on a binary 0/1 vector collapses to
a zero-width [p, p] interval at p=0 or p=1 (e.g. the capability_floor 6/6 case
reads as [1.0, 1.0], hiding all uncertainty). Wilson gives a proper interval.
For over/under-thinking rates the threshold was frozen before bootstrapping, so
its estimation variance never entered the CI — T2.3 recomputes it per replicate.
"""
import numpy as np

from eval.metrics import _wilson_ci, _thinking_rate, compute_metrics, SampleResult


# ---- T1.4: Wilson proportion CI ---------------------------------------------

def test_wilson_not_degenerate_at_p1():
    lo, hi = _wilson_ci(6, 6)          # 6/6 — the capability_floor saturation case
    assert hi == 1.0
    assert lo < 1.0                    # the fix: NOT [1.0, 1.0]
    assert 0.5 < lo < 0.75             # true Wilson ~0.61


def test_wilson_not_degenerate_at_p0():
    lo, hi = _wilson_ci(0, 6)
    assert lo == 0.0
    assert 0.25 < hi < 0.5             # ~0.39


def test_wilson_symmetric_midpoint():
    lo, hi = _wilson_ci(5, 10)
    assert lo < 0.5 < hi
    assert abs((lo + hi) / 2 - 0.5) < 0.05


def test_wilson_zero_n():
    assert _wilson_ci(0, 0) == (0.0, 0.0)


def test_accuracy_ci_uses_wilson_for_all_correct():
    # compute_metrics must surface a non-degenerate accuracy CI at acc=1.0.
    m = compute_metrics([SampleResult(correct=True, n_tokens=20) for _ in range(6)])
    assert m.accuracy == 1.0
    assert m.accuracy_ci_high == 1.0
    assert m.accuracy_ci_low < 1.0     # was [1.0, 1.0] under percentile bootstrap


# ---- T2.3: over/under-thinking rate CI propagates threshold variance --------

def test_thinking_rate_override_is_wilson_around_point():
    tokens = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    corrects = np.ones(10, dtype=bool)
    rate, thr, lo, hi = _thinking_rate(
        tokens, corrects, percentile=10, override=55.0, side="under",
        n_bootstrap=2000, ci=0.95,
    )
    assert thr == 55.0
    assert abs(rate - 0.5) < 1e-9      # {10..50} <= 55 among 10 correct
    assert lo < 0.5 < hi


def test_thinking_rate_override_degenerate_not_collapsed():
    tokens = np.array([10, 20, 30, 40], dtype=float)
    corrects = np.ones(4, dtype=bool)
    rate, thr, lo, hi = _thinking_rate(
        tokens, corrects, percentile=10, override=1000.0, side="under",
        n_bootstrap=2000, ci=0.95,
    )
    assert abs(rate - 1.0) < 1e-9
    assert hi == 1.0
    assert lo < 1.0                    # Wilson floor, not a collapsed [1, 1]


def test_thinking_rate_percentile_has_positive_width():
    rng = np.random.default_rng(0)
    tokens = rng.integers(10, 200, size=100).astype(float)
    corrects = rng.random(100) < 0.6
    rate, thr, lo, hi = _thinking_rate(
        tokens, corrects, percentile=10, override=None, side="under",
        n_bootstrap=3000, ci=0.95,
    )
    assert 0.0 <= lo <= hi <= 1.0
    assert hi - lo > 0.0               # threshold + sampling variance both present


def test_thinking_rate_no_correct_returns_none():
    tokens = np.array([10, 20, 30, 40], dtype=float)
    corrects = np.zeros(4, dtype=bool)
    assert _thinking_rate(tokens, corrects, 10, None, "under", 1000, 0.95) == (None, None, None, None)
