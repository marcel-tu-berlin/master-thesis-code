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


# --- correct-only token mean: the efficiency number ---
# `mean_token_count` pools failures, which run to the generation cap, so a change
# in failure rate there is indistinguishable from a change in length. That is the
# confound that voided e9-e21, and the honest number lived only inside plots.py.

def test_mean_token_count_correct_excludes_failures():
    m = compute_metrics(
        [SampleResult(True, 100, n_steps=1), SampleResult(True, 120, n_steps=1),
         SampleResult(False, 900, n_steps=1), SampleResult(False, 920, n_steps=1)]
    )
    assert m.mean_token_count == 510.0            # pooled, dominated by failures
    assert m.mean_token_count_correct == 110.0    # the efficiency number


def test_mean_token_count_correct_is_none_without_correct_episodes():
    # None, not 0.0 - zero would read as perfectly efficient.
    m = compute_metrics([SampleResult(False, 900, n_steps=1)])
    assert m.mean_token_count_correct is None
    assert m.mean_token_count_correct_ci_low is None


def test_mean_token_count_correct_ci_brackets_the_mean():
    m = compute_metrics([SampleResult(True, t, n_steps=1) for t in (90, 100, 110, 120)])
    assert m.mean_token_count_correct_ci_low <= m.mean_token_count_correct
    assert m.mean_token_count_correct <= m.mean_token_count_correct_ci_high


def test_correct_only_mean_agrees_with_the_plotting_helper():
    # Two implementations over different input types (SampleResult vs report
    # samples). Pin them together so they cannot drift.
    from eval.agentic_eval import _metrics_to_dict
    from eval.plots import _mean_ci_on_correct
    results = [SampleResult(True, 100, n_steps=1), SampleResult(True, 140, n_steps=1),
               SampleResult(False, 900, n_steps=1)]
    m = compute_metrics(results)
    mean_from_plots, _, _ = _mean_ci_on_correct(_metrics_to_dict(m)["samples"])
    assert mean_from_plots == m.mean_token_count_correct


# --- thinking-rate thresholds must be pinnable to a reference run ---

def _halved(results):
    return [SampleResult(r.correct, r.n_tokens // 2, n_steps=1) for r in results]


def _spread():
    # 20 episodes, wide token spread, half correct.
    return [SampleResult(i % 2 == 0, 100 + 100 * i, n_steps=1) for i in range(20)]


def test_percentile_thresholds_cannot_detect_compression():
    # Documents the defect: halving every token count leaves both rates
    # untouched, because the per-run percentile moves with the distribution.
    base = compute_metrics(_spread())
    half = compute_metrics(_halved(_spread()))
    assert base.overthinking_rate == half.overthinking_rate
    assert base.underthinking_rate == half.underthinking_rate


def test_pinned_thresholds_do_detect_compression():
    base = compute_metrics(_spread())
    half = compute_metrics(
        _halved(_spread()),
        underthinking_threshold=base.underthinking_threshold,
        overthinking_threshold=base.overthinking_threshold,
    )
    assert half.underthinking_rate > base.underthinking_rate   # more short answers
    assert half.overthinking_rate < base.overthinking_rate     # fewer long ones


def test_reference_thresholds_round_trip_through_a_report(tmp_path):
    import json
    from eval.agentic_eval import _build_report, _reference_thresholds
    ref = tmp_path / "eval_report.json"
    ref.write_text(json.dumps(_build_report(
        {"experiment_id": "e0"}, None, {"held_out": compute_metrics(_spread())})))

    loaded = _reference_thresholds({"reference_report": str(ref)})
    assert set(loaded) == {"held_out"}

    pinned = compute_metrics(_spread(), **loaded["held_out"])
    direct = compute_metrics(_spread())
    assert pinned.underthinking_threshold == direct.underthinking_threshold


def test_no_reference_configured_means_no_override():
    from eval.agentic_eval import _reference_thresholds
    assert _reference_thresholds({}) == {}
    assert _reference_thresholds({"temperature": 0.0}) == {}
