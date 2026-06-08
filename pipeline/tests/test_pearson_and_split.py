"""Pearson p-value precision (T2.4) and difficulty-plot split selection (T2.5).

A strongly significant correlation rounded to 6 dp prints as the literal 0.0,
which reads as "p exactly zero" — keep full precision and format in scientific
notation. And difficulty labels live only on Hendrycks MATH, so the scatter
must read whichever split actually carries a correlation, not always id_split
(empty for GSM8K-trained runs, where MATH is the near-OOD probe).
"""
from eval.report import _metrics_dict
from eval.plots import _split_with_difficulty
from eval.metrics import EvalMetrics, SampleResult
from eval.ood_probes import OODResults


def test_pearson_p_kept_full_precision():
    m = EvalMetrics(pearson_difficulty_length=0.42, pearson_p_value=1.23e-12, n_samples=50)
    d = _metrics_dict(m)
    assert d["pearson_p_value"] == 1.23e-12        # was round(.,6) -> 0.0


def test_split_with_difficulty_prefers_the_split_with_a_correlation():
    # GSM8K-trained: id_split (GSM8K) has no difficulty; near_ood (MATH) does.
    id_m = EvalMetrics(pearson_difficulty_length=None, raw=[SampleResult(True, 100)])
    near_m = EvalMetrics(
        pearson_difficulty_length=0.5, pearson_p_value=0.001,
        raw=[SampleResult(True, 100, difficulty=3.0) for _ in range(6)],
    )
    name, chosen = _split_with_difficulty(OODResults(id_split=id_m, near_ood=near_m))
    assert name == "near_ood"
    assert chosen is near_m


def test_split_with_difficulty_none_when_no_difficulty_anywhere():
    id_m = EvalMetrics(raw=[SampleResult(True, 100)])
    name, chosen = _split_with_difficulty(OODResults(id_split=id_m))
    assert chosen is None
