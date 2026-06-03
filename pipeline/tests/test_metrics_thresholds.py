"""Fixed-reference thinking thresholds: cross-run comparability.

Per-run percentile thresholds make "overthinking rate" incomparable across runs
(each run is scored against its own length distribution). compute_metrics must
accept absolute overrides, and load_reference_thresholds must derive per-split
P10/P75 from a reference (e0) eval_report.json.
"""
import json

import numpy as np

from eval.metrics import SampleResult, compute_metrics, load_reference_thresholds
from eval.ood_probes import _resolve_reference_thresholds


def _results(tokens, correct=None):
    correct = correct if correct is not None else [True] * len(tokens)
    return [SampleResult(correct=c, n_tokens=t) for t, c in zip(tokens, correct)]


def test_override_thresholds_used_instead_of_percentile():
    tokens = list(range(10, 101, 10))  # 10..100, all correct
    m = compute_metrics(_results(tokens), underthinking_threshold=55.0, overthinking_threshold=55.0)
    assert m.underthinking_threshold == 55.0
    assert m.overthinking_threshold == 55.0
    assert abs(m.underthinking_rate - 0.5) < 1e-9   # <=55: {10,20,30,40,50}
    assert abs(m.overthinking_rate - 0.5) < 1e-9    # >55:  {60,70,80,90,100}


def test_without_override_uses_per_run_percentile():
    tokens = list(range(10, 101, 10))
    m = compute_metrics(_results(tokens))
    expected = float(np.percentile(np.array(tokens, dtype=float), 10))
    assert abs(m.underthinking_threshold - expected) < 1e-9


def test_load_reference_thresholds_per_split(tmp_path):
    id_tokens = list(range(10, 101, 10))
    report = {
        "results": {
            "id_split": {"samples": [{"correct": True, "n_tokens": t} for t in id_tokens]},
            "near_ood": {"samples": [{"correct": False, "n_tokens": t} for t in range(100, 1001, 100)]},
            "capability_floor": {"samples": [{"correct": True, "n_tokens": 5}]},  # <4 -> skipped
        }
    }
    p = tmp_path / "eval_report.json"
    p.write_text(json.dumps(report))

    thr = load_reference_thresholds(str(p))
    arr = np.array(id_tokens, dtype=float)
    assert abs(thr["id_split"]["underthinking_threshold"] - float(np.percentile(arr, 10))) < 1e-9
    assert abs(thr["id_split"]["overthinking_threshold"] - float(np.percentile(arr, 75))) < 1e-9
    assert "near_ood" in thr
    assert "capability_floor" not in thr  # too few samples to anchor a threshold


def test_resolve_reference_run_none_and_missing(tmp_path):
    assert _resolve_reference_thresholds(None) == {}
    assert _resolve_reference_thresholds(str(tmp_path / "does-not-exist")) == {}


def test_resolve_reference_run_reads_report(tmp_path):
    report = {"results": {"id_split": {"samples": [{"correct": True, "n_tokens": t} for t in range(10, 101, 10)]}}}
    (tmp_path / "eval_report.json").write_text(json.dumps(report))
    # Accepts a run directory (joins eval_report.json) ...
    thr = _resolve_reference_thresholds(str(tmp_path))
    assert "id_split" in thr and "underthinking_threshold" in thr["id_split"]
    # ... and a direct path to the report file.
    thr2 = _resolve_reference_thresholds(str(tmp_path / "eval_report.json"))
    assert thr2 == thr
