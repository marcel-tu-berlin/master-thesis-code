"""reference_run wired into the trained-vs-trained delta (T2.2).

Previously report.py rediscovered the baseline through a fragile single-e0
heuristic (None when 0 or >1 e0 siblings exist), and vs_reward_baseline carried
no token delta. The precedence is now baseline_id > eval.reference_run >
heuristic, and the trained-vs-trained block reports Δ mean tokens too.
"""
import json
import os

from eval.report import _find_baseline, generate_report
from eval.metrics import EvalMetrics
from eval.ood_probes import OODResults


def _write_report(d, exp_id, acc, tokens):
    os.makedirs(d, exist_ok=True)
    json.dump(
        {"experiment_id": exp_id, "results": {"id_split": {"accuracy": acc, "mean_token_count": tokens}}},
        open(os.path.join(d, "eval_report.json"), "w"),
    )


def test_baseline_id_wins_over_reference_run(tmp_path):
    runs = tmp_path / "runs"
    _write_report(str(runs / "e0-A"), "e0-A", 0.5, 100)
    _write_report(str(runs / "e0-B"), "e0-B", 0.6, 120)
    run_dir = str(runs / "e9")
    os.makedirs(run_dir)
    cfg = {"baseline_id": "e0-A", "eval": {"reference_run": str(runs / "e0-B")}}
    assert _find_baseline(run_dir, cfg) == os.path.join(str(runs / "e0-A"), "eval_report.json")


def test_reference_run_used_when_no_baseline_id(tmp_path):
    runs = tmp_path / "runs"
    _write_report(str(runs / "e0-B"), "e0-B", 0.6, 120)
    # A second e0 sibling makes the heuristic ambiguous (returns None), so the
    # only way to resolve e0-B is via reference_run — this discriminates.
    _write_report(str(runs / "e0-decoy"), "e0-decoy", 0.4, 90)
    run_dir = str(runs / "e9")
    os.makedirs(run_dir)
    expected = os.path.join(str(runs / "e0-B"), "eval_report.json")
    # Directory form ...
    assert _find_baseline(run_dir, {"eval": {"reference_run": str(runs / "e0-B")}}) == expected
    # ... and a direct .json path.
    assert _find_baseline(run_dir, {"eval": {"reference_run": expected}}) == expected


def test_reference_run_missing_falls_back_to_heuristic(tmp_path):
    runs = tmp_path / "runs"
    _write_report(str(runs / "e0-baseline-math"), "e0-baseline-math", 0.5, 100)
    run_dir = str(runs / "e9")
    os.makedirs(run_dir)
    # reference_run points nowhere -> single-e0 heuristic still resolves it.
    cfg = {"eval": {"reference_run": str(runs / "does-not-exist")}}
    assert _find_baseline(run_dir, cfg) == os.path.join(str(runs / "e0-baseline-math"), "eval_report.json")


def test_vs_reward_baseline_carries_token_delta(tmp_path):
    runs = tmp_path / "runs"
    _write_report(str(runs / "e0-ref"), "e0-ref", 0.5, 100.0)
    run_dir = str(runs / "e9")
    os.makedirs(run_dir)
    cfg = {
        "experiment_id": "e9",
        "model": {"slug": "qwen-7b"},
        "seed": 42,
        "eval": {"reference_run": str(runs / "e0-ref")},
    }
    ood = OODResults(id_split=EvalMetrics(accuracy=0.7, mean_token_count=80.0, n_samples=10, n_correct=7))
    rep = generate_report(cfg, ood, run_dir)
    vrb = rep["vs_reward_baseline"]
    assert vrb["baseline_id"] == "e0-ref"
    assert abs(vrb["delta_accuracy"] - 0.2) < 1e-9
    assert abs(vrb["delta_mean_tokens"] - (-20.0)) < 1e-9   # 80 - 100, negative = more efficient
