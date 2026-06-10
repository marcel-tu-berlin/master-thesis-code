"""reference_run wired into the trained-vs-trained delta (T2.2).

Previously report.py rediscovered the baseline through a fragile single-e0
heuristic (None when 0 or >1 e0 siblings exist), and vs_reward_baseline carried
no token delta. The precedence is now baseline_id > eval.reference_run >
heuristic, and the trained-vs-trained block reports Δ mean tokens too.
"""
import json
import os

from eval.report import _find_baseline, canonical_baseline_dir, generate_report
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


def test_vs_base_model_reads_canonical_baseline(tmp_path):
    """The before-vs-after-finetune delta is discovered at the canonical
    per-slug path runs/_baselines/<slug>/, not runs/<exp>/baseline/ (Task 18)."""
    runs = tmp_path / "runs"
    slug = "qwen-7b"
    # Base-model assessment for this slug, written where runner now puts it.
    base_dir = canonical_baseline_dir(str(runs), slug)
    os.makedirs(base_dir, exist_ok=True)
    json.dump(
        {
            "experiment_id": f"_baseline-{slug}",
            "model_slug": slug,
            "results": {"id_split": {"accuracy": 0.4, "mean_token_count": 150.0}},
        },
        open(os.path.join(base_dir, "eval_report.json"), "w"),
    )
    run_dir = str(runs / "e9")
    os.makedirs(run_dir)
    cfg = {"experiment_id": "e9", "model": {"slug": slug}, "seed": 42}
    ood = OODResults(id_split=EvalMetrics(accuracy=0.7, mean_token_count=90.0, n_samples=10, n_correct=7))
    rep = generate_report(cfg, ood, run_dir)
    vbm = rep["vs_base_model"]
    assert vbm["model_slug"] == slug
    assert vbm["baseline_accuracy"] == 0.4
    assert abs(vbm["delta_accuracy"] - 0.3) < 1e-9          # 0.7 - 0.4
    assert abs(vbm["delta_mean_tokens"] - (-60.0)) < 1e-9   # 90 - 150


def test_vs_base_model_absent_when_no_canonical_baseline(tmp_path):
    """No baseline at the canonical path -> no vs_base_model block (the old
    per-exp runs/<exp>/baseline/ location is no longer consulted)."""
    runs = tmp_path / "runs"
    # A stale per-experiment baseline dir must NOT be picked up.
    stale = runs / "e9" / "baseline"
    os.makedirs(stale, exist_ok=True)
    json.dump(
        {"model_slug": "qwen-7b", "results": {"id_split": {"accuracy": 0.4, "mean_token_count": 150.0}}},
        open(os.path.join(str(stale), "eval_report.json"), "w"),
    )
    run_dir = str(runs / "e9")
    cfg = {"experiment_id": "e9", "model": {"slug": "qwen-7b"}, "seed": 42}
    ood = OODResults(id_split=EvalMetrics(accuracy=0.7, mean_token_count=90.0, n_samples=10, n_correct=7))
    rep = generate_report(cfg, ood, run_dir)
    assert "vs_base_model" not in rep
