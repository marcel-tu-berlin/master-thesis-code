"""T0.5: a failed / partial / skipped eval must still leave a discoverable
eval_report.json so the run shows up in auto-compare instead of vanishing.

Two layers: runner.py writes a status:'error' stub on a crash mid-eval; batch.py
backstops the cases runner never reached (eval skipped, or a hard kill).
"""
import json

import yaml

from eval.runner import _write_stub_report
from training.batch import _write_eval_stub


def test_runner_stub_has_status_and_identity(tmp_path):
    cfg = {
        "experiment_id": "e9",
        "model": {"slug": "qwen-7b"},
        "seed": 44,
        "rewards": {"compose_method": "naive_sum"},
    }
    _write_stub_report(cfg, str(tmp_path), status="error", error="RuntimeError: oom")
    out = json.loads((tmp_path / "eval_report.json").read_text())
    assert out["status"] == "error"
    assert out["experiment_id"] == "e9"
    assert out["model_slug"] == "qwen-7b"
    assert out["seed"] == 44
    assert out["compose_method"] == "naive_sum"
    assert out["results"] == {}
    assert "oom" in out["error"]


def test_batch_stub_written_when_absent_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # batch writes to runs/<exp_id>/ relative to cwd
    cfg_path = tmp_path / "e0-s42.yaml"
    cfg_path.write_text(yaml.dump(
        {"experiment_id": "e0-s42", "model": {"slug": "qwen-7b"}, "seed": 42}
    ))

    assert _write_eval_stub(str(cfg_path), status="skipped", note="train failed") is True
    rpt = tmp_path / "runs" / "e0-s42" / "eval_report.json"
    out = json.loads(rpt.read_text())
    assert out["status"] == "skipped" and out["note"] == "train failed"
    assert out["experiment_id"] == "e0-s42" and out["seed"] == 42

    # Must never clobber an existing report (a real one, or runner's own stub).
    rpt.write_text(json.dumps({"status": "ok", "results": {"id_split": {"accuracy": 1.0}}}))
    assert _write_eval_stub(str(cfg_path), status="error", note="late") is False
    assert json.loads(rpt.read_text())["status"] == "ok"  # untouched
