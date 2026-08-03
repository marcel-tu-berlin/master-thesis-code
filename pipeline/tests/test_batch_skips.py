"""batch skip predicates are content-aware, not existence-only (Task 17)."""
import json
from training.batch import _is_real_report


def test_stub_error_report_not_real(tmp_path):
    p = tmp_path / "r.json"; p.write_text(json.dumps({"status": "error"}))
    assert _is_real_report(str(p)) is False

def test_skipped_report_not_real(tmp_path):
    p = tmp_path / "r.json"; p.write_text(json.dumps({"status": "skipped"}))
    assert _is_real_report(str(p)) is False

def test_smoke_report_not_real(tmp_path):
    p = tmp_path / "r.json"; p.write_text(json.dumps({"smoke": True, "results": {}}))
    assert _is_real_report(str(p)) is False

def test_finished_report_is_real(tmp_path):
    p = tmp_path / "r.json"; p.write_text(json.dumps({"status": "ok", "results": {"id_split": {}}}))
    assert _is_real_report(str(p)) is True

def test_missing_report_not_real(tmp_path):
    assert _is_real_report(str(tmp_path / "nope.json")) is False


# --- the production writer must emit the key _is_real_report reads ---
# The existing tests above hand-build {"smoke": True}; nothing in production ever
# produced it, so a 4-episode --smoke report was classified as finished work and
# the follow-up unattended batch skipped the real 100-episode eval.

def _written_report(smoke: bool, tmp_path):
    from eval.agentic_eval import _build_report
    from eval.metrics import SampleResult, compute_metrics
    cfg = {"experiment_id": "e27-x", "model": {"slug": "qwen3-1.7b"}, "seed": 42}
    if smoke:
        cfg["_smoke"] = True
    metrics = {"held_out": compute_metrics([SampleResult(True, 10, n_steps=1)])}
    p = tmp_path / "eval_report.json"
    p.write_text(json.dumps(_build_report(cfg, "runs/e27-x/checkpoint-final", metrics)))
    return str(p)


def test_production_smoke_report_is_not_real(tmp_path):
    assert _is_real_report(_written_report(True, tmp_path)) is False


def test_production_full_report_is_real(tmp_path):
    assert _is_real_report(_written_report(False, tmp_path)) is True


# --- the .smoke checkpoint marker must be cleared by a later real run ---

def test_smoke_marker_cleared_by_real_train(tmp_path):
    from training.batch import mark_smoke_checkpoint, _is_real_checkpoint
    ckpt = tmp_path / "runs" / "e27-x" / "checkpoint-final"
    ckpt.mkdir(parents=True)

    mark_smoke_checkpoint(str(ckpt), smoke=True)
    assert (ckpt / ".smoke").exists()

    mark_smoke_checkpoint(str(ckpt), smoke=False)
    assert not (ckpt / ".smoke").exists()

    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert _is_real_checkpoint("e27-x") is True
    finally:
        os.chdir(cwd)


def test_clearing_an_absent_marker_is_a_noop(tmp_path):
    from training.batch import mark_smoke_checkpoint
    ckpt = tmp_path / "ckpt"; ckpt.mkdir()
    mark_smoke_checkpoint(str(ckpt), smoke=False)   # must not raise
    assert not (ckpt / ".smoke").exists()
