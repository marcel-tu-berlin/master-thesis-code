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
