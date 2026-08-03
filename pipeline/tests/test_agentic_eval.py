from eval.metrics import SampleResult, compute_metrics


def test_mean_steps_reported():
    rs = [
        SampleResult(correct=True, n_tokens=10, n_steps=1),
        SampleResult(correct=False, n_tokens=20, n_steps=3),
    ]
    m = compute_metrics(rs)
    assert abs(m.mean_steps - 2.0) < 1e-9


def test_mean_steps_none_for_dataset_eval():
    rs = [
        SampleResult(correct=True, n_tokens=10),
        SampleResult(correct=False, n_tokens=20),
    ]
    assert compute_metrics(rs).mean_steps is None


# --- --base-model must never overwrite a trained run's harvested results ---

def test_base_model_refused_over_a_trained_run(tmp_path):
    from eval.runner import base_model_conflict
    run_dir = tmp_path / "runs" / "e27-browsergym-e1-baseline"
    (run_dir / "checkpoint-final").mkdir(parents=True)
    msg = base_model_conflict(str(run_dir))
    assert msg is not None and "checkpoint" in msg


def test_base_model_allowed_for_a_dedicated_e0_dir(tmp_path):
    from eval.runner import base_model_conflict
    run_dir = tmp_path / "runs" / "e0-browsergym-base"
    run_dir.mkdir(parents=True)
    assert base_model_conflict(str(run_dir)) is None


def test_base_model_allowed_when_run_dir_absent(tmp_path):
    from eval.runner import base_model_conflict
    assert base_model_conflict(str(tmp_path / "runs" / "nope")) is None


# --- --smoke must not overwrite harvested results ---
# A smoke eval writes 4 episodes per split into the same run dir as a real one,
# destroying eval_report.json and every episodes_*.jsonl. The report can be
# snapshotted; the trajectory records cannot be recovered from anything else.

def _real_report(run_dir):
    import json
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_report.json").write_text(
        json.dumps({"experiment_id": "e27", "results": {"held_out": {"n_samples": 100}}}))


def test_smoke_refused_over_a_real_report(tmp_path):
    from eval.runner import smoke_conflict
    run_dir = tmp_path / "runs" / "e27"
    _real_report(run_dir)
    msg = smoke_conflict(str(run_dir))
    assert msg is not None and "episodes" in msg


def test_smoke_allowed_over_a_previous_smoke_report(tmp_path):
    import json
    from eval.runner import smoke_conflict
    run_dir = tmp_path / "runs" / "e27-throwaway"
    run_dir.mkdir(parents=True)
    (run_dir / "eval_report.json").write_text(json.dumps({"smoke": True, "results": {}}))
    assert smoke_conflict(str(run_dir)) is None


def test_smoke_allowed_on_a_fresh_run_dir(tmp_path):
    from eval.runner import smoke_conflict
    assert smoke_conflict(str(tmp_path / "runs" / "nope")) is None
