import json
import math
from pathlib import Path

import yaml
from probes import readiness

CONFIGS = [
    Path("configs/readiness/g3-e1.yaml"),
    Path("configs/readiness/g3-e2.yaml"),
    Path("configs/readiness/g3-e3.yaml"),
]


def test_readiness_bundle_admits_only_e1_first(monkeypatch, tmp_path):
    monkeypatch.setattr(
        readiness,
        "_local_gate",
        lambda: {"status": "pass", "returncode": 0, "output_tail": []},
    )
    report = readiness.build_report(CONFIGS, tmp_path)
    assert report["contracts"][0]["status"] == "pass"
    assert report["status"] == "not_tested"
    assert report["next_phase"] == "run_e1:readiness-g3-e1-s4001"
    assert [
        item["condition"] for item in report["contracts"][0]["evidence"]["configs"]
    ] == ["E1", "E2", "E3"]


def test_reference_dapo_loss_uses_tool_mask_and_full_batch_normalizer():
    record = {
        "inputs": {
            "advantages": [2.0],
            "completion_mask": [[1, 1, 1]],
            "tool_mask": [[1, 1, 0]],
            "old_per_token_logps": [[-1.0, -2.0, -3.0]],
            "importance_sampling_ratio": [[1.0, 0.5, 1.0]],
            "num_items_in_batch": 10,
        },
        "per_token_logps": [[-1.0, -2.0, -3.0]],
        "settings": {
            "loss_type": "dapo",
            "importance_sampling_level": "token",
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "beta": 0.0,
        },
    }
    assert math.isclose(readiness._reference_dapo_loss(record), -0.3)


def test_assess_run_ignores_final_training_summary(monkeypatch, tmp_path):
    config = readiness._load_config(CONFIGS[0])
    run_dir = tmp_path / config["experiment_id"]
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "train_log.json").write_text(
        json.dumps(
            [{"step": step, "loss": 0.1, "grad_norm": 0.2} for step in range(1, 4)]
            + [{"step": 3, "train_loss": 0.1, "train_runtime": 1.0}]
        )
    )
    (run_dir / "checkpoint-final").mkdir()
    (run_dir / "eval_report.json").write_text(
        json.dumps({"results": {"agentic": {"n_samples": 4}}})
    )
    episode = {"initial_observation": "task", "turns": [{"tool_results": []}]}
    (run_dir / "episodes_agentic.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for _ in range(4))
    )
    monkeypatch.setattr(readiness, "_assess_capture", lambda *_args: [])

    assert readiness._assess_run(run_dir, config, {}, {}) == []


def test_assess_capture_rejects_stale_runtime_stack(tmp_path):
    config = readiness._load_config(CONFIGS[0])
    run_dir = tmp_path / config["experiment_id"]
    capture = run_dir / readiness.CAPTURE_DIR
    capture.mkdir(parents=True)
    old_stack = {"packages": {"trl": "1.6.0"}}
    metadata = {
        "status": "complete",
        "config_sha256": readiness._canonical_hash(config),
        "source_hashes": {},
        "stack_sha256": readiness._canonical_hash(old_stack),
        "cuda_visible_devices": "1",
    }
    files = {
        "metadata.json": metadata,
        "reward_batch.json": {"records": []},
        "rollout_batch.json": {"prepared": {}},
        "parameters.json": {},
        "trainer_settings.json": {},
    }
    for name, value in files.items():
        (capture / name).write_text(json.dumps(value))
    for name in ("loss_shards.jsonl", "gradients.jsonl", "step_log.jsonl"):
        (capture / name).write_text("")
    (run_dir / "env_stamp.json").write_text(json.dumps({"train": old_stack}))

    errors = readiness._assess_capture(
        run_dir, config, {}, {"packages": {"trl": "changed"}}
    )

    assert "capture stack is stale relative to the current runtime" in errors
