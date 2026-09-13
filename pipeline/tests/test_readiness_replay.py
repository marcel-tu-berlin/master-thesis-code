import json
from pathlib import Path

import pytest
from probes import readiness, readiness_replay


def test_legacy_capture_accepts_only_assessor_changes(monkeypatch, tmp_path):
    key = "pipeline/probes/readiness.py"
    current = readiness.source_hashes()
    recorded = dict(current, **{key: readiness_replay.LEGACY_READINESS_SHA})
    recorded.pop("pipeline/probes/readiness_replay.py")
    approved = current["pipeline/probes/readiness_replay.py"]
    assert readiness_replay.capture_sources_match(current, current, approved)
    unreviewed = dict(current, **{"pipeline/probes/readiness_replay.py": "changed"})
    assert not readiness_replay.capture_sources_match(unreviewed, unreviewed, approved)
    assert readiness_replay.capture_sources_match(recorded, current, approved)
    assert not readiness_replay.capture_sources_match(
        recorded,
        dict(current, **{"pipeline/probes/readiness_replay.py": "changed"}),
        approved,
    )
    assert not readiness_replay.capture_sources_match(
        recorded, dict(current, **{"pipeline/training/train.py": "changed"}), approved
    )
    source = Path(readiness.__file__).read_text()
    changed = tmp_path / "readiness.py"
    changed.write_text(source.replace("CAPTURE_SCHEMA = 1", "CAPTURE_SCHEMA = 2"))
    monkeypatch.setattr(readiness, "__file__", str(changed))
    assert not readiness_replay.capture_sources_match(
        recorded, dict(current, **{key: readiness._file_hash(changed)}), approved
    )


@pytest.mark.parametrize(
    "failure_type",
    [
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        OSError,
        IndexError,
        OverflowError,
        AttributeError,
        AssertionError,
        Exception,
    ],
)
def test_missing_active_e3_evidence_cannot_pass(monkeypatch, tmp_path, failure_type):
    """The controller must run the native replay even after ordinary checks pass."""
    config = readiness._load_config(Path("configs/readiness/g3-e3.yaml"))
    capture = tmp_path / "readiness"
    capture.mkdir()
    stack = {"packages": {"trl": "1.6.0"}}
    current = readiness.source_hashes()
    records = [
        {
            "seed": i // 8,
            "prompt": f"group {i // 8}",
            "environment": {"reward": 0},
            "raw_rewards": {"env_reward": 0, "token_length": 0, "non_termination": 0},
            "composed_reward": 0,
            "completion": [{"role": "tool"}],
        }
        for i in range(32)
    ]
    prepared = {
        key: [[1]] * 32
        for key in (
            "completion_ids",
            "completion_mask",
            "tool_mask",
            "old_per_token_logps",
            "sampling_per_token_logps",
            "importance_sampling_ratio",
        )
    }
    prepared["advantages"] = [0] * 32
    settings = {
        "trl_version": "1.6.0",
        "bf16": True,
        "fp16": False,
        "loss_type": "dapo",
        "scale_rewards": "none",
        "importance_sampling_level": "token",
        "gradient_accumulation_steps": 32,
        "per_device_train_batch_size": 1,
        "num_generations": 8,
        "max_completion_length": 4096,
        "cuda_device": "test",
        "model_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["revision"],
    }
    files = {
        "metadata.json": {
            "status": "complete",
            "config_sha256": readiness._canonical_hash(config),
            "source_hashes": current,
            "stack_sha256": readiness._canonical_hash(stack),
            "cuda_visible_devices": "1",
        },
        "reward_batch.json": {"records": records},
        "rollout_batch.json": {"prepared": prepared},
        "trainer_settings.json": settings,
        "parameters.json": {
            "before": {"finite": True, "sha256": "before"},
            "after": {"finite": True, "sha256": "after"},
        },
    }
    for name, value in files.items():
        (capture / name).write_text(json.dumps(value))
    (tmp_path / "env_stamp.json").write_text(json.dumps({"train": stack}))
    (capture / "loss_shards.jsonl").write_text('{"loss": 0}\n' * 32)
    (capture / "gradients.jsonl").write_text('{"finite": true, "l2_norm": 1}\n')
    (capture / "step_log.jsonl").write_text("")
    replay_path = capture / "integration_replay.json"
    replay_path.write_text('{"status": "pass"}')
    monkeypatch.setattr(readiness, "_reference_dapo_loss", lambda _: 0)

    def missing_replay(*_args):
        raise failure_type("active E3 replay unavailable")

    monkeypatch.setattr(readiness_replay, "run_replay", missing_replay)
    if failure_type is OSError:
        monkeypatch.setattr(readiness_replay, "run_replay", lambda *_args: {})
        monkeypatch.setattr(readiness, "_atomic_json", missing_replay)
    assert readiness._assess_capture(tmp_path, config, current, stack) == [
        "installed trainer integration replay failed: "
        f"{failure_type.__name__}: {failure_type('active E3 replay unavailable')}"
    ]
    assert not replay_path.exists()


@pytest.mark.parametrize("mask", [[], [[]], [[1]], [1]])
def test_replay_rejects_misaligned_token_arrays(mask):
    with pytest.raises(ValueError, match="replay"):
        readiness_replay._validate_token_arrays(
            {"logps": [[-1.0, -2.0]], "mask": mask}, ("logps", "mask"), 1
        )


def test_native_scripted_e3_detects_inert_penalty_and_group_scaling(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    from training.rewards.non_termination import NonTerminationPenalty

    settings = {
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "loss_type": "dapo",
        "scale_rewards": "none",
        "importance_sampling_level": "token",
    }
    trainer = readiness_replay._trainer(settings)
    result = readiness_replay._scripted_e3(trainer)
    assert len(result) == 3
    trainer.scale_rewards = "group"
    with pytest.raises(ValueError, match="advantage mismatch"):
        readiness_replay._scripted_e3(trainer)
    trainer.scale_rewards = "none"
    monkeypatch.setattr(
        NonTerminationPenalty,
        "__call__",
        lambda self, prompts, completions, **kwargs: [0.0] * len(completions),
    )
    with pytest.raises(ValueError, match="advantage mismatch"):
        readiness_replay._scripted_e3(trainer)
