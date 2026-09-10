"""CPU regression checks for planned checkpoint observations and safe resume."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eval import runner
from eval.agentic_eval import _build_report, _report_md
from eval.checkpoints import completed, resolve_checkpoint_evals
from eval.metrics import SampleResult, compute_metrics
from training import batch
from training.config_schema import resolve_checkpoint_steps, validate_config


def config(steps=150):
    return {
        "experiment_id": "future-s42",
        "seed": 42,
        "model": {"slug": "qwen3-1.7b"},
        "training": {"env": "reasoning_gym", "max_steps": steps},
        "eval": {"checkpoint_schedule": "thirds", "reference_report": "reference.json"},
    }


@pytest.mark.parametrize(
    "steps,expected",
    [(150, [50, 100, 150]), (300, [100, 200, 300]), (600, [200, 400, 600])],
)
def test_thirds(steps, expected):
    cfg = config(steps)
    before = copy.deepcopy(cfg)
    validate_config(cfg)
    assert resolve_checkpoint_steps(cfg) == expected
    assert cfg == before


@pytest.mark.parametrize("steps", [151, 0, -3, 150.5, True, None])
def test_invalid_steps(steps):
    with pytest.raises(ValueError, match="divisible by 3"):
        validate_config(config(steps))


@pytest.mark.parametrize("schedule", ["quarters", "", None, False, [1, 2, 3]])
def test_unknown_schedule(schedule):
    cfg = config()
    cfg["eval"]["checkpoint_schedule"] = schedule
    with pytest.raises(ValueError, match="checkpoint_schedule"):
        validate_config(cfg)


def test_validation_requires_reference_before_training():
    cfg = config()
    del cfg["eval"]["reference_report"]
    with pytest.raises(ValueError, match="reference_report"):
        validate_config(cfg)


def test_save_steps_conflict_and_matching():
    cfg = config()
    cfg["training"]["save_steps"] = 100
    with pytest.raises(ValueError, match="save_steps conflicts"):
        validate_config(cfg)
    with pytest.raises(ValueError, match="save_steps conflicts"):
        resolve_checkpoint_steps(cfg, smoke=True)
    cfg["training"]["save_steps"] = 50
    validate_config(cfg)
    assert resolve_checkpoint_steps(cfg) == [50, 100, 150]


def test_absence_preserves_final_only():
    cfg = config()
    cfg["eval"] = {}
    cfg["training"]["save_steps"] = 73
    assert resolve_checkpoint_steps(cfg) == []
    assert resolve_checkpoint_steps(cfg, smoke=True) == []
    targets = resolve_checkpoint_evals(cfg, "runs/x")
    assert [(t.checkpoint, t.output_dir) for t in targets] == [
        ("runs/x/checkpoint-final", "runs/x")
    ]
    assert cfg["training"]["save_steps"] == 73


def test_paths_smoke_and_e0():
    cfg = config()
    targets = resolve_checkpoint_evals(cfg, "runs/x")
    assert [t.step for t in targets] == [50, 100, 150]
    assert [t.checkpoint for t in targets] == [
        "runs/x/checkpoint-50",
        "runs/x/checkpoint-100",
        "runs/x/checkpoint-final",
    ]
    assert [t.output_dir for t in targets] == [
        "runs/x/checkpoint-evals/checkpoint-50",
        "runs/x/checkpoint-evals/checkpoint-100",
        "runs/x",
    ]
    assert [t.step for t in resolve_checkpoint_evals(cfg, "runs/x", smoke=True)] == [
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError, match="E0"):
        resolve_checkpoint_evals(cfg, "runs/x", base_model=True)
    cfg["eval"] = {}
    assert (
        resolve_checkpoint_evals(cfg, "runs/x", base_model=True)[0].checkpoint is None
    )


@pytest.mark.parametrize("name", ["checkpoint-50", "checkpoint-final"])
def test_external_checkpoint_does_not_claim_run_step(name):
    cfg = config()
    checkpoint = f"runs/other-run/{name}"
    target = resolve_checkpoint_evals(cfg, "runs/future-s42", checkpoint)[0]
    assert target.step is None
    assert _build_report(cfg, checkpoint, {})["checkpoint_step"] is None


def write_result(target, cfg):
    root = Path(target.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics = {"agentic": compute_metrics([SampleResult(True, 12)])}
    report = _build_report(cfg, target.checkpoint, metrics)
    (root / "eval_report.json").write_text(json.dumps(report))
    (root / "eval_report.md").write_text(
        _report_md(cfg["experiment_id"], metrics, report["checkpoint_step"])
    )
    (root / "episodes_agentic.jsonl").write_text(
        '{"seed":42100000,"correct":true,"n_tokens":12}\n'
    )
    (root / "env_stamp.json").write_text('{"eval":{"test":true}}')


@pytest.fixture
def scheduled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config()
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "results": {
                    "agentic": {"samples": [{"n_tokens": n} for n in [10, 20, 30, 40]]}
                }
            }
        )
    )
    cfg["eval"]["reference_report"] = str(reference)
    run_dir = "runs/future-s42"
    calls = []
    preflight = []
    monkeypatch.setattr(
        runner,
        "preflight_checkpoints",
        lambda c, ts: preflight.append([t.step for t in ts]),
    )

    def worker(cmd, check):
        assert check
        worker_cfg = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
        checkpoint = cmd[cmd.index("--checkpoint") + 1]
        stage = cmd[cmd.index("--output-dir") + 1]
        target = resolve_checkpoint_evals(worker_cfg, run_dir, checkpoint)[0]
        calls.append(target.step)
        write_result(type(target)(checkpoint, target.step, stage), worker_cfg)

    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=worker))
    return cfg, run_dir, calls, preflight


def test_schedule_reserves_outputs_and_releases_lock(scheduled, monkeypatch):
    cfg, run_dir, calls, _ = scheduled
    worker = runner.subprocess.run

    def concurrent_worker(cmd, check):
        with pytest.raises(BlockingIOError):
            runner.run_checkpoint_schedule(cfg, run_dir)
        worker(cmd, check)

    monkeypatch.setattr(runner.subprocess, "run", concurrent_worker)
    runner.run_checkpoint_schedule(cfg, run_dir)
    runner.run_checkpoint_schedule(cfg, run_dir)
    assert calls == [50, 100, 150]


def test_resume_only_missing_and_preserve_final_layout(scheduled):
    cfg, run_dir, calls, preflight = scheduled
    targets = resolve_checkpoint_evals(cfg, run_dir)
    write_result(targets[0], cfg)
    before = {p: p.read_bytes() for p in Path(targets[0].output_dir).iterdir()}
    runner.run_checkpoint_schedule(cfg, run_dir)
    assert calls == [100, 150]
    assert preflight == [[50, 100, 150]]
    assert all(completed(t) for t in targets)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not (Path(run_dir) / "checkpoint-evals/checkpoint-150").exists()
    runner.run_checkpoint_schedule(cfg, run_dir)
    assert calls == [100, 150]


@pytest.mark.parametrize(
    "artifact", ["eval_report.json", "eval_report.md", "episodes_agentic.jsonl"]
)
def test_orphan_outputs_never_overwritten(scheduled, artifact):
    cfg, run_dir, calls, _ = scheduled
    root = Path(resolve_checkpoint_evals(cfg, run_dir)[1].output_dir)
    root.mkdir(parents=True)
    path = root / artifact
    path.write_text("original")
    with pytest.raises(FileExistsError):
        runner.run_checkpoint_schedule(cfg, run_dir)
    assert path.read_text() == "original"
    assert not calls


def test_failed_worker_publishes_nothing_and_resumes(scheduled, monkeypatch):
    cfg, run_dir, calls, _ = scheduled
    real_worker = runner.subprocess.run

    def fail(cmd, check):
        real_worker(cmd, check)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_checkpoint_schedule(cfg, run_dir)
    assert not list(Path(run_dir).rglob("episodes_*.jsonl"))
    assert not (Path(run_dir) / "eval_report.json").exists()
    monkeypatch.setattr(runner.subprocess, "run", real_worker)
    runner.run_checkpoint_schedule(cfg, run_dir)
    assert calls == [50, 50, 100, 150]


def test_protocol_changes_rejected(scheduled):
    cfg, run_dir, _, _ = scheduled
    runner.run_checkpoint_schedule(cfg, run_dir)
    cfg["seed"] = 43
    with pytest.raises(ValueError, match="protocol changed"):
        runner.run_checkpoint_schedule(cfg, run_dir)


def test_missing_reference_rejected_before_worker(scheduled):
    cfg, run_dir, calls, _ = scheduled
    del cfg["eval"]["reference_report"]
    with pytest.raises(ValueError, match="reference_report"):
        runner.run_checkpoint_schedule(cfg, run_dir)
    assert not calls


def test_unloadable_later_checkpoint_fails_before_any_episodes(scheduled, monkeypatch):
    cfg, run_dir, calls, _ = scheduled

    def fail(c, targets):
        assert [t.step for t in targets] == [50, 100, 150]
        raise RuntimeError("checkpoint-100 adapter shape mismatch")

    monkeypatch.setattr(runner, "preflight_checkpoints", fail)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        runner.run_checkpoint_schedule(cfg, run_dir)
    assert not calls


def test_reference_change_rejected_on_resume(scheduled):
    cfg, run_dir, calls, _ = scheduled
    runner.run_checkpoint_schedule(cfg, run_dir)
    reference = Path(cfg["eval"]["reference_report"])
    report = json.loads(reference.read_text())
    report["results"]["agentic"]["samples"][0]["n_tokens"] = 999
    reference.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="protocol changed"):
        runner.run_checkpoint_schedule(cfg, run_dir)
    assert calls == [50, 100, 150]


def test_smoke_reports_cannot_complete_real_eval_or_replace_it(scheduled):
    cfg, run_dir, calls, _ = scheduled
    runner.run_checkpoint_schedule(cfg, run_dir)
    cfg["_smoke"] = True
    with pytest.raises(ValueError, match="protocol changed"):
        runner.run_checkpoint_schedule(cfg, run_dir)
    assert calls == [50, 100, 150]
    targets = resolve_checkpoint_evals(cfg, run_dir)
    assert not completed(targets[0], smoke=True)


def test_smoke_thirds_execute_every_checkpoint(scheduled):
    cfg, run_dir, calls, _ = scheduled
    cfg["training"].update(max_steps=3, save_steps=1)
    cfg["_smoke"] = True
    runner.run_checkpoint_schedule(cfg, run_dir)
    assert calls == [1, 2, 3]
    targets = resolve_checkpoint_evals(cfg, run_dir)
    assert all(completed(t, smoke=True) for t in targets)
    assert not any(completed(t) for t in targets)


def test_batch_requires_all_three_and_no_stub(scheduled, monkeypatch):
    cfg, run_dir, _, _ = scheduled
    path = Path("config.yaml")
    path.write_text(yaml.safe_dump(cfg))
    targets = resolve_checkpoint_evals(cfg, run_dir)
    write_result(targets[-1], cfg)
    invoked = []
    monkeypatch.setattr(
        batch,
        "_run_phase",
        lambda *args: invoked.append(args) or (batch.STATUS_OK, 1, 0),
    )
    args = (str(path), cfg["experiment_id"], False, False, 0, False)
    assert batch._run_eval_phase(*args).status == batch.STATUS_OK
    write_result(targets[0], cfg)
    write_result(targets[1], cfg)
    assert batch._run_eval_phase(*args).status == batch.STATUS_SKIP
    assert len(invoked) == 1
    Path(run_dir, "eval_report.json").unlink()
    assert not batch._write_eval_stub(str(path), "error")
    assert not Path(run_dir, "eval_report.json").exists()


def test_batch_smoke_uses_cli_even_when_real_reports_match(scheduled, monkeypatch):
    cfg, run_dir, _, _ = scheduled
    cfg["training"]["max_steps"] = 3
    for target in resolve_checkpoint_evals(cfg, run_dir):
        write_result(target, cfg)
    path = Path("config.yaml")
    path.write_text(yaml.safe_dump(cfg))
    invoked = []
    monkeypatch.setattr(
        batch,
        "_run_phase",
        lambda *args: invoked.append(args) or (batch.STATUS_OK, 1, 0),
    )
    batch._run_eval_phase(str(path), cfg["experiment_id"], True, False, 0, False)
    assert len(invoked) == 1
    assert "--smoke" in invoked[0][0]


def test_explicit_cli_checkpoint_is_single(scheduled, monkeypatch):
    cfg, run_dir, calls, _ = scheduled
    path = Path("config.yaml")
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--config", str(path), "--checkpoint", run_dir + "/checkpoint-100"],
    )
    runner.main()
    assert calls == [100]
    assert not Path(run_dir, "eval_report.json").exists()


def test_report_labels_and_analysis(scheduled):
    from eval import paired, plots

    cfg, run_dir, _, _ = scheduled
    runner.run_checkpoint_schedule(cfg, run_dir)
    Path(run_dir, "config.yaml").write_text(yaml.safe_dump(cfg))
    targets = resolve_checkpoint_evals(cfg, run_dir)
    for t in targets:
        report = json.loads(Path(t.output_dir, "eval_report.json").read_text())
        assert report["checkpoint_step"] == t.step
        assert report["checkpoint"] == t.checkpoint
        assert (
            f"checkpoint step {t.step}"
            in Path(t.output_dir, "eval_report.md").read_text()
        )
        assert paired.load_config(t.output_dir) == cfg
        assert paired._short(t.output_dir) == f"future-s42 step {t.step}"
        loaded = plots.load_report(t.output_dir)
        assert loaded["checkpoint_step"] == t.step
        assert (
            plots._short(loaded["experiment_id"], loaded["checkpoint_step"])
            == f"future-s42\nstep {t.step}"
        )
    with pytest.raises(ValueError, match="one checkpoint per training seed"):
        paired.dose_rows(run_dir, [t.output_dir for t in targets[:-1]])


def test_missing_checkpoint_fails_without_model_imports():
    cfg = config()
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        runner.preflight_checkpoints(cfg, resolve_checkpoint_evals(cfg, "/missing/run"))


def test_smoke_protects_orphan_trajectories_before_loading_model(tmp_path):
    from eval.agentic_eval import run_agentic_eval

    trajectory = tmp_path / "episodes_agentic.jsonl"
    trajectory.write_text("original trajectory")
    with pytest.raises(FileExistsError, match="overwrite"):
        run_agentic_eval({"_smoke": True}, "checkpoint-final", None, str(tmp_path))
    assert trajectory.read_text() == "original trajectory"


def test_internal_smoke_worker_stays_single(scheduled, monkeypatch):
    import domains
    from eval import agentic_eval

    cfg, run_dir, _, _ = scheduled
    cfg["training"].update(max_steps=3, save_steps=1)
    cfg["_smoke"] = True
    marker = Path(run_dir, "checkpoint-final", ".smoke")
    marker.parent.mkdir(parents=True)
    marker.touch()
    path = Path("worker.yaml")
    path.write_text(yaml.safe_dump(cfg))
    calls = []
    monkeypatch.setattr(domains, "build_domain", lambda c: "domain")
    monkeypatch.setattr(
        agentic_eval, "run_agentic_eval", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--config",
            str(path),
            "--checkpoint",
            run_dir + "/checkpoint-1",
            "--output-dir",
            "stage",
        ],
    )
    runner.main()
    assert len(calls) == 1
    assert calls[0][0]["_smoke"] is True
    assert calls[0][3] == "stage"


def test_cli_without_schedule_keeps_top_level(scheduled, monkeypatch):
    import domains
    from eval import agentic_eval

    cfg, run_dir, _, _ = scheduled
    del cfg["eval"]["checkpoint_schedule"]
    path = Path("plain.yaml")
    path.write_text(yaml.safe_dump(cfg))
    calls = []
    monkeypatch.setattr(domains, "build_domain", lambda c: "domain")
    monkeypatch.setattr(
        agentic_eval, "run_agentic_eval", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--config", str(path)])
    runner.main()
    assert len(calls) == 1
    assert calls[0][1] == run_dir + "/checkpoint-final"
    assert calls[0][3] == run_dir


def test_training_freezes_resolved_saves_and_smoke(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TrainerCallback=object, set_seed=lambda s: None),
    )
    monkeypatch.setitem(
        sys.modules, "training.grpo_runner", SimpleNamespace(GRPORunner=object)
    )
    source = Path(__file__).resolve().parents[1] / "training/train.py"
    spec = importlib.util.spec_from_file_location("_train_under_test", source)
    assert spec is not None and spec.loader is not None
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    cfg = config()
    smoke = train.apply_smoke_overrides(copy.deepcopy(cfg))
    assert resolve_checkpoint_steps(smoke) == [1, 2, 3]
    assert smoke["training"]["save_steps"] == 1
    plain = copy.deepcopy(cfg)
    plain["eval"] = {}
    assert train.apply_smoke_overrides(plain)["training"]["save_steps"] == 3

    class FakeRunner:
        def __init__(self, c):
            assert c["training"]["save_steps"] == 50

        def train(self, *args, **kwargs):
            pass

        def save_lora(self, path):
            Path(path).mkdir()

    monkeypatch.setattr(train, "GRPORunner", FakeRunner)
    monkeypatch.setattr(
        train,
        "build_domain",
        lambda c: SimpleNamespace(
            build_seed_dataset=lambda *a, **k: [], make_env_factory=lambda *a: None
        ),
    )
    monkeypatch.setattr(train, "build_reward_components", lambda *a: [(object(), 1)])
    monkeypatch.setattr(train, "build_composer", lambda *a: object())
    monkeypatch.setattr(
        train,
        "build_env_server",
        lambda *a, **k: SimpleNamespace(
            repo_envs_path="unused", base_url="unused", max_concurrent=32
        ),
    )
    monkeypatch.setattr(train, "write_env_stamp", lambda *a: None)
    path = Path("train.yaml")
    path.write_text(yaml.safe_dump(cfg))
    dispatched = []

    class EvalDispatch(Exception):
        pass

    def replace_process(executable, argv):
        dispatched.append((executable, argv))
        raise EvalDispatch

    monkeypatch.setattr(train.os, "execv", replace_process)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--eval"])
    with pytest.raises(EvalDispatch):
        train.main()
    assert dispatched[0][1][1:] == [
        "-m",
        "eval.runner",
        "--config",
        "runs/future-s42/config.yaml",
    ]
    frozen = yaml.safe_load(Path("runs/future-s42/config.yaml").read_text())
    assert frozen["training"]["save_steps"] == 50
