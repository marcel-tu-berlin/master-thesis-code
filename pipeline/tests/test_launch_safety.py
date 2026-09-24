"""Exercise real entry points with model loading and server I/O stubbed out."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eval.agentic_eval import run_agentic_eval
from training import batch, env_server, env_stamp


@pytest.fixture
def entrypoints(monkeypatch):
    model = SimpleNamespace(eval=lambda: None)
    loader = SimpleNamespace(from_pretrained=lambda *a, **k: model)
    modules = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_bf16_supported=lambda: True), bfloat16="bf16"
        ),
        "peft": SimpleNamespace(
            LoraConfig=object,
            get_peft_model=object,
            get_peft_model_state_dict=object,
            prepare_model_for_kbit_training=object,
            PeftModel=object,
        ),
        "transformers": SimpleNamespace(
            AutoModelForCausalLM=loader,
            AutoTokenizer=loader,
            BitsAndBytesConfig=object,
            TrainerCallback=object,
            set_seed=lambda seed: None,
        ),
        "trl": SimpleNamespace(GRPOConfig=object, GRPOTrainer=object),
        "trl.chat_template_utils": SimpleNamespace(
            add_response_schema=lambda tokenizer: None, parse_response=object
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def load(name):
        source = Path(__file__).parents[1] / "training" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_launch_{name}", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    runner = load("grpo_runner")
    monkeypatch.setitem(sys.modules, "training.grpo_runner", runner)
    return load("train"), runner


@pytest.mark.parametrize("artifact", ["config.yaml", "checkpoint-final"])
def test_train_refuses_existing_artifacts_before_loading(
    entrypoints, tmp_path, monkeypatch, artifact
):
    train, _ = entrypoints
    monkeypatch.chdir(tmp_path)
    run = Path("runs/safety")
    run.mkdir(parents=True)
    if artifact == "config.yaml":
        (run / artifact).write_text("original frozen configuration\n")
    else:
        (run / artifact).mkdir()
    cfg = {
        "experiment_id": "safety",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"env": "browsergym"},
    }
    Path("launch.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["train", "--config", "launch.yaml"])
    monkeypatch.setattr(
        train, "build_domain", lambda cfg: pytest.fail("model path reached")
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        train.main()
    if artifact == "config.yaml":
        assert (run / artifact).read_text() == "original frozen configuration\n"


@pytest.mark.parametrize("overwrite", [False, True])
def test_metadata_only_directory_and_explicit_overwrite_are_allowed(
    entrypoints, tmp_path, monkeypatch, overwrite
):
    train, _ = entrypoints
    monkeypatch.chdir(tmp_path)
    run = Path("runs/safety")
    run.mkdir(parents=True)
    (run / "launch.json").write_text("{}")
    if overwrite:
        (run / "config.yaml").write_text("old: config")
    cfg = {
        "experiment_id": "safety",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"env": "browsergym"},
    }
    Path("launch.yaml").write_text(yaml.safe_dump(cfg))
    argv = ["train", "--config", "launch.yaml"] + (["--overwrite"] if overwrite else [])
    monkeypatch.setattr(sys, "argv", argv)

    def stop_before_model(cfg):
        raise LookupError("reached model boundary")

    monkeypatch.setattr(train, "build_domain", stop_before_model)
    with pytest.raises(LookupError, match="model boundary"):
        train.main()
    assert yaml.safe_load((run / "config.yaml").read_text()) == cfg
    assert (run / "launch.json").read_text() == "{}"
    cost = [json.loads(line) for line in (run / "costs.jsonl").read_text().splitlines()]
    assert [r["event"] for r in cost] == ["start", "end"]
    assert cost[-1]["status"] == "failed"
    assert cost[-1]["error_type"] == "LookupError"


def test_training_cost_finishes_before_eval_process_replacement(
    entrypoints, tmp_path, monkeypatch
):
    train, _ = entrypoints
    monkeypatch.chdir(tmp_path)
    cfg = {
        "experiment_id": "cost",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"env": "browsergym", "max_steps": 300},
        "eval": {"checkpoint_schedule": "thirds", "reference_report": "e0.json"},
    }
    Path("launch.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["train", "--config", "launch.yaml", "--eval"])
    domain = SimpleNamespace(build_seed_dataset=lambda *a, **k: [])
    model = SimpleNamespace(
        train=lambda *a, **k: None,
        save_lora=lambda path: Path(path).mkdir(),
    )
    monkeypatch.setattr(train, "build_domain", lambda cfg: domain)
    monkeypatch.setattr(train, "GRPORunner", lambda cfg: model)
    monkeypatch.setattr(train, "build_reward_components", lambda *a: [(lambda: 1, 1)])
    monkeypatch.setattr(train, "build_composer", lambda *a: object())
    monkeypatch.setattr(train, "write_env_stamp", lambda *a: None)
    monkeypatch.setattr(
        train,
        "build_env_server",
        lambda *a, **k: SimpleNamespace(
            repo_envs_path="unused",
            base_url="unused",
            max_concurrent=32,
        ),
    )

    def exec_eval(executable, cmd):
        assert "eval.runner" in cmd
        end = json.loads(Path("runs/cost/costs.jsonl").read_text().splitlines()[-1])
        assert end["event"] == "end" and end["status"] == "complete"
        assert end["phase"] == "train"
        raise SystemExit(0)

    monkeypatch.setattr(train.os, "execv", exec_eval)
    with pytest.raises(SystemExit):
        train.main()


@pytest.mark.parametrize("force", [False, True])
def test_batch_retries_never_add_implicit_overwrite(tmp_path, monkeypatch, force):
    monkeypatch.chdir(tmp_path)
    commands = []

    def fail(cmd, log_path):
        commands.append(list(cmd))
        return 1

    monkeypatch.setattr(batch, "_tee_subprocess", fail)
    result = batch._run_train_phase("launch.yaml", "safety", False, force, 1)
    assert result.attempts == 2
    assert commands[0] == commands[1]
    assert ("--overwrite" in commands[0]) is force


@pytest.mark.parametrize("path", ["training", "evaluation", "context"])
@pytest.mark.parametrize("failure", ["wait", "factory"])
def test_acquired_server_is_stopped_on_failure(
    entrypoints, tmp_path, monkeypatch, path, failure
):
    _, module = entrypoints
    events = []
    server = env_server.EnvServerProcess(
        env_module="unused", port=8000, repo_envs_path="unused", max_concurrent=32
    )

    def action(name):
        events.append(name)
        if name == failure:
            raise TimeoutError(name)

    monkeypatch.setattr(server, "start", lambda: action("start"))
    monkeypatch.setattr(server, "wait_until_ready", lambda: action("wait"))
    monkeypatch.setattr(server, "stop", lambda: action("stop"))

    def factory(*args):
        action("factory")
        return lambda: None

    with pytest.raises(TimeoutError, match=failure):
        if path == "context":
            with server:
                factory()
        elif path == "training":
            runner = module.GRPORunner.__new__(module.GRPORunner)
            runner.initial_adapter_info = None
            runner.train([], None, str(tmp_path), server=server, make_factory=factory)
        else:
            monkeypatch.setattr(env_server, "build_env_server", lambda *a, **k: server)
            monkeypatch.setattr(env_stamp, "write_env_stamp", lambda *a: None)
            run_agentic_eval(
                {"model": {"slug": "qwen3-1.7b"}, "training": {"env": "browsergym"}},
                None,
                SimpleNamespace(make_env_factory=factory),
                str(tmp_path),
            )
    assert events[-1] == "stop"
    assert events.count("stop") == 1
