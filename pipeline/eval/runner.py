"""Agentic eval entry point (`python -m eval.runner --config ...`).

Thin wrapper: loads the config, resolves the checkpoint, and dispatches to
eval.agentic_eval.run_agentic_eval (held-out OpenEnv episodes). Dataset-mode
eval was removed with the agentic-only migration.
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import yaml

from eval.checkpoints import (
    completed,
    refuse_existing_outputs,
    resolve_checkpoint_evals,
)
from training.config_schema import resolve_checkpoint_steps


def preflight_checkpoints(config: dict, targets) -> None:
    """Load every requested adapter on a meta base before any episodes run.

    Adapter tensors load on CPU. Invalid configs, missing tensors and shape
    mismatches fail here; no GPU model or environment is started.
    """
    for target in targets:
        if target.checkpoint is None or not Path(target.checkpoint).is_dir():
            raise FileNotFoundError(f"Missing checkpoint: {target.checkpoint}")
    from accelerate import init_empty_weights
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    from training.registry import get_model_config

    model_name = get_model_config(config["model"]["slug"])["model_name"]
    model_config = AutoConfig.from_pretrained(model_name)
    for target in targets:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(model_config)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*[Mm]issing adapter keys.*")
            loaded = PeftModel.from_pretrained(
                model,
                target.checkpoint,
                low_cpu_mem_usage=True,
                torch_device="cpu",
                local_files_only=True,
            )
        del loaded, model


def run_checkpoint_schedule(config: dict, run_dir: str, checkpoint=None) -> None:
    """Reserve the run's outputs until all requested reports are published."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Keep the lock file: unlinking it permits two processes to lock different inodes.
    with (root / ".checkpoint-eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_checkpoint_schedule(config, run_dir, checkpoint)


def _run_checkpoint_schedule(config: dict, run_dir: str, checkpoint=None) -> None:
    """Resume planned observations in fresh processes; publish reports last.

    A saved protocol rejects changed settings or reference data on resume.
    Failed workers leave no published trajectories. Existing artifacts are never
    replaced, including smoke artifacts. No checkpoint ranking is performed.
    """
    from eval.agentic_eval import _resolve_splits
    from eval.metrics import load_reference_thresholds

    steps = resolve_checkpoint_steps(config)
    config = {**config, "training": {**config["training"], "save_steps": steps[0]}}
    targets = resolve_checkpoint_evals(config, run_dir, checkpoint)
    reference = (config.get("eval") or {}).get("reference_report")
    if not reference:
        raise ValueError(
            "checkpoint_schedule: thirds requires eval.reference_report to pin thresholds across checkpoints"
        )
    reference_data = json.loads(Path(reference).read_text())
    thresholds = load_reference_thresholds(reference)
    base_n = ((config.get("eval") or {}).get("agentic") or {}).get("n_episodes", 100)
    missing = {s["name"] for s in _resolve_splits(config, base_n)} - thresholds.keys()
    if missing:
        raise ValueError(
            f"Reference report lacks threshold samples for splits: {sorted(missing)}"
        )
    protocol = {"config": config, "reference_report": reference_data}
    protocol_path = Path(run_dir) / "checkpoint-evals" / "eval_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(
            "Checkpoint evaluation protocol changed; use the original config and reference report to resume"
        )
    pending = []
    for target in targets:
        if completed(target, smoke=bool(config.get("_smoke"))):
            print(f"Skip completed checkpoint step {target.step}: {target.output_dir}")
        else:
            refuse_existing_outputs(target.output_dir)
            pending.append(target)
    if not pending:
        return
    # Check all requested checkpoints, including later ones, before a long eval.
    preflight_checkpoints(config, targets)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        with protocol_path.open("x") as f:
            json.dump(protocol, f, indent=2)
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-eval-", dir=run_dir
    ) as temporary:
        temp = Path(temporary)
        ref_path = temp / "reference.json"
        ref_path.write_text(json.dumps(reference_data))
        worker_config = {
            **config,
            "eval": {**config["eval"], "reference_report": str(ref_path)},
        }
        config_path = temp / "config.yaml"
        config_path.write_text(yaml.safe_dump(worker_config))
        for target in pending:
            if target.checkpoint is None:
                raise ValueError("Scheduled evaluation requires a trained checkpoint")
            stage = temp / f"step-{target.step}"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eval.runner",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    target.checkpoint,
                    "--output-dir",
                    str(stage),
                ],
                check=True,
            )
            staged_target = type(target)(target.checkpoint, target.step, str(stage))
            if not completed(staged_target, smoke=bool(config.get("_smoke"))):
                raise RuntimeError(f"Worker did not finish checkpoint {target.step}")
            _publish_evaluation(stage, Path(target.output_dir))


def _publish_evaluation(stage: Path, destination: Path) -> None:
    """Publish exclusive artifacts, with JSON last as the completion marker."""
    refuse_existing_outputs(str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    stamp_path = destination / "env_stamp.json"
    stamp = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
    stamp.update(json.loads((stage / "env_stamp.json").read_text()))
    stamp_path.write_text(json.dumps(stamp, indent=2))
    files = [
        *stage.glob("episodes_*.jsonl"),
        stage / "eval_report.md",
        stage / "eval_report.json",
    ]
    for path in files:
        with path.open("rb") as src, (destination / path.name).open("xb") as dst:
            shutil.copyfileobj(src, dst)


def smoke_conflict(run_dir: str) -> str | None:
    """Error message if a smoke eval would overwrite real results, else None.

    A --smoke eval writes 4 episodes per split into the same run dir as a real
    one, so pointing it at a harvested arm destroys `eval_report.json`,
    `eval_report.md` and every `episodes_*.jsonl` - the trajectory records the
    off-target panel is answered from, which no snapshot of the report alone
    brings back. Same hazard as `--base-model` against a trained run, same
    answer: refuse, and let a throwaway experiment_id take the write.
    """
    from training.batch import _is_real_report

    report_path = Path(run_dir) / "eval_report.json"
    if not _is_real_report(str(report_path)):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {}
        if isinstance(report, dict) and report.get("smoke"):
            return None
        if not list(Path(run_dir).glob("episodes_*.jsonl")):
            return None
    return (
        f"--smoke would overwrite the real eval results in {run_dir!r} "
        "(report and episodes_*.jsonl). Copy the config with a throwaway "
        "experiment_id to smoke-test against this checkpoint."
    )


def base_model_conflict(run_dir: str) -> str | None:
    """Error message if E0 results would overwrite a trained run, else None.

    run_dir keys on experiment_id alone, so `--base-model` against a trained
    arm's config used to overwrite its eval_report.json and episodes_*.jsonl
    with untrained-model numbers - silently, with no backup. E0 gets its own
    config and its own experiment_id (the e0 / e0b pattern).
    """
    if not os.path.isdir(os.path.join(run_dir, "checkpoint-final")):
        return None
    return (
        f"--base-model would write E0 results into {run_dir!r}, which holds a "
        "trained checkpoint. Use a dedicated E0 config with its own "
        "experiment_id (see configs/e0-*.yaml)."
    )


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", help=argparse.SUPPRESS)
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Evaluate the base model with no LoRA adapter (E0)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override eval.max_new_tokens (default: the training completion budget)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Limit eval to 4 episodes per split"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from training.config_schema import validate_config

    validate_config(config)

    if args.smoke and (config.get("eval") or {}).get("checkpoint_schedule"):
        frozen_dir = Path("runs") / config["experiment_id"]
        if (frozen_dir / "checkpoint-final" / ".smoke").exists():
            with (frozen_dir / "config.yaml").open() as f:
                config = yaml.safe_load(f)
    elif (
        not config.get("_smoke")
        and (config.get("eval") or {}).get("checkpoint_schedule")
        and (
            Path("runs") / config["experiment_id"] / "checkpoint-final" / ".smoke"
        ).exists()
    ):
        parser.error(
            "This is a smoke training run; pass --smoke so its reports cannot count as real evaluations"
        )

    # Same guard training gets. An eval-only invocation used to skip validation
    # entirely, so a typo'd key here was caught only if the config had also been
    # through `training.train` at some point.
    validate_config(config)
    checkpoint_steps = resolve_checkpoint_steps(config, smoke=args.smoke)

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)

    if args.smoke:
        conflict = smoke_conflict(run_dir)
        if conflict:
            parser.error(conflict)
        config["_smoke"] = True
        if checkpoint_steps:
            config["training"]["max_steps"] = checkpoint_steps[-1]
            config["training"]["save_steps"] = checkpoint_steps[0]
        print("Smoke mode: eval limited to 4 episodes per split")

    if args.base_model:
        if args.checkpoint:
            parser.error("--base-model and --checkpoint are mutually exclusive")
        target = resolve_checkpoint_evals(config, run_dir, base_model=True)[0]
        conflict = base_model_conflict(run_dir)
        if conflict:
            parser.error(conflict)
        checkpoint = target.checkpoint
    else:
        checkpoint = resolve_checkpoint_evals(config, run_dir, args.checkpoint)[
            0
        ].checkpoint

    if args.max_new_tokens is not None:
        config.setdefault("eval", {})["max_new_tokens"] = args.max_new_tokens

    if checkpoint_steps and not args.output_dir:
        run_checkpoint_schedule(config, run_dir, args.checkpoint)
        return
    if args.output_dir:
        if not args.checkpoint or not checkpoint_steps:
            parser.error("--output-dir requires a scheduled explicit checkpoint")
        run_dir = args.output_dir

    from domains import build_domain

    domain = build_domain(config)

    from eval.agentic_eval import run_agentic_eval

    run_agentic_eval(config, checkpoint, domain, run_dir)


if __name__ == "__main__":
    main()
