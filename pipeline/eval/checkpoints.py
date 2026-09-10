"""Checkpoint destinations and completion checks. No model imports or writes.

Resolve one explicit checkpoint or the planned thirds. Existing final-only runs
keep their top-level destination. Invalid schedules raise ValueError.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from training.config_schema import resolve_checkpoint_steps


def experiment_root(run_dir: str) -> Path:
    """Locate the owning frozen config for an intermediate evaluation directory."""
    path = Path(run_dir)
    return path.parent.parent if path.parent.name == "checkpoint-evals" else path


def report_step(run_dir: str) -> int | None:
    """Read checkpoint metadata without inventing it for historical reports."""
    path = Path(run_dir) / "eval_report.json"
    return (
        json.loads(path.read_text()).get("checkpoint_step") if path.exists() else None
    )


@dataclass(frozen=True)
class CheckpointEval:
    checkpoint: str | None
    step: int | None
    output_dir: str


def checkpoint_step(
    config: dict, checkpoint: str | None, *, run_dir: str | None = None
) -> int | None:
    """Identify this run's checkpoint step, never infer another run's metadata."""
    if checkpoint is None:
        return 0
    if run_dir is None:
        if not config.get("experiment_id"):
            return None
        run_dir = str(Path("runs") / config["experiment_id"])
    if Path(checkpoint).resolve().parent != Path(run_dir).resolve():
        return None
    name = Path(checkpoint).name
    if name == "checkpoint-final":
        return (config.get("training") or {}).get("max_steps", 500)
    suffix = name.removeprefix("checkpoint-")
    return (
        int(suffix) if name.startswith("checkpoint-") and suffix.isdecimal() else None
    )


def resolve_checkpoint_evals(
    config: dict,
    run_dir: str,
    checkpoint: str | None = None,
    *,
    base_model: bool = False,
    smoke: bool = False,
) -> list[CheckpointEval]:
    """Resolve paths once for train, eval, and batch. Explicit paths stay single."""
    steps = resolve_checkpoint_steps(config, smoke=smoke)
    root = Path(run_dir)
    if base_model:
        if steps:
            raise ValueError(
                "E0 --base-model rejects eval.checkpoint_schedule; evaluate E0 once"
            )
        return [CheckpointEval(None, 0, run_dir)]
    if checkpoint is not None:
        step = checkpoint_step(config, checkpoint, run_dir=run_dir)
        out = root
        if steps:
            # Arbitrary overrides also get their own destination, never the final report.
            out = root / "checkpoint-evals" / Path(checkpoint).name
            if Path(checkpoint).resolve() == (root / "checkpoint-final").resolve():
                out = root
        return [CheckpointEval(checkpoint, step, str(out))]
    if not steps:
        return [
            CheckpointEval(
                str(root / "checkpoint-final"),
                checkpoint_step(
                    config, str(root / "checkpoint-final"), run_dir=run_dir
                ),
                run_dir,
            )
        ]
    return [
        CheckpointEval(
            str(
                root
                / ("checkpoint-final" if step == steps[-1] else f"checkpoint-{step}")
            ),
            step,
            str(
                root
                if step == steps[-1]
                else root / "checkpoint-evals" / f"checkpoint-{step}"
            ),
        )
        for step in steps
    ]


def completed(target: CheckpointEval, *, smoke: bool = False) -> bool:
    """A matching finished report is the commit marker for one checkpoint."""
    path = Path(target.output_dir) / "eval_report.json"
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(report, dict)
        and report.get("status") not in ("error", "skipped")
        and bool(report.get("results"))
        and bool(report.get("smoke", False)) == smoke
        and report.get("checkpoint_step") == target.step
        and report.get("checkpoint") == target.checkpoint
    )


def refuse_existing_outputs(output_dir: str) -> None:
    """Fail rather than overwrite a report or trajectory, including orphan files."""
    root = Path(output_dir)
    existing = [*root.glob("eval_report.*"), *root.glob("episodes_*.jsonl")]
    if existing:
        raise FileExistsError(
            f"Evaluation outputs already exist in {root}: {existing}. Preserve them and use a fresh experiment_id."
        )
