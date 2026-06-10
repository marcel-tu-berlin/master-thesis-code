"""
Batch experiment runner.

Runs N configs through training, eval, and/or baseline assessment as
subprocesses so each experiment gets a clean Python process (and clean
GPU memory). Designed for overnight queues on a single-GPU box.

Each phase is its own subprocess that mirrors the existing single-run CLIs
(`training.train`, `eval.runner`). Baselines are deduplicated by model slug:
the first config touching a given slug runs the baseline; later configs
sharing that slug get a symlink into the shared baseline directory.

Usage:
    python -m training.batch configs/e0-*.yaml configs/e1-*.yaml --train --eval --baseline
    python -m training.batch configs/e0-baseline-math-qwen-7b.yaml --eval
    python -m training.batch configs/e2-*.yaml --train --eval --smoke --retries 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import yaml

# Allow running as: python -m training.batch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


PHASE_TRAIN = "train"
PHASE_EVAL = "eval"
PHASE_BASELINE = "baseline"

STATUS_OK = "ok"
STATUS_SKIP = "skip"
STATUS_FAIL = "fail"


@dataclass
class PhaseResult:
    status: str            # ok | skip | fail
    duration_s: float = 0.0
    attempts: int = 0
    log_path: Optional[str] = None
    note: str = ""         # e.g. "checkpoint exists", "symlinked from <slug>"


@dataclass
class ExperimentResult:
    config_path: str
    experiment_id: str
    model_slug: str
    phases: dict = field(default_factory=dict)   # phase -> PhaseResult


def _load_config(path: str) -> dict:
    """Load a config YAML. No model load, no validation — the subprocess does that."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Per-seed configs are materialized here (outside the source configs/ tree and
# outside any run dir, so they never trip train.py's clobber guard on
# runs/<exp>/config.yaml). The path is deterministic per exp_id so a resumed
# batch re-points at the same file rather than churning new ones.
_SEED_CONFIG_DIR = os.path.join("runs", "_seed_configs")


def _materialize_seed_config(base_cfg: dict, seed: int, exp_id: str) -> str:
    """Freeze a per-seed variant of `base_cfg` and return its path.

    Subprocess CLIs (training.train / eval.runner) only accept --config, so a
    seed / experiment_id override has to travel through a real config file
    rather than argv. Only the two top-level scalars change; the run dir keys
    on experiment_id, so a distinct suffixed id gives each seed its own dir.
    """
    cfg = dict(base_cfg)          # shallow copy: only top-level scalars change
    cfg["seed"] = int(seed)
    cfg["experiment_id"] = exp_id
    os.makedirs(_SEED_CONFIG_DIR, exist_ok=True)
    out = os.path.join(_SEED_CONFIG_DIR, f"{exp_id}.yaml")
    with open(out, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    return out


def _expand_configs(patterns: list[str]) -> list[str]:
    """Expand a mix of explicit paths and globs; dedupe; preserve sorted order."""
    seen: set[str] = set()
    out: list[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        if not matches:
            print(f"Warning: no matches for {p!r}")
            continue
        for m in matches:
            ap = os.path.abspath(m)
            if ap in seen:
                continue
            if not os.path.isfile(ap):
                print(f"Warning: not a file: {m!r}")
                continue
            seen.add(ap)
            out.append(m)
    return sorted(out)


def _tee_subprocess(cmd: list[str], log_path: str) -> int:
    """Run `cmd`, streaming combined stdout/stderr to log file and parent stdout.

    Returns the subprocess exit code. The log file is overwritten each call.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    header = f"$ {' '.join(cmd)}\n"
    print(header, end="", flush=True)
    with open(log_path, "w") as log:
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def _run_phase(cmd: list[str], log_path: str, retries: int) -> tuple[str, int, float]:
    """Run `cmd` with retry. Returns (status, attempts, duration_s)."""
    attempts = 0
    t0 = time.time()
    while attempts <= retries:
        attempts += 1
        rc = _tee_subprocess(cmd, log_path)
        if rc == 0:
            return STATUS_OK, attempts, time.time() - t0
        print(f"⚠  Exit {rc} on attempt {attempts}/{retries + 1}", flush=True)
    return STATUS_FAIL, attempts, time.time() - t0


def _run_dir(exp_id: str) -> str:
    return os.path.join("runs", exp_id)


def _checkpoint_exists(exp_id: str) -> bool:
    return os.path.isdir(os.path.join(_run_dir(exp_id), "checkpoint-final"))


def _eval_report_exists(exp_id: str) -> bool:
    return os.path.isfile(os.path.join(_run_dir(exp_id), "eval_report.json"))


def _baseline_report_exists(exp_id: str) -> bool:
    return os.path.isfile(os.path.join(_run_dir(exp_id), "baseline", "eval_report.json"))


def _write_eval_stub(config_path: str, status: str, note: str = "") -> bool:
    """Backstop for T0.5: guarantee runs/<exp>/eval_report.json exists even when
    the eval phase produced none — eval skipped before the subprocess ran (no
    checkpoint / train failed), or a hard crash (OOM-kill) that killed runner.py
    before it wrote its own error stub. Never overwrites an existing report
    (runner.py's stub, or a real one). Returns True if it wrote a stub.
    """
    try:
        cfg = _load_config(config_path)
    except Exception:
        return False
    exp_id = cfg.get("experiment_id")
    if not exp_id:
        return False
    path = os.path.join(_run_dir(exp_id), "eval_report.json")
    if os.path.exists(path):
        return False
    stub = {
        "experiment_id": exp_id,
        "model_slug": (cfg.get("model") or {}).get("slug"),
        "seed": cfg.get("seed", 42),
        "compose_method": (cfg.get("rewards") or {}).get("compose_method", "advantage_weighted"),
        "status": status,
        "results": {},
    }
    if note:
        stub["note"] = note
    os.makedirs(_run_dir(exp_id), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stub, f, indent=2)
    print(f"  Wrote stub eval_report.json (status={status}) for {exp_id}: {note}")
    return True


def _ensure_baseline_symlink(target_run_dir: str, source_baseline_dir: str) -> bool:
    """Make runs/<target>/baseline → runs/<source>/baseline.

    Returns True if a symlink was created or replaced, False if a real
    directory already exists at the target (left alone — the user may have
    run a custom baseline). Replaces an existing symlink, which may point at
    a stale location.
    """
    link_path = os.path.join(target_run_dir, "baseline")
    os.makedirs(target_run_dir, exist_ok=True)

    if os.path.islink(link_path):
        os.unlink(link_path)
    elif os.path.exists(link_path):
        return False

    rel = os.path.relpath(source_baseline_dir, start=target_run_dir)
    os.symlink(rel, link_path)
    return True


def _run_train_phase(
    config_path: str,
    exp_id: str,
    smoke: bool,
    force: bool,
    retries: int,
    vllm: bool = False,
) -> PhaseResult:
    if not force and _checkpoint_exists(exp_id):
        return PhaseResult(status=STATUS_SKIP, note="checkpoint-final exists")

    cmd = [sys.executable, "-m", "training.train", "--config", config_path]
    if smoke:
        cmd.append("--smoke")
    if force:
        cmd.append("--overwrite")
    if vllm:
        cmd.append("--vllm")

    log_path = os.path.join(_run_dir(exp_id), "batch_train.log")
    status, attempts, dur = _run_phase(cmd, log_path, retries)
    return PhaseResult(status=status, duration_s=dur, attempts=attempts, log_path=log_path)


def _run_eval_phase(
    config_path: str,
    exp_id: str,
    smoke: bool,
    force: bool,
    retries: int,
    require_checkpoint: bool,
) -> PhaseResult:
    if require_checkpoint and not _checkpoint_exists(exp_id):
        return PhaseResult(status=STATUS_SKIP, note="no checkpoint to eval")
    if not force and _eval_report_exists(exp_id):
        return PhaseResult(status=STATUS_SKIP, note="eval_report.json exists")

    cmd = [sys.executable, "-m", "eval.runner", "--config", config_path]
    if smoke:
        cmd.append("--smoke")

    log_path = os.path.join(_run_dir(exp_id), "batch_eval.log")
    status, attempts, dur = _run_phase(cmd, log_path, retries)
    return PhaseResult(status=status, duration_s=dur, attempts=attempts, log_path=log_path)


def _run_baseline_phase(
    config_path: str,
    exp_id: str,
    slug: str,
    baseline_owners: dict[str, str],
    smoke: bool,
    force: bool,
    retries: int,
    dedup: bool,
) -> PhaseResult:
    """Run baseline assessment for this experiment's base model.

    If `dedup` and the slug already has an owner (another experiment that ran
    the baseline first), symlink this experiment's `baseline/` to the owner's
    `baseline/` instead of re-running.
    """
    if dedup and slug in baseline_owners:
        owner_exp = baseline_owners[slug]
        source = os.path.join(_run_dir(owner_exp), "baseline")
        if os.path.isdir(source):
            linked = _ensure_baseline_symlink(_run_dir(exp_id), source)
            note = (f"symlinked → {owner_exp}/baseline" if linked
                    else f"kept existing baseline/ (would have symlinked → {owner_exp}/baseline)")
            return PhaseResult(status=STATUS_SKIP, note=note)

    if not force and _baseline_report_exists(exp_id):
        baseline_owners.setdefault(slug, exp_id)
        return PhaseResult(status=STATUS_SKIP, note="baseline/eval_report.json exists")

    cmd = [sys.executable, "-m", "eval.runner", "--config", config_path, "--baseline"]
    if smoke:
        cmd.append("--smoke")

    log_path = os.path.join(_run_dir(exp_id), "batch_baseline.log")
    status, attempts, dur = _run_phase(cmd, log_path, retries)
    if status == STATUS_OK:
        baseline_owners[slug] = exp_id
    return PhaseResult(status=status, duration_s=dur, attempts=attempts, log_path=log_path)


def _run_compare(eval_run_dirs: list[str], out_dir: str) -> int:
    if len(eval_run_dirs) < 2:
        print(f"Skipping compare: only {len(eval_run_dirs)} eval report(s) available (need ≥2)")
        return 0
    cmd = [sys.executable, "-m", "eval.compare", "--runs", *eval_run_dirs, "--out", out_dir]
    log_path = os.path.join(out_dir, "batch_compare.log")
    os.makedirs(out_dir, exist_ok=True)
    return _tee_subprocess(cmd, log_path)


def _format_duration(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _status_icon(status: str) -> str:
    return {STATUS_OK: "✓", STATUS_SKIP: "·", STATUS_FAIL: "✗"}.get(status, "?")


def _write_summary(
    results: list[ExperimentResult],
    enabled_phases: list[str],
    out_path: str,
    total_duration_s: float,
) -> None:
    lines = [
        "# Batch run summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Total wall time: {_format_duration(total_duration_s)}",
        f"Experiments: {len(results)}",
        f"Phases: {', '.join(enabled_phases)}",
        "",
        "| Experiment | Model | " + " | ".join(p.capitalize() for p in enabled_phases) + " | Notes |",
        "|---|---|" + "|".join([":---:"] * len(enabled_phases)) + "|---|",
    ]
    for r in results:
        cells = [r.experiment_id, r.model_slug]
        notes = []
        for p in enabled_phases:
            pr = r.phases.get(p)
            if pr is None:
                cells.append("—")
            else:
                cell = _status_icon(pr.status)
                if pr.status == STATUS_OK and pr.duration_s > 0:
                    cell += f" ({_format_duration(pr.duration_s)}"
                    if pr.attempts > 1:
                        cell += f", {pr.attempts} attempts"
                    cell += ")"
                elif pr.status == STATUS_FAIL:
                    cell += f" ({pr.attempts} attempts)"
                cells.append(cell)
                if pr.note:
                    notes.append(f"{p}: {pr.note}")
        cells.append("; ".join(notes) if notes else "")
        lines.append("| " + " | ".join(cells) + " |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_summary(results: list[ExperimentResult], enabled_phases: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Batch summary")
    print("=" * 80)
    header = f"{'experiment_id':<40} {'model':<12} " + "  ".join(f"{p:<6}" for p in enabled_phases)
    print(header)
    print("-" * len(header))
    for r in results:
        cells = [f"{r.experiment_id:<40}", f"{r.model_slug:<12}"]
        for p in enabled_phases:
            pr = r.phases.get(p)
            mark = _status_icon(pr.status) if pr else "—"
            cells.append(f"{mark:<6}")
        print("  ".join(cells))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner: train and/or eval and/or baseline-assess a list of configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Config paths or shell globs (e.g. configs/e0-*.yaml configs/e1-token-length-qwen-7b.yaml)",
    )
    parser.add_argument("--train", action="store_true", help="Train each config")
    parser.add_argument("--eval", action="store_true", help="Run eval.runner per config")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (no-LoRA) eval per config; deduplicated by model slug")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Replicate each config across these training seeds (e.g. --seeds 42 43 44). "
                             "Each seed gets its own run dir <experiment_id>-s<seed>; baselines stay "
                             "deduplicated by model slug (seeds share one base-model baseline). "
                             "Omit for a single run at the config's own seed.")
    parser.add_argument("--smoke", action="store_true", help="Pass --smoke to every subprocess")
    parser.add_argument("--vllm", action="store_true", help="Pass --vllm to training subprocesses (route GRPO rollouts through vLLM)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run phases even if their output already exists (train passes --overwrite)")
    parser.add_argument("--retries", type=int, default=1,
                        help="Retry count per failed phase before giving up")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the auto eval.compare run at end of batch")
    parser.add_argument("--no-baseline-dedup", action="store_true",
                        help="Run baseline separately per config (do not share across configs with same model slug)")
    parser.add_argument("--compare-out", default="runs/comparison",
                        help="Output directory for eval.compare artifacts")
    parser.add_argument("--summary-dir", default="runs",
                        help="Directory to write batch_summary_<timestamp>.md")
    args = parser.parse_args()

    # Default to train+eval when no phase flag is given — the most common
    # overnight invocation. Explicit `--baseline` alone, or `--eval` alone,
    # only does that one phase.
    if not (args.train or args.eval or args.baseline):
        args.train = True
        args.eval = True

    if args.retries < 0:
        parser.error("--retries must be ≥ 0")
    return args


def main() -> None:
    args = _parse_args()
    configs = _expand_configs(args.configs)
    if not configs:
        print("No configs to run.")
        sys.exit(2)

    enabled = [p for p, on in (
        (PHASE_BASELINE, args.baseline),
        (PHASE_TRAIN, args.train),
        (PHASE_EVAL, args.eval),
    ) if on]

    print(f"Configs ({len(configs)}):")
    for c in configs:
        print(f"  {c}")
    print(f"Phases (in order): {', '.join(enabled)}")
    if args.seeds:
        print(f"Seeds: {args.seeds}  ({len(configs)} base config(s) × {len(args.seeds)} = "
              f"{len(configs) * len(args.seeds)} runs)")
    print(f"Retries per phase: {args.retries}; smoke={args.smoke}; force={args.force}")
    print("")

    # Pre-read all config headers so a single bad file fails fast, before any
    # GPU work starts, and expand into one (config_path, exp_id, slug) entry per
    # (config, seed). Without --seeds this is one entry per config at its own
    # seed — unchanged behaviour. With --seeds, each seed gets a materialized
    # config and a suffixed experiment_id / run dir; the shared model.slug keeps
    # baseline dedup working (seeds reuse one base-model baseline).
    headers: list[tuple[str, str, str]] = []
    for path in configs:
        try:
            base_cfg = _load_config(path)
            base_id = base_cfg.get("experiment_id")
            slug = (base_cfg.get("model") or {}).get("slug")
            if not base_id or not slug:
                raise ValueError("missing experiment_id or model.slug")
        except Exception as exc:
            print(f"✗ Could not read {path}: {exc}")
            sys.exit(2)
        if args.seeds:
            for s in args.seeds:
                sid = f"{base_id}-s{s}"
                headers.append((_materialize_seed_config(base_cfg, s, sid), sid, slug))
        else:
            headers.append((path, base_id, slug))

    results: list[ExperimentResult] = []
    baseline_owners: dict[str, str] = {}      # slug -> exp_id that owns the canonical baseline dir
    t_start = time.time()

    def _ensure_result(path: str, exp_id: str, slug: str) -> ExperimentResult:
        for r in results:
            if r.experiment_id == exp_id:
                return r
        new = ExperimentResult(config_path=path, experiment_id=exp_id, model_slug=slug)
        results.append(new)
        return new

    # Phase A: baselines first so trained eval reports can pick up vs_base_model deltas.
    if args.baseline:
        for path, exp_id, slug in headers:
            r = _ensure_result(path, exp_id, slug)
            print(f"\n── BASELINE  {exp_id}  ({slug})")
            r.phases[PHASE_BASELINE] = _run_baseline_phase(
                path, exp_id, slug, baseline_owners,
                smoke=args.smoke,
                force=args.force,
                retries=args.retries,
                dedup=not args.no_baseline_dedup,
            )

    # Phase B: train + eval per experiment (eval inside the loop, not at end, so
    # an experiment's report exists before the next one starts — useful for tail -f).
    for path, exp_id, slug in headers:
        r = _ensure_result(path, exp_id, slug)

        if args.train:
            print(f"\n── TRAIN     {exp_id}  ({slug})")
            r.phases[PHASE_TRAIN] = _run_train_phase(
                path, exp_id, smoke=args.smoke, force=args.force, retries=args.retries, vllm=args.vllm
            )
            # If train failed and eval would need that checkpoint, skip eval rather
            # than launching it pointlessly.
            if args.eval and r.phases[PHASE_TRAIN].status == STATUS_FAIL:
                r.phases[PHASE_EVAL] = PhaseResult(status=STATUS_SKIP, note="train failed")
                _write_eval_stub(path, status="skipped", note="train failed")
                continue

        if args.eval:
            print(f"\n── EVAL      {exp_id}  ({slug})")
            r.phases[PHASE_EVAL] = _run_eval_phase(
                path, exp_id,
                smoke=args.smoke,
                force=args.force,
                retries=args.retries,
                # When only --eval is passed (no --train this run), require a
                # pre-existing checkpoint. Otherwise eval.runner would crash.
                require_checkpoint=not args.train,
            )
            # T0.5 backstop: if the eval phase produced no report (skipped before
            # the subprocess ran, or a hard crash that beat runner.py's own stub),
            # drop one so the run still shows up in auto-compare.
            pe = r.phases[PHASE_EVAL]
            if pe.status in (STATUS_SKIP, STATUS_FAIL) and not _eval_report_exists(exp_id):
                _write_eval_stub(
                    path,
                    status=("error" if pe.status == STATUS_FAIL else "skipped"),
                    note=pe.note,
                )

    total = time.time() - t_start

    # Summary first — even if compare fails, the user still gets the table.
    _print_summary(results, enabled)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.summary_dir, f"batch_summary_{ts}.md")
    os.makedirs(args.summary_dir, exist_ok=True)
    _write_summary(results, enabled, summary_path, total)
    print(f"\nSummary written: {summary_path}")
    print(f"Total wall time: {_format_duration(total)}")

    # Auto-compare across every run that produced an eval_report.json this batch.
    if args.eval and not args.no_compare:
        eval_dirs = [
            _run_dir(r.experiment_id) for r in results
            if _eval_report_exists(r.experiment_id)
        ]
        print(f"\n── COMPARE   {len(eval_dirs)} run(s)")
        rc = _run_compare(eval_dirs, args.compare_out)
        if rc != 0:
            print(f"⚠  eval.compare exited {rc}")

    # Exit non-zero if any experiment had a hard failure, so wrapping shell
    # scripts / cron jobs can detect it.
    if any(pr.status == STATUS_FAIL for r in results for pr in r.phases.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
