"""Append-only wall-time accounting for the single-GPU experiment pipeline."""

import json
import os
import socket
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@contextmanager
def measure_phase(run_dir, phase, *, checkpoint=None, scope="in_process"):
    """Persist start/end events; an unmatched start means cost is incomplete.

    GPU hours are allocated device-hours, including CPU/environment waits, not
    CUDA kernel time or energy. Unknown device allocation stays null. Retries
    append distinct attempts, and failed attempts still consume resources.
    """
    path = Path(run_dir) / "costs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_count = (
        len([d for d in visible.split(",") if d.strip() and d.strip() != "-1"])
        if visible is not None
        else None
    )
    identity = {
        "schema_version": 1,
        "attempt_id": uuid4().hex,
        "phase": phase,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "scope": scope,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": visible,
        "allocated_gpu_count": gpu_count,
    }

    def write(**event):
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {**identity, "recorded_at": datetime.now(UTC).isoformat(), **event}
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    started = time.perf_counter()
    write(event="start")
    status, error_type = "complete", None
    try:
        yield
    except BaseException as exc:
        status, error_type = "failed", type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter() - started
        write(
            event="end",
            status=status,
            error_type=error_type,
            wall_seconds=elapsed,
            allocated_gpu_hours=(
                elapsed * gpu_count / 3600 if gpu_count is not None else None
            ),
        )
