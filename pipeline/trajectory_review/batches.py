"""Review remaining trajectories in small packages with a shared time limit."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from trajectory_review.__main__ import (
    CATEGORIES,
    LABELS,
    MODELS,
    command,
    digest,
    object_schema,
    read,
    review_status,
    schema,
    validate_result,
    validate_schema,
    verify,
    write,
)
from trajectory_review.continue_census import explicit_nonreview

HERE = Path(__file__).resolve().parent


def batch_schema():
    original = schema()["properties"]
    detail = original["labels"]["properties"][CATEGORIES[0]]["properties"]
    return object_schema(
        {
            "reviews": {
                "type": "array",
                "items": object_schema(
                    {
                        "episode_id": {"type": "string"},
                        "labels": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(LABELS)},
                        },
                        "details": {
                            "type": "array",
                            "items": object_schema(
                                {
                                    "category": {
                                        "type": "string",
                                        "enum": list(CATEGORIES),
                                    },
                                    **{
                                        k: detail[k]
                                        for k in (
                                            "harm",
                                            "evidence",
                                            "available_check",
                                            "alternative_explanation",
                                            "rationale",
                                        )
                                    },
                                }
                            ),
                        },
                        "goal_alignment": original["goal_alignment"],
                        "exploratory_notes": {"type": "string"},
                    }
                ),
            }
        }
    )


def pack(episodes):
    """Intern exact repeated strings only; all original text is recoverable."""
    counts = Counter()

    def count(value):
        if isinstance(value, str) and len(value) >= 100:
            counts[value] += 1
        elif isinstance(value, dict):
            for item in value.values():
                count(item)
        elif isinstance(value, list):
            for item in value:
                count(item)

    count(episodes)
    identifiers = {s: f"t{i}" for i, (s, n) in enumerate(counts.items()) if n > 1}

    def replace(value):
        if isinstance(value, str) and value in identifiers:
            return {"text_ref": identifiers[value]}
        if isinstance(value, dict):
            if set(value) == {"text_ref"}:
                raise ValueError("Source collides with text-reference encoding")
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return value

    return {
        "texts": {identifier: s for s, identifier in identifiers.items()},
        "episodes": replace(episodes),
    }


def unpack(packet):
    def restore(value):
        if isinstance(value, dict):
            if set(value) == {"text_ref"}:
                return packet["texts"][value["text_ref"]]
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value

    return restore(packet["episodes"])


def normalize(response, episodes):
    validate_schema(response, batch_schema())
    expected = {e["episode_id"]: e for e in episodes}
    returned = [r["episode_id"] for r in response["reviews"]]
    if len(returned) != len(expected) or set(returned) != set(expected):
        raise ValueError("Missing, duplicate or foreign episode in batch response")
    results = {}
    for row in response["reviews"]:
        if len(row["labels"]) != len(CATEGORIES):
            raise ValueError("Expected exactly six construct labels")
        details = {d["category"]: d for d in row["details"]}
        if len(details) != len(row["details"]):
            raise ValueError("Duplicate construct detail")
        required = {
            c
            for c, label in zip(CATEGORIES, row["labels"], strict=True)
            if label in ("present", "unknown")
        }
        if set(details) != required:
            raise ValueError("Details must cover exactly present and unknown labels")
        labels = {}
        for category, label in zip(CATEGORIES, row["labels"], strict=True):
            if category in details:
                labels[category] = {
                    "label": label,
                    **{k: v for k, v in details[category].items() if k != "category"},
                }
            else:
                labels[category] = {
                    "label": label,
                    "harm": "none_supported",
                    "evidence": [],
                    "available_check": None,
                    "alternative_explanation": None,
                    "rationale": "Not elicited for negative labels in batch-v2; this is a coordinator format note, not a reviewer rationale.",
                }
        result = {
            "episode_id": row["episode_id"],
            "rubric_version": "1",
            "labels": labels,
            "goal_alignment": row["goal_alignment"],
            "exploratory_notes": row["exploratory_notes"],
        }
        issues = validate_result(result, expected[row["episode_id"]])
        results[row["episode_id"]] = {
            "review": result,
            "evidence_issues": issues,
            "review_status": review_status(result),
            "annotation_protocol": "batch-v2",
        }
    return results


def split(rows, size):
    """Fill packages in order without two trajectories of one question in any."""
    packages = []
    for row in rows:
        question = (row["stratum"], row["seed"])
        for package in packages:
            if len(package) < size and all(
                (r["stratum"], r["seed"]) != question for r in package
            ):
                package.append(row)
                break
        else:
            packages.append([row])
    return packages


def prepare(source, output, size):
    verify(source)
    rows = read(source / "coordinator_key.json")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "schema.json", batch_schema())
    (output / "rubric.md").write_text((HERE / "batch_rubric.md").read_text())
    sources = {
        str(p): digest(p)
        for p in (
            source / "manifest.json",
            source / "coordinator_key.json",
            Path(__file__),
            HERE / "batch_rubric.md",
        )
    }
    plans = {}
    for reviewer in MODELS:
        retained, pending = [], []
        for row in rows:
            job = source / "reviews" / reviewer / row["episode_id"]
            receipt_path = job / "complete.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt["packet_sha256"] != row["packet_sha256"]:
                    raise ValueError("Changed singleton input")
                for name, sha in receipt["output_sha256"].items():
                    if digest(job / name) != sha:
                        raise ValueError("Changed singleton output")
                sources[str(receipt_path)] = digest(receipt_path)
                retained.append(row["episode_id"])
            else:
                pending.append(row)
        folder = output / reviewer
        folder.mkdir()
        jobs = []
        for index, selected in enumerate(split(pending, size)):
            episodes = []
            for row in selected:
                path = source / "packets" / f"{row['episode_id']}.json"
                if digest(path) != row["packet_sha256"]:
                    raise ValueError("Changed trajectory packet")
                sources[str(path)] = digest(path)
                episodes.append(read(path))
            packet = pack(episodes)
            if unpack(packet) != episodes:
                raise ValueError("Lossy batch encoding")
            path = folder / f"{index:04}.json"
            write(path, packet)
            sources[str(path)] = digest(path)
            jobs.append(
                {
                    "packet": str(path),
                    "episode_ids": [e["episode_id"] for e in episodes],
                }
            )
        plans[reviewer] = {"retained": retained, "jobs": jobs}
    write(
        output / "manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_root": str(source),
            "models": MODELS,
            "effort": "xhigh",
            "batch_size": size,
            "episodes": len(rows),
            "plans": plans,
            "source_sha256": sources,
            "frozen_sha256": {
                n: digest(output / n) for n in ("rubric.md", "schema.json")
            },
            "protocol_change": "User authorized small packages on 4 November. Retain existing singleton reviews without rerating; new annotations share context in groups and omit rationales for negative labels. Report modes separately; do not claim context independence or protocol equivalence.",
            "time_limit": "One shared 60-minute window from first pilot request. No automatic continuation after the deadline or quota failure.",
            "refusals": "Previously refused requests stay unavailable and are never resubmitted. A refused batch stays unavailable without splitting or rephrasing.",
        },
    )
    print(
        json.dumps(
            {
                r: {"retained": len(p["retained"]), "new_batches": len(p["jobs"])}
                for r, p in plans.items()
            }
        )
    )


def verify_batch(root):
    manifest = read(root / "manifest.json")
    verify(Path(manifest["source_root"]))
    for name, sha in manifest["source_sha256"].items():
        if digest(name) != sha:
            raise ValueError(f"Changed batch source: {name}")
    for name, sha in manifest["frozen_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError(f"Changed batch protocol: {name}")
    return manifest


def invoke(args, prompt, job, deadline):
    env = os.environ.copy()
    for key in (
        "CLAUDECODE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_EFFORT_LEVEL",
    ):
        env.pop(key, None)
    with (
        tempfile.TemporaryDirectory(prefix="trajectory-batch-") as cwd,
        (job / "stdout.jsonl").open("w") as out,
        (job / "stderr.txt").open("w") as err,
    ):
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            text=True,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        try:
            process.communicate(
                prompt, timeout=max(0.01, min(600, deadline - time.time()))
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise
        return process.returncode


def run_one(root, reviewer, spec, deadline):
    packet = Path(spec["packet"])
    job = root / "results" / reviewer / packet.stem
    job.mkdir(parents=True, exist_ok=False)
    prompt = (
        (root / "rubric.md").read_text()
        + "\nBEGIN TRAJECTORY DATA\n"
        + json.dumps(read(packet), ensure_ascii=False, separators=(",", ":"))
        + "\nEND TRAJECTORY DATA\n"
    )
    (job / "prompt.txt").write_text(prompt)
    started = time.time()
    try:
        args = command(reviewer, root, job)
        code = invoke(args, prompt, job, deadline)
        raw = (job / "stdout.jsonl").read_text()
        unavailable = False
        if reviewer == "claude":
            envelope = json.loads(raw)
            explanation = str(envelope.get("result", "")).lower()
            unavailable = explicit_nonreview(envelope) or (
                envelope.get("structured_output") is None
                and "safety classifier" in explanation
                and "withheld" in explanation
            )
            if not unavailable and (code or envelope.get("is_error")):
                raise RuntimeError(f"Claude provider error: {envelope.get('result')}")
            result = envelope.get("structured_output")
            usage = {
                "usage": envelope.get("usage"),
                "modelUsage": envelope.get("modelUsage"),
            }
        else:
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
            completed = [e for e in events if e.get("type") == "turn.completed"]
            if code or not completed:
                raise RuntimeError("Codex request failed; inspect retained raw output")
            result = read(job / "model_result.json")
            usage = completed[-1].get("usage")
        normalized = None if unavailable else normalize(result, unpack(read(packet)))
        write(job / "labels.json", result)
        write(job / "normalized.json", normalized)
        write(
            job / "complete.json",
            {
                "reviewer": reviewer,
                "model": MODELS[reviewer],
                "effort": "xhigh",
                "episode_ids": spec["episode_ids"],
                "review_status": "unavailable" if unavailable else "evaluated",
                "started_utc": datetime.fromtimestamp(started, UTC).isoformat(),
                "elapsed_seconds": time.time() - started,
                "usage": usage,
                "manifest_sha256": digest(root / "manifest.json"),
                "packet_sha256": digest(packet),
                "cli_version": subprocess.check_output(
                    [reviewer, "--version"], text=True
                ).strip(),
                "output_sha256": {
                    p.name: digest(p) for p in job.iterdir() if p.is_file()
                },
            },
        )
        return {
            "batch": packet.stem,
            "episodes": len(spec["episode_ids"]),
            "unavailable": unavailable,
            "seconds": round(time.time() - started, 1),
        }
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        write(
            job / "failure.json",
            {
                "error": str(exc),
                "started_utc": datetime.fromtimestamp(started, UTC).isoformat(),
                "elapsed_seconds": time.time() - started,
            },
        )
        raise


def run(root, reviewer, workers, limit):
    manifest = verify_batch(root)
    with (root / f"{reviewer}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        window = root / "window.json"
        with (root / "window.lock").open("a") as window_lock:
            fcntl.flock(window_lock, fcntl.LOCK_EX)
            if not window.exists():
                write(window, {"start_epoch": time.time(), "duration_seconds": 3600})
            timing = read(window)
        deadline = timing["start_epoch"] + timing["duration_seconds"]
        if time.time() >= deadline:
            raise ValueError("One-hour review window ended; no requests started")
        pending = []
        for spec in manifest["plans"][reviewer]["jobs"]:
            job = root / "results" / reviewer / Path(spec["packet"]).stem
            if (job / "complete.json").exists():
                receipt = read(job / "complete.json")
                if receipt["manifest_sha256"] != digest(root / "manifest.json"):
                    raise ValueError("Changed completed batch protocol")
                for name, sha in receipt["output_sha256"].items():
                    if digest(job / name) != sha:
                        raise ValueError("Changed completed batch output")
            elif job.exists():
                raise ValueError(f"Unresolved batch attempt: {job}; no automatic retry")
            else:
                pending.append(spec)
        stop = threading.Event()

        def worker(spec):
            if stop.is_set() or time.time() >= deadline:
                return
            try:
                print(json.dumps(run_one(root, reviewer, spec, deadline)), flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                stop.set()
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(worker, spec)
                for spec in (pending[:limit] if limit else pending)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        verify_batch(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--size", type=int, default=16, choices=range(2, 17))
    p = sub.add_parser("run")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--reviewer", required=True, choices=MODELS)
    p.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    p.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.source.resolve(), args.output.resolve(), args.size)
    else:
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        run(args.root.resolve(), args.reviewer, args.workers, args.limit)
