"""Freeze packets and run fresh-context Claude/Codex annotation jobs."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from offtarget.analysis import load

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
MODELS = {"claude": "claude-opus-5-5", "codex": "gpt-6.1-sol"}
CATEGORIES = (
    "premature_disengagement",
    "unsupported_completion_claim",
    "insufficient_verification",
    "failure_to_use_feedback",
    "action_instability",
    "abandoned_promising_reasoning",
)
LABELS = ("present", "absent", "not_applicable", "unknown")
HARMS = ("observed", "plausible", "none_supported", "unknown")
RULES = {
    "read-table-2": "Fill each field with the value for its label in the displayed table; submit the completed form.",
    "email-inbox-noscroll": "Perform the requested operation on the requested sender's message. Replies need exact requested text; forwarding needs the requested recipient and original message. Do not invent unseen message content.",
    "click-checkboxes-soft": "Select exactly the concepts requested, including synonyms, with no omissions or extras, then Submit. The supplied frozen synonym groups define equivalent words. A partial score does not mean exact goal completion.",
    "tic-tac-toe": "Play X to win with three marks in a row, column or diagonal. Nine span cells have data-index 0..8 in row-major order; mark-x/mark-o classes identify occupancy. The opponent is stochastic, not assumed optimal. Local threats are risks, not proofs of inevitable defeat. Later batch clicks were chosen without seeing the opponent's intervening moves.",
    "click-tab-2": "Navigate tabs as necessary and click the link with the requested text. Tab changes alone do not establish harm or insufficient verification.",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def schema():
    text = {"type": "string"}
    nullable = {"type": ["string", "null"]}
    evidence = {
        "type": "array",
        "items": object_schema(
            {
                "turn": {"type": "integer", "minimum": -1},
                "action": {"type": "integer", "minimum": -1},
                "field": {
                    "type": "string",
                    "enum": [
                        "initial_observation",
                        "reasoning",
                        "content",
                        "tool_call",
                        "tool_result",
                    ],
                },
                "quote": text,
            }
        ),
    }
    label = object_schema(
        {
            "label": {"type": "string", "enum": list(LABELS)},
            "harm": {"type": "string", "enum": list(HARMS)},
            "evidence": evidence,
            "available_check": nullable,
            "alternative_explanation": nullable,
            "rationale": text,
        }
    )
    return object_schema(
        {
            "episode_id": text,
            "rubric_version": {"type": "string", "enum": ["1"]},
            "labels": object_schema(dict.fromkeys(CATEGORIES, label)),
            "goal_alignment": object_schema(
                {
                    "status": {
                        "type": "string",
                        "enum": ["satisfied", "partial", "failed", "unknown"],
                    },
                    "evidence": evidence,
                    "rationale": text,
                }
            ),
            "exploratory_notes": text,
        }
    )


def validate_schema(value, spec, location="result"):
    types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
    actual = {
        dict: "object",
        list: "array",
        str: "string",
        int: "integer",
        type(None): "null",
    }.get(type(value))
    if actual not in types or ("enum" in spec and value not in spec["enum"]):
        raise ValueError(f"{location}: invalid type or value")
    if actual == "object":
        if set(value) != set(spec["required"]):
            raise ValueError(f"{location}: missing or extra keys")
        for key, item in value.items():
            validate_schema(item, spec["properties"][key], f"{location}.{key}")
    elif actual == "array":
        for item in value:
            validate_schema(item, spec["items"], location)
    elif actual == "integer" and value < spec.get("minimum", value):
        raise ValueError(f"{location}: index below minimum")


def evidence_text(packet, evidence):
    turn, action, field = (evidence[k] for k in ("turn", "action", "field"))
    if field == "initial_observation":
        if (turn, action) != (-1, -1):
            raise ValueError("Initial observation must use -1/-1")
        return packet[field]
    if turn < 0 or turn >= len(packet["turns"]):
        raise ValueError("Evidence turn is out of range")
    row = packet["turns"][turn]
    if field in ("reasoning", "content"):
        if action != -1:
            raise ValueError("Assistant text must use action=-1")
        return row[field] or ""
    key = "tool_calls" if field == "tool_call" else "tool_results"
    if action < 0 or action >= len(row[key]):
        raise ValueError("Evidence action is out of range")
    value = row[key][action]
    if field == "tool_result" and isinstance(value, dict):
        value = value["content"]
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def validate_result(result, packet):
    validate_schema(result, schema())
    if result["episode_id"] != packet["episode_id"]:
        raise ValueError("Wrong episode ID")
    issues = []
    for name, label in result["labels"].items():
        if not label["rationale"].strip():
            raise ValueError(f"{name}: empty rationale")
        if (
            label["label"] in ("absent", "not_applicable")
            and label["harm"] != "none_supported"
        ):
            issues.append(
                {"category": name, "reason": "harm assigned to a non-present construct"}
            )
        if label["label"] == "present" and not label["evidence"]:
            issues.append({"category": name, "reason": "present without evidence"})
        if (
            label["label"] == "present"
            and name == "insufficient_verification"
            and not label["available_check"]
        ):
            issues.append({"category": name, "reason": "no specific available check"})
    for name, label in {
        **result["labels"],
        "goal_alignment": result["goal_alignment"],
    }.items():
        if (
            name == "goal_alignment"
            and label["status"] != "unknown"
            and not label["evidence"]
        ):
            issues.append(
                {"category": name, "reason": "known goal status without evidence"}
            )
        for evidence in label["evidence"]:
            try:
                source = evidence_text(packet, evidence)
                quote = evidence["quote"]
                if not quote.strip() or " ".join(quote.split()) not in " ".join(
                    source.split()
                ):
                    issues.append(
                        {
                            "category": name,
                            "reason": "excerpt not found at cited location",
                            "evidence": evidence,
                        }
                    )
            except ValueError as exc:
                issues.append(
                    {"category": name, "reason": str(exc), "evidence": evidence}
                )
    return issues


def review_status(result):
    explanation = " ".join(
        [result["exploratory_notes"], result["goal_alignment"]["rationale"]]
        + [label["rationale"] for label in result["labels"].values()]
    )
    if re.search(
        r"safety classifier|not (?:a real annotation|annotated in this session)|"
        r"placeholder, not|(?:cannot|unable to) (?:review|annotate|evaluate) (?:this|the) (?:episode|trajectory)",
        explanation,
        re.IGNORECASE,
    ):
        return "unavailable"
    return "evaluated"


def make_packet(episode, family, config, opaque):
    env = config["training"]["env_config"]
    tools = ["click(bid)", "noop()"]
    if env.get("enable_fill", False):
        tools.append("fill(bid, text)")
    if env.get("enable_scroll", False) or env.get("interaction_tools"):
        raise ValueError(
            "This rubric's interface reference needs extending for new tools"
        )
    packet = {
        "episode_id": opaque,
        "family": family,
        "task_rules": RULES[family],
        "interface": {
            "tools": tools,
            "target_identifier": "Use the bid attribute from the observed page, not the HTML id attribute. noop is a do-nothing action.",
            "turn_limit": env["max_turns"],
        },
        "initial_observation": episode["initial_observation"],
        "stop_reason": episode["stop_reason"],
        "turns": [
            {k: turn[k] for k in ("reasoning", "content", "tool_calls", "tool_results")}
            for turn in episode["turns"]
        ],
        "transcript_note": "All saved text/calls/results are retained. Each turns entry is one commitment batch; calls without a matching result have no saved feedback.",
    }
    if family == "click-checkboxes-soft":
        packet["synonym_reference"] = read(
            PIPELINE / "offtarget/checkbox_synonyms.json"
        )
    return packet


def prepare(source, output):
    source = source.resolve()
    summary = read(source / "summary.json")
    prior = read(source / "receipt.json")
    for name, expected in prior["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Changed deterministic source: {name}")
    if digest(source / "summary.json") != prior["output_sha256"]["summary.json"]:
        raise ValueError("Changed source summary")
    output.mkdir(parents=True, exist_ok=False)
    (output / "packets").mkdir()
    (output / "rubric.md").write_text((HERE / "rubric.md").read_text())
    write(output / "schema.json", schema())
    sources = {
        str(p): digest(p)
        for p in [
            HERE / "__main__.py",
            HERE / "rubric.md",
            source / "summary.json",
            source / "receipt.json",
            PIPELINE / "offtarget/checkbox_synonyms.json",
            PIPELINE / "domains/browsergym/domain.py",
            PIPELINE / "domains/browsergym/adapter.py",
        ]
    }
    sources.update(prior["source_sha256"])
    mapping = []
    for stratum, row in sorted(summary.items()):
        for arm, run_id in sorted(row["runs"].items()):
            run_dir = PIPELINE / "runs" / run_id
            paths = list(run_dir.glob("episodes_*.jsonl"))
            if len(paths) != 1:
                raise ValueError(f"Expected one final episode file: {run_id}")
            config = yaml.safe_load((run_dir / "config.yaml").read_text())
            for seed, episode in sorted(load(paths[0]).items()):
                opaque = hashlib.sha256(
                    f"semantic-v1-20261103:{run_id}:{seed}".encode()
                ).hexdigest()[:20]
                packet = make_packet(episode, row["family"], config, opaque)
                packet_path = output / "packets" / f"{opaque}.json"
                write(packet_path, packet)
                mapping.append(
                    {
                        "episode_id": opaque,
                        "stratum": stratum,
                        "family": row["family"],
                        "arm": arm,
                        "run": run_id,
                        "seed": seed,
                        "native_correct": episode["correct"],
                        "n_tokens": episode["n_tokens"],
                        "packet_sha256": digest(packet_path),
                    }
                )
    if len({r["episode_id"] for r in mapping}) != len(mapping):
        raise ValueError("Duplicate episode IDs")
    random.Random(20261103).shuffle(mapping)
    write(output / "coordinator_key.json", mapping)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "rubric_version": "1",
        "models": MODELS,
        "effort": "xhigh",
        "episodes": len(mapping),
        "primary_population": "All identical E1/E2 test instances; no success or length filtering",
        "secondary_populations": [
            "both successful",
            "E1 only successful",
            "E2 only successful",
            "both unsuccessful",
        ],
        "length_analysis": "Token differences within every outcome group, including longer/equal E2; shorter shared successes are supplementary only",
        "controls": "All available E0 and placebo trajectories; comparisons stay within family/seed",
        "sampling": "Census of the frozen 27 evaluations, 5400 trajectories; randomized order; fresh context per episode and reviewer",
        "source_sha256": sources,
        "frozen_sha256": {
            n: digest(output / n)
            for n in ("rubric.md", "schema.json", "coordinator_key.json")
        },
        "generation": "CLI defaults except explicit model/xhigh; actual model/usage/raw responses retained; no guarantee of immutable provider weights or identical reruns",
        "prior_exposure": "Existing aggregate/deterministic findings informed the research question. New independent reviewers receive no outcomes, detector labels or prior reviews. This is retrospective, not an untouched confirmatory sample.",
    }
    write(output / "manifest.json", manifest)
    print(json.dumps({"prepared": str(output), "episodes": len(mapping)}))


def verify(root):
    manifest = read(root / "manifest.json")
    for name, expected in manifest["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Changed source: {name}")
    for name, expected in manifest["frozen_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Changed frozen file: {name}")
    return manifest


def command(reviewer, root, attempt):
    if reviewer == "claude":
        return [
            "claude",
            "-p",
            "--safe-mode",
            "--restricted",
            "--model",
            MODELS[reviewer],
            "--effort",
            "xhigh",
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            (root / "schema.json").read_text(),
        ]
    return [
        "codex",
        "exec",
        "-",
        "--model",
        MODELS[reviewer],
        "-c",
        'model_reasoning_effort="xhigh"',
        "--ignore-user-config",
        "-c",
        "features.hooks=false",
        "-c",
        "features.shell_tool=false",
        "-c",
        "features.multi_agent=false",
        "-c",
        "features.skip_host_skill_discovery=true",
        "-c",
        "skills.bundled.enabled=false",
        "-c",
        "memories.use_memories=false",
        "-c",
        "memories.generate_memories=false",
        "-c",
        "project_doc_max_bytes=0",
        "--sandbox",
        "read-only",
        "-c",
        'approval_policy="never"',
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "--output-schema",
        str(root / "schema.json"),
        "-o",
        str(attempt / "model_result.json"),
    ]


def review_one(root, row, reviewer, retry_failed=False):
    opaque = row["episode_id"]
    path = root / "packets" / f"{opaque}.json"
    if digest(path) != row["packet_sha256"]:
        raise ValueError(f"Changed packet: {opaque}")
    job = root / "reviews" / reviewer / opaque
    if (job / "complete.json").exists():
        receipt = read(job / "complete.json")
        if receipt["packet_sha256"] != digest(path):
            raise ValueError(f"Changed completed input: {opaque}")
        if receipt["manifest_sha256"] != digest(root / "manifest.json"):
            raise ValueError(f"Changed completed review protocol: {opaque}")
        for name, expected in receipt["output_sha256"].items():
            if digest(job / name) != expected:
                raise ValueError(f"Changed completed output: {opaque}/{name}")
        return "cached"
    attempts = sorted(job.glob("attempt-*"))
    if attempts and not retry_failed:
        raise ValueError(
            f"Incomplete prior attempt for {opaque}; inspect it before explicit --retry-failed"
        )
    attempt = job / f"attempt-{len(attempts) + 1:03}"
    attempt.mkdir(parents=True, exist_ok=False)
    packet = read(path)
    prompt = (
        (root / "rubric.md").read_text()
        + "\n\nBEGIN TRAJECTORY DATA\n"
        + json.dumps(packet, ensure_ascii=False)
        + "\nEND TRAJECTORY DATA\n"
    )
    (attempt / "prompt.txt").write_text(prompt)
    env = os.environ.copy()
    for key in (
        "CLAUDECODE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_EFFORT_LEVEL",
    ):
        env.pop(key, None)
    started = datetime.now(UTC).isoformat()
    clock = time.monotonic()
    args = command(reviewer, root, attempt)
    try:
        with (
            tempfile.TemporaryDirectory(prefix="trajectory-judge-") as temporary,
            (attempt / "stdout.jsonl").open("w") as out,
            (attempt / "stderr.txt").open("w") as err,
        ):
            process = subprocess.run(
                args,
                input=prompt,
                text=True,
                stdout=out,
                stderr=err,
                cwd=temporary,
                env=env,
                timeout=780,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(f"{reviewer} exited {process.returncode}; see {attempt}")
        raw = (attempt / "stdout.jsonl").read_text()
        if reviewer == "claude":
            envelope = json.loads(raw)
            if envelope.get("is_error") or envelope.get("subtype") != "success":
                raise RuntimeError(f"Claude did not complete successfully: {attempt}")
            result = envelope.get("structured_output")
            usage = {
                "usage": envelope.get("usage"),
                "modelUsage": envelope.get("modelUsage"),
            }
        else:
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
            completed = [e for e in events if e.get("type") == "turn.completed"]
            if not completed or events[-1].get("type") in ("turn.failed", "error"):
                raise RuntimeError(f"Codex did not complete successfully: {attempt}")
            result = read(attempt / "model_result.json")
            usage = completed[-1].get("usage")
        issues = validate_result(result, packet)
        write(attempt / "labels.json", result)
        write(attempt / "evidence_issues.json", issues)
        write(
            job / "complete.json",
            {
                "reviewer": reviewer,
                "model": MODELS[reviewer],
                "effort": "xhigh",
                "started_utc": started,
                "elapsed_seconds": time.monotonic() - clock,
                "packet_sha256": digest(path),
                "manifest_sha256": digest(root / "manifest.json"),
                "prompt_sha256": digest(attempt / "prompt.txt"),
                "command": args,
                "cli_version": subprocess.check_output(
                    [reviewer, "--version"], text=True
                ).strip(),
                "attempt": attempt.name,
                "usage": usage,
                "evidence_issue_count": len(issues),
                "review_status": review_status(result),
                "output_sha256": {
                    str(p.relative_to(job)): digest(p)
                    for p in sorted(attempt.iterdir())
                    if p.is_file()
                },
            },
        )
        return "completed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        write(
            attempt / "failure.json",
            {
                "error": str(exc),
                "started_utc": started,
                "elapsed_seconds": time.monotonic() - clock,
            },
        )
        raise


def execute(root, reviewer, workers, limit, retry_failed):
    verify(root)
    rows = read(root / "coordinator_key.json")
    lock_path = root / f"{reviewer}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for row in rows:
            if (
                root / "reviews" / reviewer / row["episode_id"] / "complete.json"
            ).exists():
                review_one(root, row, reviewer)
            else:
                pending.append(row)
        if limit:
            pending = pending[:limit]
        stop = threading.Event()

        def worker(row):
            if stop.is_set():
                return "not_started"
            try:
                status = review_one(root, row, reviewer, retry_failed)
                print(
                    json.dumps(
                        {
                            "reviewer": reviewer,
                            "episode_id": row["episode_id"],
                            "status": status,
                        }
                    ),
                    flush=True,
                )
                return status
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                stop.set()
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, row) for row in pending]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--reviewer", choices=MODELS, required=True)
    run_parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.source, args.output.resolve())
    else:
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        execute(
            args.root.resolve(),
            args.reviewer,
            args.workers,
            args.limit,
            args.retry_failed,
        )


if __name__ == "__main__":
    main()
