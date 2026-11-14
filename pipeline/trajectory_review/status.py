"""Read local census progress without querying either provider."""

import argparse
import json
import subprocess
from pathlib import Path

from trajectory_review.__main__ import read


def status(root):
    expected = {r["episode_id"] for r in read(root / "coordinator_key.json")}
    processes = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    output = {"expected_per_reviewer": len(expected), "reviewers": {}}
    for reviewer in ("claude", "codex"):
        folder = root / "reviews" / reviewer
        completed = {p.parent.name: read(p) for p in folder.glob("*/complete.json")}
        if not completed.keys() <= expected:
            raise ValueError("Unexpected episode in completed review directory")
        unavailable = sum(
            r.get("review_status") == "unavailable" for r in completed.values()
        )
        active = [
            int(line.split(None, 1)[0])
            for line in processes
            if "trajectory_review" in line
            and root.name in line
            and f"--reviewer {reviewer}" in line
        ]
        unresolved = [
            str(p.relative_to(root))
            for p in folder.glob("*/attempt-*/failure.json")
            if p.parent.parent.name not in completed
        ]
        output["reviewers"][reviewer] = {
            "recorded_requests": len(completed),
            "evaluated": len(completed) - unavailable,
            "unavailable": unavailable,
            "active_processes": active,
            "unresolved_failure_files": unresolved,
            "stage_complete": len(completed) == len(expected),
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(status(args.root.resolve()), sort_keys=True))
