"""Record an inspected provider non-review without fabricating semantic labels."""

import argparse
from pathlib import Path

from trajectory_review.__main__ import MODELS, digest, read, verify, write


def record(root, reviewer, episode_id, reason):
    verify(root)
    rows = {r["episode_id"]: r for r in read(root / "coordinator_key.json")}
    row = rows[episode_id]
    job = root / "reviews" / reviewer / episode_id
    if (job / "complete.json").exists():
        raise FileExistsError(job / "complete.json")
    attempts = sorted(job.glob("attempt-*"))
    if not attempts or not (attempts[-1] / "failure.json").exists():
        raise ValueError("Only an inspected failed attempt can be marked unavailable")
    attempt = attempts[-1]
    if any((attempt / n).exists() for n in ("labels.json", "evidence_issues.json")):
        raise ValueError("Preserve existing parsed output; do not replace it")
    packet = root / "packets" / f"{episode_id}.json"
    if digest(packet) != row["packet_sha256"]:
        raise ValueError("Changed packet")
    write(attempt / "labels.json", None)
    write(attempt / "evidence_issues.json", [])
    write(
        attempt / "unavailable.json",
        {
            "reason": reason,
            "status": "No semantic annotation; raw provider response retained. No rerating attempted.",
        },
    )
    write(
        job / "complete.json",
        {
            "reviewer": reviewer,
            "model": MODELS[reviewer],
            "effort": "xhigh",
            "packet_sha256": digest(packet),
            "manifest_sha256": digest(root / "manifest.json"),
            "prompt_sha256": digest(attempt / "prompt.txt"),
            "attempt": attempt.name,
            "review_status": "unavailable",
            "evidence_issue_count": 0,
            "labels_origin": "No labels returned; null is a coordinator record, not a model judgment",
            "unavailable_reason": reason,
            "output_sha256": {
                str(p.relative_to(job)): digest(p)
                for p in sorted(attempt.iterdir())
                if p.is_file()
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reviewer", required=True, choices=MODELS)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    record(args.root.resolve(), args.reviewer, args.episode_id, args.reason)
