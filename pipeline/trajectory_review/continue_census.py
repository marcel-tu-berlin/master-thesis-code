"""Continue past explicit provider non-reviews, never past other failures."""

import argparse
from pathlib import Path

from trajectory_review.__main__ import MODELS, execute, read
from trajectory_review.unavailable import record


def explicit_nonreview(envelope):
    text = str(envelope.get("result", "")).lower()
    if (
        envelope.get("is_error")
        and "safeguards flagged this message" in text
        and "can't respond to this message" in text
    ):
        return True
    if (
        envelope.get("is_error")
        or envelope.get("subtype") != "success"
        or envelope.get("structured_output") is not None
    ):
        return False
    return (
        "not calling structuredoutput" in text
        and "episode needs to be annotated outside this session" in text
    ) or (
        ("safety classifier" in text or "safety check" in text)
        and ("stopped" in text or "blocked" in text)
    )


def main(root, reviewer, workers):
    while True:
        try:
            execute(root, reviewer, workers, 0, False)
            return
        except (ValueError, RuntimeError):
            failures = [
                p
                for p in (root / "reviews" / reviewer).glob("*/attempt-*/failure.json")
                if not (p.parent.parent / "complete.json").exists()
            ]
            if reviewer != "claude" or not failures:
                raise
            # Do not retry a refusal. Other provider/schema failures remain fatal.
            if not all(
                explicit_nonreview(read(p.parent / "stdout.jsonl")) for p in failures
            ):
                raise
            for failure in failures:
                episode_id = failure.parent.parent.name
                record(
                    root,
                    reviewer,
                    episode_id,
                    "Explicit provider refusal or non-review in raw response; no annotation returned. Retained as unavailable without rerating, rephrasing, changing models or bypassing the provider restriction.",
                )
                print(
                    f"{reviewer}: unavailable {episode_id}; continuing other episodes",
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reviewer", required=True, choices=MODELS)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 9))
    args = parser.parse_args()
    main(args.root.resolve(), args.reviewer, args.workers)
