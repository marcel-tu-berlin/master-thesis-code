"""Check batch evidence against the compact JSON actually shown to reviewers."""

import argparse
import json
from pathlib import Path

from trajectory_review.__main__ import digest, read, validate_result, write
from trajectory_review.batches import unpack, verify_batch


def check(result, packet):
    issues = []
    for issue in validate_result(result, packet):
        evidence = issue.get("evidence", {})
        if (
            issue["reason"] == "excerpt not found at cited location"
            and evidence.get("field") == "tool_call"
        ):
            call = packet["turns"][evidence["turn"]]["tool_calls"][evidence["action"]]
            shown = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
            if evidence["quote"].strip() and " ".join(
                evidence["quote"].split()
            ) in " ".join(shown.split()):
                continue
        issues.append(issue)
    return issues


def main(root):
    manifest = verify_batch(root)
    for reviewer, plan in manifest["plans"].items():
        for spec in plan["jobs"]:
            job = root / "results" / reviewer / Path(spec["packet"]).stem
            receipt_path = job / "complete.json"
            if not receipt_path.exists():
                continue
            receipt = read(receipt_path)
            for name, sha in receipt["output_sha256"].items():
                if digest(job / name) != sha:
                    raise ValueError(f"Changed raw batch result: {job}")
            results = read(job / "normalized.json")
            packets = {p["episode_id"]: p for p in unpack(read(Path(spec["packet"])))}
            for identifier, row in (results or {}).items():
                row["evidence_issues"] = check(row["review"], packets[identifier])
            value = {
                "parent_receipt_sha256": digest(receipt_path),
                "validator_sha256": digest(Path(__file__)),
                "reviews": results,
                "note": "Derived evidence check accepts the compact tool-call JSON in the batch prompt as well as the legacy serialization. Original responses and normalized output are unchanged.",
            }
            output = job / "checked.json"
            if output.exists():
                if read(output) != value:
                    raise ValueError(f"Changed derived evidence check: {output}")
            else:
                write(output, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    main(parser.parse_args().root.resolve())
