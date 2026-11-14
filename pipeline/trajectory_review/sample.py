"""Freeze an outcome-stratified paired sample and reuse existing annotations."""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from trajectory_review.__main__ import digest, read, verify, write
from trajectory_review.batch_evidence import main as recheck
from trajectory_review.batches import verify_batch
from trajectory_review.consolidate import outcome_group

GROUPS = (
    "both_successful",
    "e1_only_successful",
    "e2_only_successful",
    "both_unsuccessful",
)


def select(rows, count=100):
    strata = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        strata[row["stratum"]][row["arm"]][row["seed"]] = row
    rng = random.Random(20261104)
    selected, allocation = [], {}
    for index, (stratum, arms) in enumerate(sorted(strata.items())):
        if arms["e1"].keys() != arms["e2"].keys():
            raise ValueError("Different E1/E2 question sets")
        cells = {group: [] for group in GROUPS}
        for seed in sorted(arms["e1"]):
            group = outcome_group(
                arms["e1"][seed]["native_correct"], arms["e2"][seed]["native_correct"]
            )
            cells[group].append(seed)
        for cell in cells.values():
            rng.shuffle(cell)
        target = count // len(strata) + (index < count % len(strata))
        if target > sum(map(len, cells.values())):
            raise ValueError("Sample target exceeds stratum")
        chosen = {group: [] for group in GROUPS}
        while sum(map(len, chosen.values())) < target:
            for group in GROUPS:
                if cells[group] and sum(map(len, chosen.values())) < target:
                    chosen[group].append(cells[group].pop())
        allocation[stratum] = {
            group: {
                "population": len(cells[group]) + len(chosen[group]),
                "sample": len(chosen[group]),
                "seeds": sorted(chosen[group]),
            }
            for group in GROUPS
        }
        pairs = []
        for seeds in chosen.values():
            for seed in seeds:
                pair = [arms["e1"][seed], arms["e2"][seed]]
                rng.shuffle(pair)
                pairs.append(pair)
        rng.shuffle(pairs)
        selected.extend(pairs)
    rng.shuffle(selected)
    return [row for pair in selected for row in pair], allocation


def prepare(source, output):
    original = verify(source)
    rows, allocation = select(read(source / "coordinator_key.json"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "packets").mkdir()
    for row in rows:
        p = source / "packets" / f"{row['episode_id']}.json"
        if digest(p) != row["packet_sha256"]:
            raise ValueError("Changed source packet")
        shutil.copyfile(p, output / "packets" / p.name)
    for name in ("rubric.md", "schema.json"):
        shutil.copyfile(source / name, output / name)
    write(output / "coordinator_key.json", rows)
    write(
        output / "selection.json",
        {
            "pairs": len(rows) // 2,
            "random_seed": 20261104,
            "allocation": allocation,
            "rule": "Equal family/seed targets (14 or 15); round-robin allocation over available native-outcome groups with rare-cell saturation; uniform random selection within each cell. Selection never reads LLM labels, annotation availability, token change or detector events. Paired arm order is randomized.",
        },
    )
    manifest = dict(original)
    manifest.update(
        {
            "episodes": len(rows),
            "review_scope": "stratified_paired_sample",
            "full_corpus_manifest_sha256": digest(source / "manifest.json"),
            "sampling": "100 outcome-stratified paired E1/E2 questions across seven family/seed strata. Unweighted exploratory descriptions, not population prevalence; rare outcomes are intentionally oversampled.",
            "primary_population": "LLM: the selected 100 E1/E2 pairs. Deterministic analysis: all original matched questions in the frozen corpus.",
            "controls": "No additional LLM calls for E0/placebo; their previous reviews remain preserved outside the selected primary sample. Full deterministic controls remain included.",
            "annotation_protocol": "Reuse singleton-v1 and batch-v2 results; fill only missing selected annotations with batch-v2. Context modes must remain explicit.",
            "source_sha256": {
                **original["source_sha256"],
                str(source / "manifest.json"): digest(source / "manifest.json"),
                str(Path(__file__)): digest(Path(__file__)),
            },
            "frozen_sha256": {
                n: digest(output / n)
                for n in (
                    "rubric.md",
                    "schema.json",
                    "coordinator_key.json",
                    "selection.json",
                )
            },
        }
    )
    write(output / "manifest.json", manifest)


def import_one(
    root,
    row,
    reviewer,
    source_job,
    receipt,
    result,
    issues,
    protocol,
    status,
    source_batch=None,
):
    job = root / "reviews" / reviewer / row["episode_id"]
    if (job / "complete.json").exists():
        return
    for name, sha in receipt["output_sha256"].items():
        if digest(source_job / name) != sha:
            raise ValueError("Changed original review output")
    attempt = job / "imported"
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / "labels.json", result)
    write(attempt / "evidence_issues.json", issues)
    sources = {
        str((source_job / name).resolve()): sha
        for name, sha in receipt["output_sha256"].items()
    }
    sources[str(source_job / "complete.json")] = digest(source_job / "complete.json")
    if (source_job / "checked.json").exists():
        sources[str(source_job / "checked.json")] = digest(source_job / "checked.json")
    write(
        job / "complete.json",
        {
            "reviewer": reviewer,
            "model": receipt["model"],
            "effort": receipt["effort"],
            "packet_sha256": row["packet_sha256"],
            "manifest_sha256": digest(root / "manifest.json"),
            "attempt": "imported",
            "review_status": status,
            "annotation_protocol": protocol,
            "source_batch": source_batch,
            "source_receipt": str(source_job / "complete.json"),
            "output_sha256": {
                **sources,
                **{str(p.relative_to(job)): digest(p) for p in attempt.iterdir()},
            },
        },
    )


def import_reviews(root, singleton, batches):
    verify(root)
    rows = {r["episode_id"]: r for r in read(root / "coordinator_key.json")}
    if singleton:
        verify(singleton)
        for reviewer in ("claude", "codex"):
            for row in rows.values():
                source_job = singleton / "reviews" / reviewer / row["episode_id"]
                if not (source_job / "complete.json").exists():
                    continue
                receipt = read(source_job / "complete.json")
                if receipt["packet_sha256"] != row["packet_sha256"]:
                    raise ValueError("Different imported trajectory")
                result = read(source_job / receipt["attempt"] / "labels.json")
                issues = read(source_job / receipt["attempt"] / "evidence_issues.json")
                import_one(
                    root,
                    row,
                    reviewer,
                    source_job,
                    receipt,
                    result,
                    issues,
                    "singleton-v1",
                    receipt.get("review_status", "evaluated"),
                )
    for batch in batches:
        manifest = verify_batch(batch)
        batch_rows = {
            row["episode_id"]: row
            for row in read(Path(manifest["source_root"]) / "coordinator_key.json")
        }
        recheck(batch)
        for reviewer, plan in manifest["plans"].items():
            for spec in plan["jobs"]:
                source_job = batch / "results" / reviewer / Path(spec["packet"]).stem
                if not (source_job / "complete.json").exists():
                    continue
                receipt = read(source_job / "complete.json")
                results = read(source_job / "checked.json")["reviews"]
                for identifier in spec["episode_ids"]:
                    if identifier not in rows:
                        continue
                    if (
                        batch_rows[identifier]["packet_sha256"]
                        != rows[identifier]["packet_sha256"]
                    ):
                        raise ValueError("Different imported batch trajectory")
                    item = results[identifier] if results else None
                    import_one(
                        root,
                        rows[identifier],
                        reviewer,
                        source_job,
                        receipt,
                        item["review"] if item else None,
                        item["evidence_issues"] if item else [],
                        "batch-v2",
                        item["review_status"] if item else "unavailable",
                        str(source_job),
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("import")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--singleton", type=Path)
    p.add_argument("--batches", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.source.resolve(), args.output.resolve())
    else:
        import_reviews(
            args.root.resolve(),
            args.singleton.resolve() if args.singleton else None,
            [p.resolve() for p in args.batches],
        )
