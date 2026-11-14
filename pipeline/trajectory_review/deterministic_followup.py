"""Rerun the full deterministic panel after the complete semantic review stage."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from offtarget.__main__ import run
from offtarget.analysis import compare, load, summarize

from trajectory_review.__main__ import PIPELINE, digest, read, verify, write
from trajectory_review.consolidate import outcome_group


def groups(base, treatment):
    if base.keys() != treatment.keys():
        raise ValueError("Different question sets")
    seeds = sorted(base)
    result = {"all": seeds}
    for name in (
        "both_successful",
        "e1_only_successful",
        "e2_only_successful",
        "both_unsuccessful",
    ):
        result[name] = [
            s
            for s in seeds
            if outcome_group(base[s]["correct"], treatment[s]["correct"]) == name
        ]
    result["supplement_both_successful_e2_shorter"] = [
        s
        for s in result["both_successful"]
        if treatment[s]["n_tokens"] < base[s]["n_tokens"]
    ]
    return result


def main(root, semantic, output):
    manifest = verify(root)
    semantic_receipt = read(semantic / "receipt.json")
    semantic_summary = read(semantic / "summary.json")
    semantic_manifest = semantic_summary["manifest"]
    same_corpus = semantic_manifest == manifest or (
        semantic_manifest.get("review_scope") == "stratified_paired_sample"
        and semantic_manifest.get("full_corpus_manifest_sha256")
        == digest(root / "manifest.json")
    )
    if (
        digest(semantic / "summary.json")
        != semantic_receipt["output_sha256"]["summary.json"]
        or not same_corpus
    ):
        raise ValueError("Semantic consolidation does not match this frozen corpus")
    if (
        sum(run["episodes"] for run in semantic_summary["runs"].values())
        != semantic_manifest["episodes"]
    ):
        raise ValueError("Incomplete semantic census")
    rows = read(root / "coordinator_key.json")
    strata = defaultdict(dict)
    for row in rows:
        strata[row["stratum"]][row["arm"]] = (row["family"], row["run"])
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        str(p): digest(p)
        for p in (
            Path(__file__),
            root / "manifest.json",
            semantic / "summary.json",
            semantic / "receipt.json",
        )
    }
    summary = {}
    for stratum, arms in sorted(strata.items()):
        family = next(iter(arms.values()))[0]
        paths = {}
        for arm, (_, run_id) in arms.items():
            matches = list((PIPELINE / "runs" / run_id).glob("episodes_*.jsonl"))
            if len(matches) != 1:
                raise ValueError(f"Expected one saved evaluation: {run_id}")
            paths[arm] = matches[0]
        episodes = {arm: load(path) for arm, path in paths.items()}
        summary[stratum] = {
            "family": family,
            "runs": {arm: run_id for arm, (_, run_id) in arms.items()},
            "comparisons": {},
        }
        contrasts = [("e1", "e2"), ("e0", "e1")]
        if "placebo" in arms:
            contrasts.append(("placebo", "e2"))
        for baseline, treatment in contrasts:
            contrast = f"{treatment}-vs-{baseline}"
            folder = output / stratum / contrast
            report = run(paths[baseline], paths[treatment], family, folder)
            if (
                report["comparison"]["paired_instances"] != 200
                or report["comparison"]["excluded"]
            ):
                raise ValueError("Expected 200 fully paired episodes")
            for name, value in read(folder / "receipt.json")["source_sha256"].items():
                if name in source_hashes and source_hashes[name] != value:
                    raise ValueError(f"Source changed between comparisons: {name}")
                source_hashes[name] = value
            events = {
                arm: {
                    r["seed"]: r
                    for r in map(
                        json.loads,
                        (folder / f"events_{arm}.jsonl").read_text().splitlines(),
                    )
                }
                for arm in ("baseline", "treatment")
            }
            base, trt = episodes[baseline], episodes[treatment]
            cohorts = (
                groups(base, trt)
                if baseline == "e1" and treatment == "e2"
                else {"all": sorted(base)}
            )
            results = {}
            for cohort, seeds in cohorts.items():
                br, tr = (
                    [events[arm][s] for s in seeds] for arm in ("baseline", "treatment")
                )
                differences = [trt[s]["n_tokens"] - base[s]["n_tokens"] for s in seeds]
                results[cohort] = {
                    "episodes": len(seeds),
                    "seeds": seeds,
                    "tokens": {
                        "shorter": sum(d < 0 for d in differences),
                        "equal": sum(d == 0 for d in differences),
                        "longer": sum(d > 0 for d in differences),
                        "median_change": median(differences) if differences else None,
                    },
                    "arms": {"baseline": summarize(br), "treatment": summarize(tr)}
                    if seeds
                    else None,
                    "comparison": compare(
                        {s: base[s] for s in seeds}, {s: trt[s] for s in seeds}, br, tr
                    )
                    if seeds
                    else None,
                }
            if results["all"]["comparison"] != report["comparison"]:
                raise ValueError(
                    "Full-population comparison changed during cohort analysis"
                )
            summary[stratum]["comparisons"][contrast] = results
    write(output / "summary.json", summary)
    for name, expected in source_hashes.items():
        if digest(name) != expected:
            raise ValueError(f"Changed source: {name}")
    verify(root)
    write(
        output / "receipt.json",
        {
            "source_sha256": source_hashes,
            "output_sha256": {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
            "primary_population": "All 200 matched questions; four native-outcome groups are secondary; shorter shared successes supplementary",
            "semantic_unavailable_count": len(semantic_summary["unavailable_reviews"]),
            "semantic_review_scope": semantic_manifest.get(
                "review_scope", "full_census"
            ),
            "semantic_review_episodes": semantic_manifest["episodes"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), args.semantic.resolve(), args.output.resolve())
