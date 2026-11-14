"""Consolidate a complete two-reviewer census without adjudicating disagreement."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from eval.metrics import _wilson_ci
from eval.paired import exact_binomial_p
from trajectory_review.__main__ import CATEGORIES, LABELS, digest, read, verify, write


def outcome_group(base, treatment):
    return {
        (True, True): "both_successful",
        (True, False): "e1_only_successful",
        (False, True): "e2_only_successful",
        (False, False): "both_unsuccessful",
    }[base, treatment]


def census(labels):
    counts = dict(Counter(labels))
    n = len(labels)
    known = n - counts.get("unknown", 0)
    present = counts.get("present", 0)
    applicable_known = present + counts.get("absent", 0)
    return {
        "episodes": n,
        "labels": counts,
        "known_episodes": known,
        "present_per_known": present / known if known else None,
        "present_per_applicable_known": present / applicable_known
        if applicable_known
        else None,
        "wilson_95": list(_wilson_ci(present, known)) if known else None,
        "unknown_bounds_per_all": [
            present / n,
            (present + counts.get("unknown", 0)) / n,
        ]
        if n
        else None,
    }


def paired(base, treatment):
    if len(base) != len(treatment):
        raise ValueError("Unpaired label arrays")
    pairs = [
        (b, t) for b, t in zip(base, treatment, strict=True) if "unknown" not in (b, t)
    ]
    n = len(pairs)
    b = sum(x == "present" for x, _ in pairs)
    t = sum(y == "present" for _, y in pairs)
    new = sum(x != "present" and y == "present" for x, y in pairs)
    removed = sum(x == "present" and y != "present" for x, y in pairs)
    bounds = None
    if n:
        blo, bhi = _wilson_ci(b, n, ci=0.975)
        tlo, thi = _wilson_ci(t, n, ci=0.975)
        bounds = [tlo - bhi, thi - blo]
    return {
        "baseline": census(base),
        "treatment": census(treatment),
        "known_pairs": n,
        "unknown_pairs": len(base) - n,
        "baseline_present_on_known_pairs": b,
        "treatment_present_on_known_pairs": t,
        "new": new,
        "removed": removed,
        "difference": (t - b) / n if n else None,
        "difference_interval_95_conservative": bounds,
        "mcnemar_p_unadjusted": exact_binomial_p(new, new + removed) if n else None,
        "transitions": [
            {"baseline": x, "treatment": y, "count": k}
            for (x, y), k in sorted(Counter(zip(base, treatment, strict=True)).items())
        ],
    }


def agreement(left, right, categories=LABELS):
    indexes = {label: i for i, label in enumerate(categories)}
    encoded = np.array(
        [[indexes[a], indexes[b]] for a, b in zip(left, right, strict=True)], dtype=int
    )
    matrix = np.zeros((4, 4), dtype=int)
    np.add.at(matrix, (encoded[:, 0], encoded[:, 1]), 1)

    def kappa(m):
        n = m.sum()
        observed = np.trace(m) / n
        expected = float(np.dot(m.sum(axis=0), m.sum(axis=1))) / (n * n)
        return (observed - expected) / (1 - expected) if expected < 1 else None

    observed_kappa = kappa(matrix)
    interval = None
    if observed_kappa is not None:
        rng = np.random.default_rng(20261103)
        samples = rng.multinomial(len(left), matrix.ravel() / len(left), size=2000)
        values = [k for m in samples if (k := kappa(m.reshape(4, 4))) is not None]
        if values:
            interval = np.quantile(values, [0.025, 0.975]).tolist()
    return {
        "labels": categories,
        "matrix_claude_rows_codex_columns": matrix.tolist(),
        "raw_agreement": float(np.trace(matrix) / len(left)),
        "kappa": observed_kappa,
        "kappa_bootstrap_95": interval,
        "bootstrap_valid": len(values) if observed_kappa is not None else 0,
    }


def consensus(a, b):
    # Exact agreement only: do not adjudicate unknown or applicability conflicts.
    return a if a == b else "unknown"


def observed_harm(record, category):
    label = record["checked"][category]
    harm = record["harm"][category]
    if label == "unknown" or harm == "unknown":
        return "unknown"
    return "present" if label == "present" and harm == "observed" else "absent"


def main(root, output):
    manifest = verify(root)
    mapping = read(root / "coordinator_key.json")
    missing = [
        (reviewer, row["episode_id"])
        for row in mapping
        for reviewer in ("claude", "codex")
        if not (
            root / "reviews" / reviewer / row["episode_id"] / "complete.json"
        ).exists()
    ]
    if missing:
        raise ValueError(
            f"Census incomplete: {len(missing)} missing reviewer/episode results; no aggregate report written"
        )
    output.mkdir(parents=True, exist_ok=False)
    labels, metadata, hashes, unavailable = {}, {}, {}, []
    disagreement_file = (output / "disagreements.jsonl").open("x")
    with disagreement_file:
        for row in mapping:
            opaque = row["episode_id"]
            if digest(root / "packets" / f"{opaque}.json") != row["packet_sha256"]:
                raise ValueError(f"Changed blinded packet: {opaque}")
            metadata[opaque] = row
            labels[opaque] = {}
            records = {}
            for reviewer in ("claude", "codex"):
                job = root / "reviews" / reviewer / opaque
                receipt = read(job / "complete.json")
                if receipt["packet_sha256"] != row["packet_sha256"] or receipt[
                    "manifest_sha256"
                ] != digest(root / "manifest.json"):
                    raise ValueError(f"Review receipt mismatch: {job}")
                for name, expected in receipt["output_sha256"].items():
                    if digest(job / name) != expected:
                        raise ValueError(f"Review artifact changed: {job / name}")
                hashes[str(job / "complete.json")] = digest(job / "complete.json")
                result = read(job / receipt["attempt"] / "labels.json")
                issues = read(job / receipt["attempt"] / "evidence_issues.json")
                bad = {issue["category"] for issue in issues}
                refused = receipt.get("review_status") == "unavailable"
                if result is None and not refused:
                    raise ValueError(
                        f"Missing semantic response without unavailable status: {job}"
                    )
                if refused:
                    unavailable.append(
                        {
                            "reviewer": reviewer,
                            "episode_id": opaque,
                            "stratum": row["stratum"],
                            "arm": row["arm"],
                        }
                    )
                records[reviewer] = {
                    "review": result,
                    "issues": issues,
                    "review_status": receipt.get("review_status", "evaluated"),
                }
                labels[opaque][reviewer] = {
                    "annotation_protocol": receipt.get(
                        "annotation_protocol", "singleton-v1"
                    ),
                    "raw": {
                        c: result["labels"][c]["label"] if not refused else "unknown"
                        for c in CATEGORIES
                    },
                    "checked": {
                        c: result["labels"][c]["label"]
                        if c not in bad and not refused
                        else "unknown"
                        for c in CATEGORIES
                    },
                    "harm": {
                        c: result["labels"][c]["harm"]
                        if c not in bad and not refused
                        else "unknown"
                        for c in CATEGORIES
                    },
                    "goal": result["goal_alignment"]["status"]
                    if "goal_alignment" not in bad and not refused
                    else "unknown",
                }
            labels[opaque]["agreement_only"] = {
                mode: {
                    c: consensus(
                        labels[opaque]["claude"][mode][c],
                        labels[opaque]["codex"][mode][c],
                    )
                    for c in CATEGORIES
                }
                for mode in ("raw", "checked")
            }
            conflicts = [
                c
                for c in CATEGORIES
                if labels[opaque]["claude"]["checked"][c]
                != labels[opaque]["codex"]["checked"][c]
            ]
            if (
                conflicts
                or any(
                    r["issues"] or r["review_status"] == "unavailable"
                    for r in records.values()
                )
                or labels[opaque]["claude"]["goal"] != labels[opaque]["codex"]["goal"]
            ):
                disagreement_file.write(
                    json.dumps(
                        {
                            "metadata": row,
                            "conflicting_categories": conflicts,
                            "reviews": records,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    runs = defaultdict(list)
    strata = defaultdict(lambda: defaultdict(dict))
    for row in mapping:
        runs[row["run"]].append(row["episode_id"])
        strata[row["stratum"]][row["arm"]][row["seed"]] = row["episode_id"]
    run_results = {}
    for run, ids in sorted(runs.items()):
        run_results[run] = {
            "episodes": len(ids),
            "annotation_protocols": {
                reviewer: dict(
                    Counter(labels[i][reviewer]["annotation_protocol"] for i in ids)
                )
                for reviewer in ("claude", "codex")
            },
            "agreement_raw": {
                c: agreement(
                    [labels[i]["claude"]["raw"][c] for i in ids],
                    [labels[i]["codex"]["raw"][c] for i in ids],
                )
                for c in CATEGORIES
            },
            "goal_alignment": {
                reviewer: dict(Counter(labels[i][reviewer]["goal"] for i in ids))
                for reviewer in ("claude", "codex")
            },
            "goal_vs_native": {
                reviewer: [
                    {"native_correct": native, "goal_alignment": goal, "count": n}
                    for (native, goal), n in sorted(
                        Counter(
                            (metadata[i]["native_correct"], labels[i][reviewer]["goal"])
                            for i in ids
                        ).items()
                    )
                ]
                for reviewer in ("claude", "codex")
            },
            "agreement_goal": agreement(
                [labels[i]["claude"]["goal"] for i in ids],
                [labels[i]["codex"]["goal"] for i in ids],
                ("satisfied", "partial", "failed", "unknown"),
            ),
        }
    comparisons = {}
    for stratum, arms in sorted(strata.items()):
        contrasts = [("e1", "e2")]
        if "e0" in arms:
            contrasts.append(("e0", "e1"))
        if "placebo" in arms:
            contrasts.append(("placebo", "e2"))
        comparisons[stratum] = {}
        for baseline, treatment in contrasts:
            if arms[baseline].keys() != arms[treatment].keys():
                raise ValueError(f"Nonmatching test instances: {stratum}")
            seeds = sorted(arms[baseline])
            groups = {"all": seeds}
            if (baseline, treatment) == ("e1", "e2"):
                for group in (
                    "both_successful",
                    "e1_only_successful",
                    "e2_only_successful",
                    "both_unsuccessful",
                ):
                    groups[group] = [
                        s
                        for s in seeds
                        if outcome_group(
                            metadata[arms[baseline][s]]["native_correct"],
                            metadata[arms[treatment][s]]["native_correct"],
                        )
                        == group
                    ]
                assert sum(len(v) for k, v in groups.items() if k != "all") == len(
                    seeds
                )
                groups["supplement_both_successful_e2_shorter"] = [
                    s
                    for s in groups["both_successful"]
                    if metadata[arms[treatment][s]]["n_tokens"]
                    < metadata[arms[baseline][s]]["n_tokens"]
                ]
                for reviewer in ("claude", "codex"):
                    groups[f"sensitivity_same_annotation_protocol_{reviewer}"] = [
                        s
                        for s in seeds
                        if labels[arms[baseline][s]][reviewer]["annotation_protocol"]
                        == labels[arms[treatment][s]][reviewer]["annotation_protocol"]
                    ]
            results = {}
            for group, selected in groups.items():
                base_ids, treatment_ids = (
                    [arms[arm][s] for s in selected] for arm in (baseline, treatment)
                )
                differences = [
                    metadata[t]["n_tokens"] - metadata[b]["n_tokens"]
                    for b, t in zip(base_ids, treatment_ids, strict=True)
                ]
                percentages = [
                    100 * (metadata[t]["n_tokens"] / metadata[b]["n_tokens"] - 1)
                    for b, t in zip(base_ids, treatment_ids, strict=True)
                ]
                results[group] = {
                    "pairs": len(selected),
                    "seeds": selected,
                    "native_successes": {
                        arm: sum(
                            metadata[arms[arm][s]]["native_correct"] for s in selected
                        )
                        for arm in (baseline, treatment)
                    },
                    "tokens": {
                        "median_difference": median(differences)
                        if differences
                        else None,
                        "median_percent_change": median(percentages)
                        if percentages
                        else None,
                        "shorter": sum(d < 0 for d in differences),
                        "equal": sum(d == 0 for d in differences),
                        "longer": sum(d > 0 for d in differences),
                    },
                    "reviewers": {
                        reviewer: {
                            mode: {
                                c: paired(
                                    [labels[i][reviewer][mode][c] for i in base_ids],
                                    [
                                        labels[i][reviewer][mode][c]
                                        for i in treatment_ids
                                    ],
                                )
                                for c in CATEGORIES
                            }
                            for mode in ("raw", "checked")
                        }
                        for reviewer in ("claude", "codex", "agreement_only")
                    },
                    "goal_transitions": {
                        reviewer: [
                            {"baseline": a, "treatment": b, "count": n}
                            for (a, b), n in sorted(
                                Counter(
                                    (
                                        labels[i][reviewer]["goal"],
                                        labels[j][reviewer]["goal"],
                                    )
                                    for i, j in zip(
                                        base_ids, treatment_ids, strict=True
                                    )
                                ).items()
                            )
                        ]
                        for reviewer in ("claude", "codex")
                    },
                    "observed_harm": {
                        reviewer: {
                            c: paired(
                                [
                                    observed_harm(labels[i][reviewer], c)
                                    for i in base_ids
                                ],
                                [
                                    observed_harm(labels[i][reviewer], c)
                                    for i in treatment_ids
                                ],
                            )
                            for c in CATEGORIES
                        }
                        for reviewer in ("claude", "codex")
                    },
                }
            comparisons[stratum][f"{treatment}-vs-{baseline}"] = results
    write(
        output / "summary.json",
        {
            "manifest": manifest,
            "runs": run_results,
            "comparisons": comparisons,
            "unavailable_reviews": unavailable,
            "interpretation": "Exploratory. Evidence-invalid or unavailable labels become unknown only in derived data; original reviews remain intact. Exact model agreement is not human adjudication. Four-outcome groups and compression selection are descriptive. P-values are unadjusted; intervals describe question variability, not training-seed robustness.",
        },
    )
    write(
        output / "receipt.json",
        {
            "source_sha256": {
                **hashes,
                str(root / "manifest.json"): digest(root / "manifest.json"),
                str(Path(__file__)): digest(Path(__file__)),
            },
            "output_sha256": {
                p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        },
    )
    print(
        json.dumps(
            {
                "episodes": len(mapping),
                "unavailable_reviews": len(unavailable),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
