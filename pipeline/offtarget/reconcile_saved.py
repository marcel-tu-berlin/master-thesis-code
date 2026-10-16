"""Reproduce the fixed 2026-10-15 implementation check from local run archives.

Run from pipeline/: python -m offtarget.reconcile_saved --output runs/<new-folder>
This is a regression check of 13 archived evaluations, not another experiment.
"""

import argparse
import json
from pathlib import Path

from offtarget.__main__ import digest, run
from offtarget.analysis import analyze, load, summarize

CORPUS = (
    ("read-table-2", "s4016", 1, 2),
    ("email-inbox-noscroll", "s4021-c4096", 1, 2),
    ("tic-tac-toe", "s4230-c4096", 1, 2),
    ("click-tab-2", "s4231-c4096", 0, 1),
    ("click-checkboxes-soft", "s4232-c4096", 0, 1),
)
AUDITS = {
    "tic-tac-toe": (
        "board_audit.json",
        {
            "occupied_cell_at_decision": "known_occupied_cell_click",
            "missed_immediate_win": "missed_available_win",
            "unblocked_single_threat": "missed_unique_block",
        },
    ),
    "click-tab-2": (
        "tab_audit.json",
        {
            "wrong_visible_link": "wrong_visible_link",
            "leave_visible_target": "tab_change_with_visible_target",
        },
    ),
    "click-checkboxes-soft": ("checkbox_audit.json", {}),
}


def reference_checks(family, episodes, records, audit):
    """Return per-check agreement and located disagreements, without grading harm."""
    checks, disagreements = {}, []
    for seed, episode in episodes.items():
        measured = records[seed]["measurements"]
        comparisons = {}
        if family in AUDITS:
            prior = audit[seed]
            for name, old_name in AUDITS[family][1].items():
                comparisons[name] = (
                    measured[name]["events"],
                    sum(e[old_name] for e in prior["events"]),
                )
            if family == "click-checkboxes-soft":
                comparisons["native_success_incomplete"] = (
                    measured["native_success_incomplete"]["present"],
                    prior["native_success_incomplete"],
                )
                comparisons["incomplete_checkbox_submit"] = (
                    measured["incomplete_checkbox_submit"]["present"],
                    prior["attributed_submission"]
                    and not prior["final_state"]["exact"],
                )
        elif episode["terminated"]:
            name = (
                "table_submit_mismatch"
                if family == "read-table-2"
                else "inbox_commit_mismatch"
            )
            observations = measured[name]["observations"]
            comparisons[name + "_vs_native"] = (
                observations[-1]["occurred"] if observations else None,
                not episode["correct"],
            )
        for name, (actual, expected) in comparisons.items():
            counts = checks.setdefault(
                name, {"compared": 0, "agreed": 0, "disagreed": 0}
            )
            counts["compared"] += 1
            counts["agreed" if actual == expected else "disagreed"] += 1
            if actual != expected:
                disagreements.append(
                    {
                        "seed": seed,
                        "check": name,
                        "actual": actual,
                        "expected": expected,
                    }
                )
    return checks, disagreements


def reconcile(output):
    pipeline = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths, sources = {}, set(Path(__file__).parent.glob("*.py"))
    sources.update(
        [
            Path(__file__).with_name("README.md"),
            Path(__file__).with_name("checkbox_synonyms.json"),
            pipeline / "tests/test_offtarget_analysis.py",
        ]
    )
    for family, suffix, _, last_arm in CORPUS:
        for arm in range(last_arm + 1):
            root = pipeline / "runs" / f"e{arm}-{family}-{suffix}"
            if output.is_relative_to(root):
                raise ValueError("Output must be outside every source experiment")
            matches = list(root.glob("episodes_*.jsonl"))
            if len(matches) != 1:
                raise ValueError(f"Expected one archived episode file in {root}")
            paths[family, arm] = matches[0]
            sources.add(matches[0])
            if family in AUDITS:
                sources.add(root / AUDITS[family][0])
    hashes = {str(p.relative_to(pipeline)): digest(p) for p in sorted(sources)}
    validation = {"definition_version": 1, "source_sha256": hashes, "runs": {}}
    failed = False
    for (family, arm), path in paths.items():
        episodes = load(path)
        records = {s: analyze(e, family) for s, e in episodes.items()}
        metrics = summarize(list(records.values()))["metrics"]
        coverage = {
            name: {key: m[key] for key in ("unknown_episodes", "unknown_observations")}
            for name, m in metrics.items()
        }
        audit = {}
        if family in AUDITS:
            audit = {
                r["seed"]: r
                for r in json.loads(path.with_name(AUDITS[family][0]).read_text())[
                    "records"
                ]
            }
        checks, disagreements = reference_checks(family, episodes, records, audit)
        failed |= (
            len(records) != 200
            or bool(disagreements)
            or any(v for m in coverage.values() for v in m.values())
        )
        validation["runs"][path.parent.name] = {
            "family": family,
            "arm": arm,
            "episodes": len(records),
            "expected_episodes": 200,
            "coverage": coverage,
            "checks": checks,
            "disagreements": disagreements,
        }
    output.mkdir(parents=True)
    if not failed:
        for family, _, baseline, treatment in CORPUS:
            run(
                paths[family, baseline],
                paths[family, treatment],
                family,
                output / f"{family}-e{treatment}-vs-e{baseline}",
            )
            if baseline == 1:
                run(paths[family, 0], None, family, output / f"{family}-e0")
    if any(digest(pipeline / p) != expected for p, expected in hashes.items()):
        raise ValueError("Reconciliation inputs changed; no validation receipt written")
    validation["status"] = "failed" if failed else "passed"
    validation["report_receipt_sha256"] = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.glob("*/receipt.json"))
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    if failed:
        raise ValueError(f"Reconciliation failed; inspect {output / 'validation.json'}")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reconcile(args.output)
    print(f"Verified all 13 saved evaluations; wrote {args.output / 'validation.json'}")


if __name__ == "__main__":
    main()
