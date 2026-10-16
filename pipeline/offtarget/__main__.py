"""Usage: python -m offtarget --help (CPU-only, reads saved JSONL files)."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from eval.checkpoints import experiment_root
from eval.paired import load_config, split_tasks
from offtarget.analysis import analyze, compare, load, summarize
from offtarget.tasks import TASK_METRICS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(report):
    lines = [
        "# Offline off-target measurements",
        "",
        f"Family: `{report['family']}`. Definition version: 1.",
        "",
        "Exploratory state-grounded events. These are not automatic labels of harmful behavior or reward-bias substitution.",
        "",
    ]
    for arm, summary in report["arms"].items():
        lines += [
            f"## {arm}",
            "",
            f"Episodes: {summary['episodes']}. Stop reasons: `{json.dumps(summary['stop_reasons'], sort_keys=True)}`.",
            "",
            f"Answered calls: {summary['answered_calls']}; later calls in a precommitted batch: {summary['precommitted_answered_calls']}. Episodes without saved turns: {summary['episodes_without_turns']}.",
            "",
            "| Event | Episodes with event / known | Unknown episodes | Events / observed opportunities | Unknown observations |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, m in summary["metrics"].items():
            lines.append(
                f"| {name} | {m['episodes_with_event']} / {m['known_episodes']} | {m['unknown_episodes']} | {m['events']} / {m['opportunities']} | {m['unknown_observations']} |"
            )
        lines += [""]
    if "comparison" in report:
        comparison = report["comparison"]
        lines += [
            "## Paired treatment minus baseline",
            "",
            f"Identical initial observations: {comparison['paired_instances']}. Missing-observation exclusions: {len(comparison['excluded'])}. Baseline-only seeds: {len(comparison['baseline_only'])}; treatment-only seeds: {len(comparison['treatment_only'])}.",
            "",
            "| Event | Known pairs | Unknown pairs | Opportunities baseline / treatment | Pairs exposed in both | New / removed event | Difference (percentage points) | Approximate conservative 95% interval |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in comparison["metrics"].items():
            difference = (
                "unknown"
                if m["difference"] is None
                else f"{100 * m['difference']:+.1f}"
            )
            bounds = m["difference_interval_95_conservative"]
            interval = (
                "unknown"
                if bounds is None
                else f"[{100 * bounds[0]:+.1f}, {100 * bounds[1]:+.1f}]"
            )
            lines.append(
                f"| {name} | {m['known_pairs']} | {m['unknown_pairs']} | {m['baseline_observed_opportunities']} / {m['treatment_observed_opportunities']} | {m['pairs_with_opportunities_both']} | {m['increased']} / {m['decreased']} | {difference} | {interval} |"
            )
        lines += ["", "Stop-reason transitions:", ""]
        for row in comparison["stop_transitions"]:
            lines.append(f"- {row['baseline']} -> {row['treatment']}: {row['count']}")
    lines += [
        "",
        "Episode records contain zero-based turn/action evidence and coverage. Missing evidence is unknown. An episode with no applicable opportunity contributes a negative occurrence and no opportunity count.",
        "",
        "JSON includes marginal Wilson intervals and exploratory, unadjusted exact McNemar p-values. Difference intervals combine 97.5% marginal Wilson bounds with Bonferroni; they do not exploit pairing and are deliberately conservative. They are approximate, not exact coverage guarantees. These uncertainties concern evaluated instances, not variability across training seeds. No equivalence or harm verdict is inferred.",
        "",
        "Read pipeline/offtarget/README.md and docs/plans/off-target-human-llm-review.md before interpreting these numbers.",
        "",
    ]
    return "\n".join(lines)


def run(baseline, treatment, family, output):
    paths = {"baseline": Path(baseline).resolve()}
    if treatment is not None:
        paths["treatment"] = Path(treatment).resolve()
        if paths["baseline"] == paths["treatment"]:
            raise ValueError("Baseline and treatment must be different files")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    roots = {p: experiment_root(str(p.parent)) for p in paths.values()}
    if any(output.is_relative_to(root) for root in roots.values()):
        raise ValueError("Output must be outside the input run directories")
    sources = set(paths.values())
    for path in paths.values():
        for name in ("config.yaml", "env_stamp.json"):
            sibling = roots[path] / name
            if sibling.is_file():
                sources.add(sibling)
        if path.with_name("eval_report.json").is_file():
            sources.add(path.with_name("eval_report.json"))
    sources.update(Path(__file__).parent.glob("*.py"))
    sources.add(Path(__file__).with_name("checkbox_synonyms.json"))
    sources.add(Path(__file__).with_name("README.md"))
    for name in ("metrics.py", "paired.py", "checkpoints.py"):
        sources.add(Path(__file__).parents[1] / "eval" / name)
    hashes = {str(p): digest(p) for p in sorted(sources)}
    for path in paths.values():
        config = load_config(str(path.parent))
        split = path.stem.removeprefix("episodes_")
        tasks = split_tasks(config, split)
        if tasks and (len(tasks) != 1 or (family != "generic" and tasks != [family])):
            raise ValueError(
                f"{path}: declared family {family} disagrees with split tasks {tasks}; mixed-family input is unsupported"
            )
    episodes = {arm: load(path) for arm, path in paths.items()}
    records = {
        arm: [analyze(e, family) for _, e in sorted(rows.items())]
        for arm, rows in episodes.items()
    }
    report = {
        "definition_version": 1,
        "family": family,
        "inputs": {arm: str(p) for arm, p in paths.items()},
        "arms": {arm: summarize(rows) for arm, rows in records.items()},
    }
    if treatment is not None:
        report["comparison"] = compare(
            episodes["baseline"],
            episodes["treatment"],
            records["baseline"],
            records["treatment"],
        )
    if any(digest(Path(path)) != expected for path, expected in hashes.items()):
        raise ValueError(
            "An input or analysis file changed during analysis; no output written"
        )
    output.mkdir(parents=True)
    for arm, rows in records.items():
        with (output / f"events_{arm}.jsonl").open("x") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(markdown(report))
    receipt = {
        "definition_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "PyYAML": version("PyYAML"),
        },
        "source_sha256": hashes,
        "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir())},
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=TASK_METRICS)
    parser.add_argument(
        "--baseline", required=True, type=Path, help="Saved episodes_<split>.jsonl"
    )
    parser.add_argument("--treatment", type=Path, help="Optional matched episode JSONL")
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory outside input runs"
    )
    args = parser.parse_args()
    run(args.baseline, args.treatment, args.family, args.output)
    print(f"Wrote {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
