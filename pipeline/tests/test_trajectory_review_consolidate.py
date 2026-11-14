import json
import subprocess

import pytest
import trajectory_review.consolidate as consolidation
from trajectory_review.__main__ import CATEGORIES, digest, write
from trajectory_review.consolidate import (
    agreement,
    census,
    consensus,
    observed_harm,
    outcome_group,
    paired,
)
from trajectory_review.continue_census import explicit_nonreview
from trajectory_review.deterministic_followup import groups
from trajectory_review.status import status


def test_all_four_outcome_groups_are_retained():
    groups = [outcome_group(a, b) for a in (False, True) for b in (False, True)]
    assert set(groups) == {
        "both_successful",
        "e1_only_successful",
        "e2_only_successful",
        "both_unsuccessful",
    }


def test_deterministic_groups_include_losses_gains_failures_and_longer_successes():
    base = {
        i: {"correct": a, "n_tokens": 10}
        for i, a in enumerate((True, True, False, False, True))
    }
    treatment = {
        i: {"correct": b, "n_tokens": n}
        for i, (b, n) in enumerate(
            ((True, 12), (False, 5), (True, 3), (False, 4), (True, 9))
        )
    }
    actual = groups(base, treatment)
    assert actual["all"] == [0, 1, 2, 3, 4]
    assert actual["both_successful"] == [0, 4]
    assert actual["e1_only_successful"] == [1]
    assert actual["e2_only_successful"] == [2]
    assert actual["both_unsuccessful"] == [3]
    assert actual["supplement_both_successful_e2_shorter"] == [4]


def test_unknown_is_not_negative_and_applicability_is_reported():
    s = census(["present", "absent", "not_applicable", "unknown"])
    assert s["known_episodes"] == 3
    assert s["present_per_known"] == pytest.approx(1 / 3)
    assert s["present_per_applicable_known"] == 0.5
    assert s["unknown_bounds_per_all"] == [0.25, 0.5]


def test_paired_statistics_use_known_intersection_and_keep_unknown_transitions():
    r = paired(
        ["absent", "present", "unknown", "absent"],
        ["present", "absent", "present", "unknown"],
    )
    assert (r["known_pairs"], r["unknown_pairs"], r["new"], r["removed"]) == (
        2,
        2,
        1,
        1,
    )
    assert r["difference"] == 0
    assert sum(x["count"] for x in r["transitions"]) == 4
    assert paired([], [])["difference"] is None


def test_model_disagreement_is_not_automatically_adjudicated():
    assert consensus("present", "present") == "present"
    assert consensus("present", "absent") == "unknown"
    assert consensus("absent", "not_applicable") == "unknown"


def test_observed_harm_is_separate_from_present_or_plausible():
    r = {"checked": {"c": "present"}, "harm": {"c": "plausible"}}
    assert observed_harm(r, "c") == "absent"
    r["harm"]["c"] = "observed"
    assert observed_harm(r, "c") == "present"
    r["checked"]["c"] = "unknown"
    assert observed_harm(r, "c") == "unknown"


def test_agreement_handles_perfect_and_degenerate_cases():
    r = agreement(["present", "absent"] * 20, ["present", "absent"] * 20)
    assert r["raw_agreement"] == r["kappa"] == 1
    assert r["kappa_bootstrap_95"] == [1, 1]
    r = agreement(["absent"] * 20, ["absent"] * 20)
    assert r["raw_agreement"] == 1 and r["kappa"] is None


def test_only_explicit_provider_nonreviews_are_continued_without_retry():
    response = {
        "is_error": False,
        "subtype": "success",
        "structured_output": None,
        "result": "I'm still not calling StructuredOutput. This episode needs to be annotated outside this session.",
    }
    assert explicit_nonreview(response)


def test_rate_errors_remain_fatal_but_explicit_safeguard_refusals_are_missing():
    response = {"subtype": "success", "is_error": True, "result": "Usage limit reached"}
    assert not explicit_nonreview(response)
    response["result"] = "Safety classifier: cannot provide the annotation"
    assert not explicit_nonreview(response)
    response["result"] = (
        "Opus safeguards flagged this message. Claude Code can't respond to this message."
    )
    assert explicit_nonreview(response)
    response.update(
        is_error=False,
        structured_output=None,
        result="I can't submit the review again. That earlier output was stopped by a safety check. This episode has no saved result.",
    )
    assert explicit_nonreview(response)


def test_complete_census_consolidates_all_outcomes_and_unavailable_reviews(
    tmp_path, monkeypatch
):
    root = tmp_path / "reviews"
    root.mkdir()
    (root / "packets").mkdir()
    manifest = {"episodes": 12}
    write(root / "manifest.json", manifest)
    mapping = []
    for arm in ("e0", "e1", "e2"):
        for seed in range(4):
            opaque = f"{arm}-{seed}"
            p = root / "packets" / f"{opaque}.json"
            write(p, {"test": opaque})
            correct = (seed in (0, 1)) if arm != "e2" else (seed in (0, 2))
            mapping.append(
                {
                    "episode_id": opaque,
                    "stratum": "test",
                    "family": "test",
                    "arm": arm,
                    "run": arm,
                    "seed": seed,
                    "native_correct": correct,
                    "n_tokens": 8 if arm == "e2" else 10,
                    "packet_sha256": digest(p),
                }
            )
            for reviewer in ("claude", "codex"):
                job = root / "reviews" / reviewer / opaque
                attempt = job / "attempt-001"
                attempt.mkdir(parents=True)
                unavailable = reviewer == "claude" and opaque == "e2-3"
                response = {
                    "labels": {
                        c: {
                            "label": "present" if arm == "e2" else "absent",
                            "harm": "observed" if arm == "e2" else "none_supported",
                        }
                        for c in CATEGORIES
                    },
                    "goal_alignment": {"status": "satisfied" if correct else "failed"},
                }
                write(attempt / "labels.json", None if unavailable else response)
                write(attempt / "evidence_issues.json", [])
                write(
                    job / "complete.json",
                    {
                        "packet_sha256": digest(p),
                        "manifest_sha256": digest(root / "manifest.json"),
                        "attempt": "attempt-001",
                        "review_status": "unavailable" if unavailable else "evaluated",
                        "output_sha256": {
                            str(p.relative_to(job)): digest(p)
                            for p in attempt.iterdir()
                        },
                    },
                )
    write(root / "coordinator_key.json", mapping)
    monkeypatch.setattr(consolidation, "verify", lambda _: manifest)
    output = tmp_path / "output"
    consolidation.main(root, output)
    summary = json.loads((output / "summary.json").read_text())
    groups = summary["comparisons"]["test"]["e2-vs-e1"]
    assert groups["all"]["pairs"] == 4
    for name in (
        "both_successful",
        "e1_only_successful",
        "e2_only_successful",
        "both_unsuccessful",
    ):
        assert groups[name]["pairs"] == 1
    assert len(summary["unavailable_reviews"]) == 1
    assert (
        groups["all"]["reviewers"]["claude"]["checked"][CATEGORIES[0]]["unknown_pairs"]
        == 1
    )
    assert (
        groups["all"]["reviewers"]["codex"]["checked"][CATEGORIES[0]]["known_pairs"]
        == 4
    )


def test_progress_keeps_unavailable_distinct_and_rejects_foreign_episodes(
    tmp_path, monkeypatch
):
    write(tmp_path / "coordinator_key.json", [{"episode_id": "a"}, {"episode_id": "b"}])
    job = tmp_path / "reviews/claude/a"
    job.mkdir(parents=True)
    write(job / "complete.json", {"review_status": "unavailable"})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a,
            0,
            stdout=f"42 python -m trajectory_review run --root {tmp_path.name} --reviewer claude\n",
        ),
    )
    s = status(tmp_path)
    assert s["reviewers"]["claude"]["unavailable"] == 1
    assert s["reviewers"]["claude"]["evaluated"] == 0
    assert s["reviewers"]["claude"]["active_processes"] == [42]
    assert not s["reviewers"]["codex"]["stage_complete"]
    foreign = tmp_path / "reviews/codex/not-in-corpus"
    foreign.mkdir(parents=True)
    write(foreign / "complete.json", {"review_status": "evaluated"})
    with pytest.raises(ValueError, match="Unexpected episode"):
        status(tmp_path)
