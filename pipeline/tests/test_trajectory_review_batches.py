import copy
import json
import subprocess
import sys
import time

import pytest
from trajectory_review import batches
from trajectory_review.__main__ import CATEGORIES, digest, validate_result, write
from trajectory_review.batch_evidence import check


def episode():
    return {
        "episode_id": "test",
        "family": "read-table-2",
        "initial_observation": "Goal: fill four",
        "turns": [
            {
                "reasoning": "Fill four",
                "content": "",
                "tool_calls": [],
                "tool_results": [{"content": "Task complete"}],
            }
        ],
    }


def response():
    return {
        "reviews": [
            {
                "episode_id": "test",
                "labels": ["absent"] * 6,
                "details": [],
                "goal_alignment": {
                    "status": "satisfied",
                    "rationale": "Completed.",
                    "evidence": [
                        {
                            "turn": 0,
                            "action": 0,
                            "field": "tool_result",
                            "quote": "Task complete",
                        }
                    ],
                },
                "exploratory_notes": "",
            }
        ]
    }


def test_pack_is_lossless_and_reduces_repeated_text():
    data = [episode(), copy.deepcopy(episode())]
    data[1]["episode_id"] = "second"
    for row in data:
        row["initial_observation"] = "Exact repeated page with spaces.\n" * 30
    packed = batches.pack(data)
    assert batches.unpack(packed) == data
    assert len(json.dumps(packed)) < len(json.dumps(data))
    assert len(packed["texts"]) == 1
    with pytest.raises(ValueError, match="collides"):
        batches.pack([{"text_ref": "original source value"}])


def test_normalization_preserves_judgments_and_marks_unelicited_rationales():
    data = batches.normalize(response(), [episode()])["test"]
    assert data["evidence_issues"] == []
    assert data["annotation_protocol"] == "batch-v2"
    assert data["review"]["labels"][CATEGORIES[0]]["label"] == "absent"
    assert (
        "coordinator format note"
        in data["review"]["labels"][CATEGORIES[0]]["rationale"]
    )


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "foreign", "short", "missing_detail"]
)
def test_batch_requires_every_episode_and_all_six_labels(mutation):
    r = response()
    if mutation == "duplicate":
        r["reviews"].append(copy.deepcopy(r["reviews"][0]))
    elif mutation == "missing":
        r["reviews"] = []
    elif mutation == "foreign":
        r["reviews"][0]["episode_id"] = "other"
    elif mutation == "short":
        r["reviews"][0]["labels"].pop()
    else:
        r["reviews"][0]["labels"][0] = "present"
    with pytest.raises(ValueError):
        batches.normalize(r, [episode()])


def test_evidence_is_checked_against_its_own_episode():
    r = response()
    r["reviews"][0]["goal_alignment"]["evidence"][0]["quote"] = "Other episode"
    issues = batches.normalize(r, [episode()])["test"]["evidence_issues"]
    assert issues[0]["category"] == "goal_alignment"


def test_compact_call_quote_matches_the_prompt_without_changing_arguments():
    packet = episode()
    packet["turns"][0]["tool_calls"] = [
        {"arguments": {"bid": "18", "text": "a b"}, "name": "fill"}
    ]
    result = batches.normalize(response(), [packet])["test"]["review"]
    evidence = result["goal_alignment"]["evidence"][0]
    evidence.update(
        field="tool_call",
        quote='{"arguments":{"bid":"18","text":"a b"},"name":"fill"}',
    )
    assert validate_result(result, packet)
    assert check(result, packet) == []
    evidence["quote"] = evidence["quote"].replace('"18"', '"19"')
    assert check(result, packet)
    evidence["quote"] = '{"arguments":{"bid":"18","text":"ab"},"name":"fill"}'
    assert check(result, packet)
    for quote in ("", " \t\n"):
        evidence["quote"] = quote
        assert check(result, packet)


def test_expired_shared_window_starts_no_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(batches, "verify_batch", lambda root: {})
    write(
        tmp_path / "window.json",
        {"start_epoch": time.time() - 3601, "duration_seconds": 3600},
    )
    with pytest.raises(ValueError, match="One-hour"):
        batches.run(tmp_path, "codex", 1, 0)
    assert not (tmp_path / "results").exists()


def test_inflight_request_is_terminated_at_deadline(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        batches.invoke(
            [sys.executable, "-c", "import time;time.sleep(10)"],
            "",
            tmp_path,
            time.time() + 0.15,
        )


def test_prepare_reuses_existing_refusals_without_resubmission(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "packets").mkdir()
    p = source / "packets/test.json"
    write(p, episode())
    write(source / "manifest.json", {})
    write(
        source / "coordinator_key.json",
        [{"episode_id": "test", "packet_sha256": digest(p)}],
    )
    job = source / "reviews/claude/test"
    job.mkdir(parents=True)
    write(
        job / "complete.json",
        {
            "packet_sha256": digest(p),
            "review_status": "unavailable",
            "output_sha256": {},
        },
    )
    monkeypatch.setattr(batches, "verify", lambda root: {})
    output = tmp_path / "batch"
    batches.prepare(source, output, 16)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["plans"]["claude"] == {"retained": ["test"], "jobs": []}
    assert manifest["plans"]["codex"]["jobs"][0]["episode_ids"] == ["test"]
