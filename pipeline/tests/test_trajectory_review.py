import json
import subprocess

import pytest
from trajectory_review.__main__ import (
    CATEGORIES,
    digest,
    make_packet,
    review_one,
    review_status,
    schema,
    validate_result,
    write,
)


def packet():
    episode = {
        "seed": 42,
        "correct": True,
        "reward": 1,
        "n_tokens": 123,
        "initial_observation": "Task: fill A.\nPage: A=4",
        "stop_reason": "env_done",
        "turns": [
            {
                "reasoning": "Fill A then submit.",
                "content": "",
                "n_tokens": 123,
                "tool_calls": [
                    {"name": "fill", "arguments": {"bid": "2", "text": "4"}},
                    {"name": "click", "arguments": {"bid": "3"}},
                ],
                "tool_results": [
                    {"content": "Page now: A=4"},
                    {"content": "Task done"},
                ],
            }
        ],
    }
    config = {
        "seed": 42,
        "rewards": {"weight": 0.1},
        "training": {"env_config": {"enable_fill": True, "max_turns": 8}},
    }
    return episode, make_packet(episode, "read-table-2", config, "opaque")


def result():
    return {
        "episode_id": "opaque",
        "rubric_version": "1",
        "labels": {
            name: {
                "label": "absent",
                "harm": "none_supported",
                "evidence": [],
                "available_check": None,
                "alternative_explanation": None,
                "rationale": "No supported event.",
            }
            for name in CATEGORIES
        },
        "goal_alignment": {
            "status": "satisfied",
            "evidence": [
                {"turn": 0, "action": 1, "field": "tool_result", "quote": "Task done"}
            ],
            "rationale": "Completed.",
        },
        "exploratory_notes": "",
    }


def test_packet_blinds_metadata_and_preserves_all_text_and_batches():
    raw, p = packet()
    assert not {"seed", "correct", "reward", "n_tokens", "arm", "run"} & p.keys()
    assert len(p["turns"]) == 1
    assert p["turns"][0] == {
        k: v for k, v in raw["turns"][0].items() if k != "n_tokens"
    }
    assert p["initial_observation"] == raw["initial_observation"]


def test_unknown_interface_requires_new_reference():
    raw, _ = packet()
    with pytest.raises(ValueError, match="new tools"):
        make_packet(
            raw,
            "read-table-2",
            {"training": {"env_config": {"enable_scroll": True}}},
            "id",
        )


def test_valid_labels_and_invalid_evidence_are_distinct():
    _, p = packet()
    r = result()
    assert validate_result(r, p) == []
    label = r["labels"]["insufficient_verification"]
    label.update(
        label="present",
        harm="observed",
        evidence=[
            {"turn": 8, "action": 0, "field": "tool_result", "quote": "invented"}
        ],
    )
    issues = validate_result(r, p)
    assert {i["reason"] for i in issues} == {
        "no specific available check",
        "Evidence turn is out of range",
    }
    assert (
        label["label"] == "present"
    )  # Preserve the reviewer, do not silently relabel it.


def test_wrong_quote_location_and_missing_positive_evidence_flagged():
    _, p = packet()
    r = result()
    r["goal_alignment"]["evidence"][0]["action"] = 0
    r["labels"]["action_instability"]["label"] = "present"
    issues = validate_result(r, p)
    assert len(issues) == 2
    assert {i["category"] for i in issues} == {"goal_alignment", "action_instability"}


def test_html_excerpts_match_decoded_tool_content():
    _, p = packet()
    p["turns"][0]["tool_results"][1]["content"] = '<input bid="3" checked=""/>\n  red'
    r = result()
    r["goal_alignment"]["evidence"][0]["quote"] = '<input bid="3" checked=""/> red'
    assert validate_result(r, p) == []


def test_provider_placeholder_is_not_a_semantic_unknown_annotation():
    r = result()
    r["exploratory_notes"] = (
        "This is a placeholder, not a real annotation; safety classifier."
    )
    assert review_status(r) == "unavailable"
    r["exploratory_notes"] = (
        "The feedback contains an error but recovery is successful."
    )
    assert review_status(r) == "evaluated"


@pytest.mark.parametrize("mutation", ["wrong_id", "missing_label", "bool_index"])
def test_bad_response_schema_fails(mutation):
    _, p = packet()
    r = result()
    if mutation == "wrong_id":
        r["episode_id"] = "other"
    elif mutation == "missing_label":
        del r["labels"][CATEGORIES[0]]
    else:
        r["goal_alignment"]["evidence"][0]["turn"] = True
    with pytest.raises(ValueError):
        validate_result(r, p)


def root_and_row(tmp_path):
    _, p = packet()
    (tmp_path / "packets").mkdir()
    write(tmp_path / "packets/opaque.json", p)
    write(tmp_path / "schema.json", schema())
    write(tmp_path / "manifest.json", {"version": "test"})
    (tmp_path / "rubric.md").write_text("test rubric")
    return {
        "episode_id": "opaque",
        "packet_sha256": digest(tmp_path / "packets/opaque.json"),
    }


def test_complete_results_are_immutable_and_reused_without_another_call(
    tmp_path, monkeypatch
):
    row = root_and_row(tmp_path)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        kwargs["stdout"].write(
            json.dumps(
                {"subtype": "success", "is_error": False, "structured_output": result()}
            )
        )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "test-version")
    assert review_one(tmp_path, row, "claude") == "completed"
    assert review_one(tmp_path, row, "claude") == "cached"
    assert len(calls) == 1
    labels = tmp_path / "reviews/claude/opaque/attempt-001/labels.json"
    labels.write_text("{}")
    with pytest.raises(ValueError, match="Changed completed output"):
        review_one(tmp_path, row, "claude")


def test_failure_keeps_evidence_and_requires_explicit_retry(tmp_path, monkeypatch):
    row = root_and_row(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda args, **k: subprocess.CompletedProcess(args, 1)
    )
    with pytest.raises(RuntimeError, match="exited 1"):
        review_one(tmp_path, row, "claude")
    assert (tmp_path / "reviews/claude/opaque/attempt-001/failure.json").exists()
    assert not (tmp_path / "reviews/claude/opaque/complete.json").exists()
    with pytest.raises(ValueError, match="Incomplete prior attempt"):
        review_one(tmp_path, row, "claude")
