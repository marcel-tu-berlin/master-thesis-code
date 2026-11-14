from collections import Counter

import pytest
from trajectory_review.__main__ import digest, read, write
from trajectory_review.sample import import_one, select


def rows():
    return [
        {
            "stratum": f"family-{family}",
            "arm": arm,
            "seed": seed,
            "episode_id": f"{family}-{arm}-{seed}",
            "native_correct": (seed % 4 in ((0, 1) if arm == "e1" else (0, 2))),
            "n_tokens": 100 + seed,
        }
        for family in range(7)
        for arm in ("e1", "e2")
        for seed in range(200)
    ]


def test_sample_has_100_distinct_pairs_and_all_available_outcomes():
    selected, allocation = select(rows())
    assert len(selected) == len({r["episode_id"] for r in selected}) == 200
    pairs = Counter((r["stratum"], r["seed"]) for r in selected)
    assert len(pairs) == 100 and set(pairs.values()) == {2}
    assert all(cell["sample"] > 0 for s in allocation.values() for cell in s.values())
    assert (
        sorted(sum(c["sample"] for c in s.values()) for s in allocation.values())
        == [14] * 5 + [15] * 2
    )
    assert select(rows()) == (selected, allocation)


def test_selection_does_not_depend_on_tokens_or_review_availability():
    baseline, _ = select(rows())
    changed = rows()
    for index, row in enumerate(changed):
        row.update(
            n_tokens=index * 1000, model_label="present", annotation_available=index % 2
        )
    selected, _ = select(changed)
    assert [r["episode_id"] for r in selected] == [r["episode_id"] for r in baseline]


def test_rare_groups_are_kept_without_duplicating_questions():
    data = rows()
    for row in data:
        row["native_correct"] = row["seed"] != (0 if row["arm"] == "e1" else 1)
    selected, allocation = select(data)
    assert len(selected) == 200
    for cells in allocation.values():
        assert cells["e1_only_successful"]["sample"] == 1
        assert cells["e2_only_successful"]["sample"] == 1
        assert cells["both_unsuccessful"]["sample"] == 0


def test_import_retains_original_evidence_and_never_overwrites(tmp_path):
    root, source = tmp_path / "sample", tmp_path / "original"
    root.mkdir()
    source.mkdir()
    write(root / "manifest.json", {"sample": True})
    write(source / "raw.json", {"raw": "unchanged"})
    receipt = {
        "model": "gpt-6.1-sol",
        "effort": "xhigh",
        "output_sha256": {"raw.json": digest(source / "raw.json")},
    }
    write(source / "complete.json", receipt)
    row = {"episode_id": "id", "packet_sha256": "test"}
    import_one(
        root,
        row,
        "codex",
        source,
        receipt,
        {"label": "original"},
        [],
        "singleton-v1",
        "evaluated",
    )
    imported = root / "reviews/codex/id/complete.json"
    first = digest(imported)
    import_one(
        root,
        row,
        "codex",
        source,
        receipt,
        {"label": "changed"},
        [],
        "batch-v2",
        "evaluated",
    )
    assert digest(imported) == first
    assert read(root / "reviews/codex/id/imported/labels.json") == {"label": "original"}
    for name, expected in read(imported)["output_sha256"].items():
        assert digest(imported.parent / name) == expected
    (source / "raw.json").write_text("changed")
    with pytest.raises(ValueError, match="Changed original"):
        import_one(
            root,
            {**row, "episode_id": "other"},
            "codex",
            source,
            receipt,
            {},
            [],
            "singleton-v1",
            "evaluated",
        )
