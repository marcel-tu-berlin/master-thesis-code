"""Canonical per-slug baseline dir replaces the symlink-based dedup (Task 18).

The base-model assessment for a slug lives at one shared path,
runs/_baselines/<slug>/. runner writes it there, report reads it there, and
batch's skip check keys on it — same path everywhere, no symlinks.
"""
from eval.report import canonical_baseline_dir


def test_canonical_baseline_dir_is_per_slug():
    assert canonical_baseline_dir("runs", "qwen-7b") == "runs/_baselines/qwen-7b"
    # two different experiments, same slug -> same canonical dir
    a = canonical_baseline_dir("runs", "qwen-7b")
    b = canonical_baseline_dir("runs", "qwen-7b")
    assert a == b
