"""Multi-seed batch expansion (T0.3).

Subprocess CLIs only accept --config, so a per-seed run must travel through a
materialized config file: each seed gets its own seed value and a suffixed
experiment_id (hence its own run dir), while the rest of the config — crucially
model.slug, which keeps baseline dedup working — is preserved unchanged.
"""
import yaml

from training.batch import _materialize_seed_config


def test_materialize_overrides_seed_and_id_without_mutating_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = {
        "experiment_id": "e2-multi-cosine-qwen-7b-vllm",
        "seed": 42,
        "model": {"slug": "qwen-7b"},
        "rewards": {"accuracy": {"enabled": True}},
    }
    path = _materialize_seed_config(base, 44, "e2-multi-cosine-qwen-7b-vllm-s44")

    # The base dict must not be mutated — it is reused for every other seed.
    assert base["seed"] == 42
    assert base["experiment_id"] == "e2-multi-cosine-qwen-7b-vllm"

    with open(path) as f:
        out = yaml.safe_load(f)
    assert out["seed"] == 44
    assert out["experiment_id"] == "e2-multi-cosine-qwen-7b-vllm-s44"
    assert out["model"]["slug"] == "qwen-7b"        # rest preserved
    assert out["rewards"]["accuracy"]["enabled"] is True


def test_seed_coerced_to_int(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = {"experiment_id": "e0", "model": {"slug": "qwen-7b"}}
    path = _materialize_seed_config(base, "43", "e0-s43")  # YAML/CLI may hand us a str
    with open(path) as f:
        out = yaml.safe_load(f)
    assert out["seed"] == 43 and isinstance(out["seed"], int)


# --- each seed owns a disjoint block of the seed -> question mapping ---
# Passing the raw config seed through as the dataset base made "replicates"
# overlap almost completely: at size 500, seeds 42/43/44 shared 499 of 500
# training questions, and at a 100-episode eval 99 of 100 eval questions. Eval
# is greedy, so the shared ones decode identically - pooling three such runs
# counts the same questions three times.

def _blocks(seed, size=500, n_eval=100, offsets=(100_000, 200_000)):
    from eval.agentic_eval import seed_block
    base = seed_block(seed)
    out = {"train": set(range(base, base + size))}
    for off in offsets:
        out[f"eval{off}"] = set(range(base + off, base + off + n_eval))
    return out


def test_training_sets_are_disjoint_across_seeds():
    a, b, c = (_blocks(s)["train"] for s in (42, 43, 44))
    assert not (a & b) and not (b & c) and not (a & c)


def test_eval_sets_are_disjoint_across_seeds():
    for off in ("eval100000", "eval200000"):
        a, b, c = (_blocks(s)[off] for s in (42, 43, 44))
        assert not (a & b) and not (b & c) and not (a & c)


def test_eval_splits_stay_disjoint_from_training_and_each_other():
    b = _blocks(42)
    assert not (b["train"] & b["eval100000"])
    assert not (b["train"] & b["eval200000"])
    assert not (b["eval100000"] & b["eval200000"])


def test_block_size_exceeds_every_split_offset():
    # If a seed's offset reached the next block, seed 42's shifted split would
    # land inside seed 43's questions.
    from eval.agentic_eval import _EVAL_SEED_OFFSET
    from training.config_schema import SEED_BLOCK
    assert SEED_BLOCK > 2 * _EVAL_SEED_OFFSET


def test_old_scheme_would_have_overlapped():
    # Documents the defect the block scheme fixes: adjacent seeds under the raw
    # pass-through shared all but one question.
    old = [set(range(s, s + 500)) for s in (42, 43)]
    assert len(old[0] & old[1]) == 499
