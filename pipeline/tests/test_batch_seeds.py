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
