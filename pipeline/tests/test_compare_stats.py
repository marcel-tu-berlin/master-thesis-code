"""Seed-as-replicate statistics for eval.compare (T1.1 hierarchical bootstrap,
T1.2 BH-FDR correction, T1.3 vectorized 10k bootstrap).

The variance that flipped the fork-mask effect's sign between batches lives
*between training seeds*. A within-run bootstrap cannot see it; a two-level
bootstrap (resample seeds, then eval samples) can. With one seed it must fall
back cleanly to the single-run paired bootstrap.
"""
import numpy as np

from eval.compare import (
    _multiseed_bootstrap_delta,
    _correct_pvalues,
    _family_key,
    _strip_seed_suffix,
    _write_compare_pairwise,
)


# ---- family grouping --------------------------------------------------------

def test_family_key_strips_seed_vllm_and_slug():
    assert _family_key("e2-multi-signal-cosine-qwen-7b-vllm-s43", "qwen-7b") == "e2-multi-signal-cosine"
    assert _family_key("e0-baseline-math-qwen-7b-vllm", "qwen-7b") == "e0-baseline-math"


def test_strip_seed_suffix_only_trailing():
    assert _strip_seed_suffix("e1-token-length-cosine-qwen-7b-vllm-s100") == "e1-token-length-cosine-qwen-7b-vllm"
    assert _strip_seed_suffix("e1-no-seed-suffix") == "e1-no-seed-suffix"


# ---- two-level bootstrap ----------------------------------------------------

def test_multiseed_single_seed_each_gives_point_delta():
    a = [np.ones(50)]      # all correct
    b = [np.zeros(50)]     # all wrong
    d, lo, hi, p = _multiseed_bootstrap_delta(a, b, n_bootstrap=2000)
    assert abs(d - 1.0) < 1e-9
    assert 0.0 <= lo <= hi <= 1.0
    assert p <= 0.05       # separation is total


def test_multiseed_length_mismatch_returns_none():
    assert _multiseed_bootstrap_delta([np.ones(50)], [np.ones(49)], n_bootstrap=100) is None


def test_multiseed_between_seed_variance_widens_ci():
    rng = np.random.default_rng(0)
    N = 100
    a = [(rng.random(N) < 0.5).astype(float) for _ in range(3)]
    # Same family-mean accuracy (~0.5) but very different between-seed spread.
    b_consistent = [(rng.random(N) < 0.5).astype(float) for _ in range(3)]
    b_spread = [
        (np.arange(N) < 20).astype(float),   # 0.2
        (np.arange(N) < 50).astype(float),   # 0.5
        (np.arange(N) < 80).astype(float),   # 0.8
    ]
    _, lo1, hi1, _ = _multiseed_bootstrap_delta(a, b_consistent, n_bootstrap=4000)
    _, lo2, hi2, _ = _multiseed_bootstrap_delta(a, b_spread, n_bootstrap=4000)
    assert (hi2 - lo2) > (hi1 - lo1)   # between-seed variance must inflate the CI


# ---- multiple-comparisons correction ---------------------------------------

def test_correct_pvalues_none_is_identity():
    assert list(_correct_pvalues([0.01, 0.2], "none")) == [0.01, 0.2]


def test_correct_pvalues_bh_at_least_raw():
    p = [0.001, 0.02, 0.5, 0.9]
    q = _correct_pvalues(p, "bh")
    assert all(q[i] >= p[i] - 1e-12 for i in range(len(p)))
    assert all(qi <= 1.0 + 1e-12 for qi in q)


def test_correct_pvalues_holm_at_least_bh():
    # Holm controls FWER, BH controls FDR -> Holm is elementwise >= BH.
    p = [0.001, 0.01, 0.04, 0.3]
    bh = _correct_pvalues(p, "bh")
    holm = _correct_pvalues(p, "holm")
    assert all(holm[i] >= bh[i] - 1e-9 for i in range(len(p)))


# ---- integration: matrix groups seeds into families ------------------------

def _mkrep(exp_id, slug, seed, correct_flags):
    return {
        "experiment_id": exp_id,
        "model_slug": slug,
        "seed": seed,
        "_run_dir": f"runs/{exp_id}",
        "results": {"id_split": {"samples": [
            {"correct": bool(c), "n_tokens": 100} for c in correct_flags
        ]}},
    }


def test_pairwise_groups_seeds_into_families(tmp_path):
    reports = [
        _mkrep("e0-baseline-math-qwen-7b-vllm-s42", "qwen-7b", 42, [1] * 20 + [0] * 20),
        _mkrep("e0-baseline-math-qwen-7b-vllm-s43", "qwen-7b", 43, [1] * 22 + [0] * 18),
        _mkrep("e1-token-length-cosine-qwen-7b-vllm-s42", "qwen-7b", 42, [1] * 30 + [0] * 10),
        _mkrep("e1-token-length-cosine-qwen-7b-vllm-s43", "qwen-7b", 43, [1] * 28 + [0] * 12),
    ]
    _write_compare_pairwise(reports, str(tmp_path), correction="bh")
    md = (tmp_path / "compare_pairwise.md").read_text()
    assert "e0-baseline-math" in md
    assert "e1-token-length-cosine" in md
    assert "q=" in md          # BH q-values rendered in cells
    assert "n_seeds" in md     # per-family seed count surfaced
