# Robustness plan — implementation notes

Implements `ROBUSTNESS_PLAN.md` (2026-06-07). All 16 items (T0.1–T2.6) are done.

The statistics and config/IO code was written test-first and verified here: **50
torch-free tests pass**. The seed threading, per-component logging, and the full
training/eval loop need the GPU box (torch + vLLM + CUDA) — see *Verify on the
box* below. Run that before trusting a real batch.

## What changed

| Item | Change | Files | Tests |
|---|---|---|---|
| T0.1/T0.2 | Seed threaded into the vLLM engine + `GRPOConfig` | `training/grpo_runner.py` | box only (runtime) |
| T0.3 | `--seeds` multi-seed batch; per-seed config materialized to `runs/_seed_configs/` | `training/batch.py` | `test_batch_seeds.py` |
| T0.4 | `seed` persisted in `eval_report.json` | `eval/report.py` | `test_report_deltas.py` |
| T0.5 | `eval_report.json` always written (status `ok`/`error`/`skipped` stub) | `eval/runner.py`, `training/batch.py` | `test_eval_stub.py` |
| T0.6 | Cosine length reward is now the default; configs consolidated to qwen-7b (both backends), linear kept as `e1-token-length-linear-ablation-*` | `configs/`, `training/rewards/__init__.py` | config validated |
| T1.1 | Seed-as-replicate hierarchical bootstrap (resample seeds, then samples) | `eval/compare.py` | `test_compare_stats.py` |
| T1.2 | Multiple-comparisons correction (BH default, holm, none) | `eval/compare.py` | `test_compare_stats.py` |
| T1.3 | Vectorized bootstrap, 10k replicates | `eval/compare.py`, `eval/metrics.py` | `test_compare_stats.py`, `test_stats_ci.py` |
| T1.4 | Wilson accuracy CI (no collapse at p=0/1) | `eval/metrics.py` | `test_stats_ci.py` |
| T2.1 | Per-component reward logging via composer + `on_log` callback | `training/rewards/compose.py`, `training/train.py` | `test_compose_metrics.py` |
| T2.2 | `reference_run` wired into deltas; token delta on `vs_reward_baseline` | `eval/report.py` | `test_report_deltas.py` |
| T2.3 | Over/under-thinking threshold recomputed inside the bootstrap | `eval/metrics.py` | `test_stats_ci.py` |
| T2.4 | Pearson p kept at full precision, shown as `.2e` | `eval/report.py`, `eval/plots.py` | `test_pearson_and_split.py` |
| T2.5 | Difficulty scatter picks the split that carries the correlation | `eval/plots.py` | `test_pearson_and_split.py` |
| T2.6 | Capability floor = 50 graded GSM8K test-tail problems (config-driven) | `eval/ood_probes.py`, 4 configs | `test_capability_floor_dataset.py` |

## New knobs

- `training.batch --seeds 42 43 44 45 46` — one run dir `<exp>-s<seed>` per seed. Baselines stay deduplicated by `model.slug`.
- `eval.compare --correction {bh,holm,none}` — default `bh`. Pairwise cells bold at `q<0.05` (or raw `p<0.05` under `none`).
- Config `eval.capability_floor_dataset` / `_hf_split` / `_limit` / `_take` — when set, the floor grades a benchmark slice instead of the 6 instruction prompts. Unset keeps the old behavior.

## Verify on the box

From `pipeline/`, with `.venv` active:

```bash
# 1. Full suite (adds the torch-dependent tests skipped on the dev machine)
python -m pytest tests/ -q

# 2. Smoke one config end to end — exercises seed threading + per-component logging
python -m training.train --config configs/e0-baseline-math-qwen-7b-vllm.yaml --smoke --overwrite --eval

# 3. Confirm T0.1: the engine now runs at the configured seed
#    PASS = engine config line reads `seed=42` AND the arg_utils.py global-seed-0 warning is gone
grep -nE "seed=|arg_utils.*seed" runs/e0-baseline-math-qwen-7b-vllm/batch_train.log 2>/dev/null \
  || python -m training.train --config configs/e0-baseline-math-qwen-7b-vllm.yaml --smoke --overwrite 2>&1 | grep -iE "seed="

# 4. Confirm T2.1: per-component metrics landed in the trainer log
python - <<'PY'
import glob, json
state = sorted(glob.glob("runs/e0-baseline-math-qwen-7b-vllm/checkpoint-*/trainer_state.json"))[-1]
keys = {k for e in json.load(open(state))["log_history"] for k in e if k.startswith("reward/")}
print("per-component keys:", sorted(keys) or "MISSING — check the on_log callback")
PY

# 5. Smoke the multi-seed harness (creates runs/<exp>-s42, -s43 + runs/_seed_configs/)
python -m training.batch configs/e0-baseline-math-qwen-7b-vllm.yaml --seeds 42 43 --smoke --train --eval
```

## Recommended batch (the plan's depth-first cell)

4 configs × 5 seeds = 20 runs, plus one shared baseline per model. Auto-compare
at the end emits the BH-corrected, hierarchical-bootstrap pairwise table.

```bash
python -m training.batch \
  configs/e0-baseline-math-qwen-7b-vllm.yaml \
  configs/e1-token-length-qwen-7b-vllm.yaml \
  configs/e2-multi-signal-qwen-7b-vllm.yaml \
  configs/e3-ablation-naive-sum-qwen-7b-vllm.yaml \
  --seeds 42 43 44 45 46 --train --eval --baseline
```

Read `runs/comparison/compare_pairwise.md` for the seed-grouped result: each
reward family pools its 5 seeds, and the bootstrap reports between-seed error.

## One scoping decision

T2.1 logs per-component reward stats (raw mean/std and L1 contribution),
including the `TokenEntropyReward` component when it is enabled — that component's
raw mean is the per-token entropy signal. A separate policy action-distribution
entropy metric is not added: it needs the policy logits, which are not available
in the reward function or the callback without a TRL-version-specific hook. If
you want it, check whether the installed TRL already logs `entropy` (recent
versions do); if not, add it where TRL computes the GRPO per-token logps.
