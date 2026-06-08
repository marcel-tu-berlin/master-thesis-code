# Pipeline robustness + statistical-significance plan

Derived from an audit of the two GRPO reward-signal batches (`2026-05-27-signals-baseline-sweep`, `2026-06-05-vllm-signals-rerun`) against this source tree, 2026-06-07. All line numbers are against the files as read on that date — re-check before editing.

**Goal:** make the experiments robust and statistically defensible, depth-first — i.e. seed-replicate the existing GSM8K cell with upgraded statistics before adding new signals.

---

## Tier 0 — blockers (do before re-running anything)

| # | Change | File:line | Effort |
|---|---|---|---|
| T0.1 | **Thread seed into vLLM.** Add `load_kwargs['seed'] = int(config.get('seed',42))` in the vLLM-only kwargs block. Today the engine runs at seed 0 regardless of `seed:42` (train log shows `arg_utils.py:1390` global-seed-0 + engine `seed=0`). This is the cause of the cross-batch baseline shift. | `training/grpo_runner.py:38-42` | small |
| T0.2 | Belt-and-suspenders: add `seed=int(self.config.get('seed',42))` to the `GRPOConfig(...)` constructor (TRL trainer + colocate sampler seed). | `training/grpo_runner.py:64-87` | trivial |
| T0.3 | **Multi-seed harness.** Add `--seeds 42 43 44 ...` to `batch.py`; per base config, freeze a temp config with `cfg['seed']=s` and `cfg['experiment_id']=f'{id}-s{s}'` → unique run dir per seed (run dir keys only on `experiment_id`, so distinct ids are required). Keep baseline dedup slug-keyed (seeds share one base-model baseline). Subprocess CLIs only take `--config`, so the override must go through the config, not argv. | `batch.py:340-414`, `batch.py:137`, `batch.py:63-74` | large |
| T0.4 | Persist the training seed so compare can group by it: `report['seed'] = config.get('seed')`. | `eval/report.py:81-82` | trivial |
| T0.5 | **Always emit `eval_report.json`.** Today it is all-or-nothing — written only if the eval subprocess completes (`runner.py:79-81`); a train-only/failed/crashed eval leaves no JSON and the run silently vanishes from auto-compare. This is why batch-1 has no JSON and its pairwise stats aren't regenerable. Emit a stub `{experiment_id, model_slug, results:{}, status:'error'/'skipped'}` on partial/failed eval. Per-sample `samples` vectors are already always written when the file exists (`report.py:50-53`). | `eval/runner.py:53-84`, `batch.py:439-463` | medium |
| T0.6 | Rebuild `e2`/`e3` on the **cosine** length reward (config only: `token_length: {shape: cosine, max_len: 256, r_correct_short: 1.0, r_correct_long: 0.5, r_wrong_short: -1.0, r_wrong_long: -0.5, ...}`). They currently use the linear penalty, which collapses under z-scoring (`e2` over-compressed to 76 tok / 0.66 acc for this reason). | new configs | trivial |

> **Verify T0.1:** after the fix, the `arg_utils.py:1390` warning disappears and the engine config line reads `seed=42`.

---

## Tier 1 — statistical significance

| # | Change | File:line | Effort |
|---|---|---|---|
| T1.1 | **Seed as replicate (two-level / hierarchical bootstrap).** New `_multiseed_bootstrap_delta`: per iteration, resample SEEDS with replacement (between-run level), then within each chosen seed resample eval samples with replacement (within-run level), pool, recompute the family-level delta. Group rows by `(reward_family, model_slug)` (reuse `_reward_family` at `compare.py:122`); seed = replicate. Positional pairing already holds (eval shuffle fixed at `ood_probes.py:268` seed=42, independent of training seed) so the existing index-resample (`compare.py:400-402`) is the inner loop. When only one seed exists, fall back to the current single-run bootstrap and label the table `n_seeds=1`. | `eval/compare.py` (new fn; data dep on T0.4) | large |
| T1.2 | **Multiple-comparisons correction.** The matrix runs N·(N-1)/2 unique tests (15 at N=6, 21 at N=7), each bolded at raw p<0.05 → family-wise false-positive ~54-66%. Collect all p's, dedupe to unordered pairs (matrix is symmetric, delta(A,B)=-delta(B,A)), apply Benjamini-Hochberg via `scipy.stats.false_discovery_control(pvals, method='bh')` (scipy already imported at `metrics.py:5`), bold on **q<0.05**, render `p=… (q=…)` in each cell. Add `--correction {none,bh,holm}` (default bh) in `main()`. | `eval/compare.py:459-473`, CLI near `:514-525` | medium |
| T1.3 | **Bump `n_bootstrap` 2000 → 10k-20k + vectorize.** MC error at p=0.05/B=2000 is ±0.0049 → borderline cells flip bold/not-bold between seeds. Vectorize: `idx = rng.integers(0,n,size=(B,n)); boot = a[idx].mean(1) - b[idx].mean(1)` (one allocation, makes 10-20k trivial). Pass explicitly at the call site so it isn't silently defaulted. Mirror the bump in the per-metric CI. Optionally report `mc_err = sqrt(p(1-p)/B)` on borderline cells. | `eval/compare.py:369,400-402,465`; `eval/metrics.py:46` | small |
| T1.4 | **Wilson (or Clopper-Pearson) accuracy CI.** Replace `_bootstrap_ci(corrects)` — percentile-bootstrap-on-binary collapses to `[1.0,1.0]` at p=1 (the capability_floor 6/6 case; true Wilson ≈ [0.61,1.0]). Add `_proportion_ci(n_correct, n, ci=0.95)`: Wilson `z=1.96, phat=k/n, denom=1+z²/n, center=(phat+z²/2n)/denom, half=(z/denom)·sqrt(phat(1-phat)/n + z²/4n²)`. Clopper-Pearson (`scipy.stats.beta.ppf`) is the conservative exact alternative. | `eval/metrics.py:152-153` | small |

> **The lever that matters:** T1.2-T1.4 are necessary but **not sufficient**. Within-run bootstrapping cannot recover between-training-seed variance — the variance that flipped the fork-mask effect's sign between batches. T0.3 + T1.1 make it *measurable*; only actually running ≥3 (ideally 5) seeds *delivers* significance.

---

## Tier 2 — correctness / observability

| # | Change | File:line | Effort |
|---|---|---|---|
| T2.1 | **Per-component + policy-entropy logging.** TRL sees one composed reward function, so only `rewards/<composer>/{mean,std}` is logged — individual accuracy/format/length/entropy contributions and policy entropy are invisible. Have the composer stash per-component raw mean/std + post-z-score weighted contribution on `self` (expose `pop_step_metrics()`); drain it from a `TrainerCallback` (extend the `train.py:21-29` pattern) via `trainer.log({...})`. Keeps the advantage math unchanged. | `rewards/compose.py:58,77,90-100` + callback `train.py:21-29` + wire `grpo_runner.py:92,95` | medium |
| T2.2 | **Wire `reference_run` into the deltas.** Currently `eval.reference_run` only sets thresholds (`ood_probes.py:107-128`); `report.py` rediscovers the baseline via a fragile single-e0 heuristic (`_find_baseline`, returns None when 0 or >1 e0 siblings exist) and only `vs_reward_baseline` (e0) carries a trained-vs-trained delta, with no token delta. Prefer `config['eval']['reference_run']` for both delta blocks (precedence `baseline_id > reference_run > heuristic`); add `delta_mean_tokens` to `vs_reward_baseline`. | `eval/report.py:96-131,138-172` | small |
| T2.3 | **Threshold-conditional rate CIs understate uncertainty.** The over/under-thinking percentile threshold is estimated once and frozen, then only the rate is bootstrapped. Recompute the threshold *inside* each bootstrap replicate (skip when an absolute override is set) so threshold variance propagates. Also fix the same degenerate-at-0/1 binary-bootstrap issue (Wilson floor). | `eval/metrics.py:131-150` | medium |
| T2.4 | **Pearson p-value underflow.** Rounded to 6 dp before display → a strongly significant correlation prints as literal `0.0`. Keep full precision; format `:.2e`. | `eval/report.py:42`, `eval/plots.py:190-191` | trivial |
| T2.5 | Pearson(difficulty,length) only exists on the MATH split (only MATH carries difficulty levels); the plot reads `id_split` only → empty for GSM8K-trained runs. Have the plot pick whichever split has non-None `pearson_difficulty_length`. | `eval/plots.py:175-191` | small |
| T2.6 | **Harder capability floor** (n≥50, non-trivial probes) so it can catch a regression instead of saturating at acc=1.0. | probe data/config | small |

---

## Recommended next batch (depth-first)

- **Freeze:** vLLM backend + T0.1/T0.2 seed fix applied.
- **Cell:** `{e0, e1-cosine, e2-cosine, e3-cosine}` × **5 seeds** (e.g. 42-46) = 20 runs.
  - confirms the cosine Pareto win (0.795 acc / 127 tok) with between-seed error bars — currently n=1,
  - settles **e2 vs e3** (advantage-weighted vs naive-sum / the DIET question) with real replicate variance — both prior batches favored e3 on one seed,
  - runs e2/e3 on the fixed cosine length reward (T0.6) so the comparison is clean.
- **Report:** mean ± across-seed SD, hierarchical-bootstrap CIs (T1.1), BH-FDR-corrected pairwise table (T1.2), Wilson accuracy CIs (T1.4).

Free win to bank regardless: `frac_reward_zero_std=0.40` on the baseline — 40% of GRPO groups get zero gradient under pure accuracy reward; continuous shaping signals recover them (~0). Independent argument for shaping signals.

---

## Resolved: token_entropy semantics (no code change needed)

Read `training/rewards/token_entropy.py`. It is a **reward** on per-token entropy (`reward_scale · mean_t H_t`), **not** a Wang-style gradient mask. `fork_mask_top_frac` restricts which tokens enter the reward mean, not the gradient. Positive `reward_scale=0.1` ⇒ it **rewards higher entropy** = an exploration/diversity regularizer, not an effort/overthinking penalty. Entropy comes from a fresh forward pass through the *current* (post-update) policy (off-policy proxy; vLLM `max_logprobs=0` is irrelevant) and re-tokenizes when `completion_ids` are absent. **Decision (2026-06-07):** keep the positive sign, frame entropy as an exploration signal. The cross-batch fork-mask sign flip is consistent with single-seed noise on an off-policy proxy — no entropy claim until ≥3-seed replication.

cosine_length / token_length verified correct: four corners `correct-short +1.0 > correct-long +0.5 > wrong-long -0.5 > wrong-short -1.0`; length clamp `min(prog,1)` prevents the cosine wrapping past `max_len`. The linear penalty's α + cosine *schedule* cancel under per-group z-scoring (`compose.py:77`) — confirmed; an α-grid would change nothing.
