# Why does no trained arm beat e0 - audit and fix plan

Started 2026-08-11. The browsergym bs4 campaign (e27bs4/e28bs4/e29bs4) produced
no arm that beats the untrained base model on held-out accuracy (e0 0.680 vs
0.670/0.650/0.640), which breaks the premise of the efficiency study: E2/E3
cannot be read as "efficiency at held accuracy" if E1 itself buys nothing.
This plan lists every hypothesis considered, what the initial audit already
established, and the test order: code first, then environment alignment, then
model scale.

## What the 2026-08-10 audit already established

All of this is verified against TRL 1.6.0 source on the box and the harvested
`train_log.json` files, not inferred.

**Prime suspect: TRL's vLLM importance-sampling correction discards most of the
gradient, preferentially by completion length.**

1. vLLM samples the rollouts; the trainer recomputes the sampled tokens'
   logprobs with its own HF forward at the same weights. The two disagree by
   ~0.018 abs per token (numerics; `sampling/sampling_logp_difference/mean`).
2. TRL 1.6 defaults `vllm_importance_sampling_mode = "sequence_mask"` with
   `clip_max = 3.0`. The per-episode weight is `exp(signed per-token diff
   summed over every assistant token of the episode)`; a ratio above 3.0 is
   masked to exactly 0.0. No config in this repo sets the key, so every vLLM
   run trained under it.
3. `grpo_trainer.py:2613` multiplies the per-token loss by that weight, so an
   episode at weight ~0 contributes no gradient.
4. The summed drift grows with completion length, so long episodes land far
   from ratio 1.0 and lose their gradient; short episodes survive. A
   length-dependent gradient filter, inside a study about length.

Measured run means of `sampling/importance_sampling_ratio/mean`: e27bs4 0.283,
e28bs4 0.291, e29bs4 0.282, e24bs4 (poly) 0.486. Min is 0.0000 at nearly every
step; max sits at the 2.996 cap. The ratio is bad from step 1, which rules out
weight-sync staleness (that would start near 1.0 and decay).

The phenotype this explains: training EnvReward flat across all 150 steps
(first-10 mean 0.788, last-30 0.737 - the optimizer never improved its own
objective; the earlier e27 "0.375 -> 0.72 rise" was single-page-per-step noise
at batch_size 1), click-dialog-2 (~300 tokens, survives the filter) up 0.78 ->
0.84-0.88, click-menu-2 (~2000 tokens, filtered) down 0.58 -> 0.42-0.48, and
no arm beating e0 because the only headroom was on the filtered family.

Also settled on 2026-08-10:

- Eval-side bugs refuted. The LoRA adapter loads and changes behaviour (only
  2/100 held-out episodes have identical token counts between e0 and e27bs4);
  eval is greedy HF `generate` with no vLLM involved, so every eval number on
  disk is untouched by the filter.
- Train/eval prompt construction mirrors by design (same `_LEAD_IN`, same
  reset-return appended, same two tools).
- Greedy-vs-sampled objective mismatch demoted: the sampled (temp 1.0)
  training reward is flat too, so the optimizer never gained on the objective
  it actually optimizes.
- Trainer health context: `tools/failure_frequency` 0.0, grad_norm 0.05-0.09,
  KL ~0.003, entropy ~0.165.

## Hypothesis list

### A. Code / pipeline

- A1 **TIS sequence_mask filter** (prime suspect, evidence above). Decisive
  test: instrumented probe over a few real training steps logging per-episode
  completion length, task family, ISR weight, and reward. Prediction: weight
  ~0 above roughly 1000 tokens, click-menu-2 episodes near-excluded.
- A2 **Reward attribution** (reward paired with the wrong completion inside a
  group). Low probability - plausible raw means, zero tool failures - but the
  same probe dumps (seed, env reward, completion head) for a hand-check of ~10
  episodes.
- A3 Greedy-eval vs temp-1.0 objective mismatch. Demoted (see above); after
  the fix, one temp-1.0 eval documents the gap for the record.
- A4 vLLM weight staleness. Refuted (bad at step 1, no decay pattern).
- A5 Eval-side bugs (adapter load, budget, template). Refuted 2026-08-10.

### B. Optimization knobs (live even after A1 is fixed)

- B1 **Learning rate / adapter capacity too small.** lr 5e-6 with LoRA r16
  gives grad_norm ~0.05 and KL ~0.003 after 150 steps - the policy barely
  moves in distribution space. Test: short probe at lr 2e-5 on the fixed
  trainer; watch the EnvReward slope and clip/KL health.
- B2 **Exploration too weak.** Entropy ~0.165 and about half of all
  prompt-groups have zero reward variance, so many groups carry no learning
  signal. Test: rollout temperature 1.2 probe; watch `frac_reward_zero_std`
  and the reward slope.
- B3 **Mixed families dilute the batch.** click-dialog-2 groups are
  near-saturated (mostly dead under group-relative advantages) while
  click-menu-2 carried the live signal that the filter then discarded. Test:
  click-menu-2-only probe as the clean learnability measurement.

### C. Environment alignment

- C1 **Observation truncation.** The adapter caps the accessibility tree at
  `_MAX_OBS_CHARS = 2000` chars. If click-menu-2 trees exceed that, the model
  cannot see the target and success is partly luck - nothing learnable. Test:
  reset ~20 menu-2 seeds, measure tree sizes, check target visibility.
- C2 **Chance floor unknown.** A random-click scripted agent over 100
  episodes gives the floor; if it lands near the 0.58 base accuracy, base
  "skill" is mostly luck (the "solving by chance" hypothesis, tested
  directly).
- C3 **Interaction pattern.** Inspect several full episode transcripts: what
  the page looks like after the first click, whether the goal stays visible.
- C4 **Task-set fit.** click-dialog-2 at 0.78-0.88 sits at or above the top
  of the 40-80% qualification band (e26 criterion), contributing near-dead
  groups. Candidate: swap in a mid-band family from the 15 probed ones.

### D. Model scale

- D1 **Qwen3-1.7B capacity.** Eval Qwen3-4B base (no training) on the same
  seeds through the identical loop; eval-only fits the L4. If menu-2 accuracy
  jumps, the model is the binding constraint. C2 + D1 together answer "too
  dumb / solving by chance". An 8B (4-bit) variant is optional after 4B.

### E. Statistical power (cross-cutting)

n=100 single-seed paired McNemar detects only ~15-point accuracy effects at
the observed discordance rates. The training log (4800 episodes per run) is
the stronger readout, and it independently says the reward never rose - so
power is not the explanation, but any future accuracy claim needs a larger
eval n and/or 3 seeds, fixed before launch.

## Ordered execution plan

Phase gates matter: later phases are only interpretable on a fixed trainer.

**Phase 1 - trainer correctness (A):**

1. Instrumented probe on the box (~30 min GPU): subclass the trainer, run 3
   real steps of the e27bs4 recipe, dump per-episode (seed, family, assistant
   token count, ISR weight, env reward, completion head). Confirms A1
   empirically and hand-checks A2.
2. Fix the ISR handling. Candidates, decided on probe evidence:
   `vllm_importance_sampling_mode = "token_truncate"` (per-token clamp, every
   episode keeps bounded-weight gradient) vs correction off (pre-TIS
   behaviour). Wire it as an explicit config key + schema entry + unit test so
   frozen configs record it. This is a measurement change: record it in
   LAB_NOTES, name the runs it makes incomparable.
3. Verification run: 50-step E1-style probe on the fixed trainer.
   **Go/no-go: the EnvReward slope, overall and on click-menu-2
   specifically.** If reward climbs, proceed; if flat, B and C move up.

**Phase 2 - optimization knobs (B), one at a time, short probes:**

4. B1 lr probe: 2e-5 vs the 5e-6 baseline, ~30-50 steps, same seed. Read:
   reward slope, clip ratio, KL.
5. B2 exploration probe: rollout temperature 1.2, same length. Read:
   `frac_reward_zero_std`, reward slope.
6. B3 family isolation: click-menu-2-only probe. Read: does the family learn
   at all when it owns the whole batch.
7. Gate: if the Phase-1 verification run already shows a healthy slope at
   baseline knobs, compress Phase 2 to a single confirmation that the chosen
   knobs are not leaving obvious gains on the table, and move on.

**Phase 3 - environment alignment (C), cheap, interleavable with 1-2:**

8. C1 obs-size audit (CPU + env server only).
9. C2 random-click chance floor.
10. C3 episode transcript inspection.
11. C4 task-set decision on the fixed trainer's evidence: keep the pair, or
    swap click-dialog-2 for a mid-band family before the campaign re-run.

**Phase 4 - model scale (D), only if menu-2 stays flat on the fixed trainer:**

12. D1 Qwen3-4B base eval on the same seeds. Decision afterwards: stay at
    1.7B, move to 4B (training feasibility on the L4 to be checked - QLoRA +
    colocate), or change environment.

**Phase 5 - campaign re-run:**

13. Re-run E1 with the fixed trainer and chosen knobs/task set. E1 must beat
    e0 (training-reward slope plus paired eval) before E2/E3 are trained on
    top. Power plan (eval n, number of seeds) fixed before launch, per E.

## Bookkeeping owed alongside

- LAB_NOTES entry for the TIS finding and the fix (measurement change).
- Addendum in `pipeline/runs/e27bs4_e28bs4_e29bs4_findings.md`: the gradient
  the arms received was filtered by length; the "gradient asymmetry" section
  understates the distortion.
- Caveat in `pipeline/runs/e24bs4_e25bs4_pair_findings.md`: the pair trained
  at ISR ~0.49 under the same filter, and the filter suppresses exactly the
  long-completion gradient the cosine reward acts through. The seeds 43/44
  replication waits for the fix.
- taskwarrior: reconcile open tasks against this plan.
