# First read-table-2 efficiency contrast

Declared: 2026-09-21, before the new held-out evaluations and E2 training.
Authority: decision 0018 and the user's instruction to continue through a first
informative result, then consider more families or environments.

## Fixed experiment

Use seed 4009, Qwen3-1.7B revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e,
and the qualified runtime and stack from readiness_feedback_v2.json. Reuse
table-pilot-e1-s4009 as E1. E2 starts from that same original base, with the
same 500 training questions, 4 x 8 geometry, 300 updates, 100/200/300 saved
observations, optimizer, learning rate, decoding, LoRA and budgets. Its only
reward change is enabling the existing token_length component at weight 0.4.
E2 applies from the first rollout; the existing optimizer warmup is unchanged.

The raw cosine endpoints remain correct [0.5, 1] and wrong [-1, -0.5], with
max_len 4096. At weight 0.4, even the worst possible future placebo assignment
leaves successful total reward at least 0.6 and unsuccessful reward at most
0.4. This is below the requalified E2 execution weight 1.0. Check actual reward
arithmetic and mean-only centering locally; do not repeat GPU readiness solely
for this lower weight while source and stack match. Other weights are deferred.

Freeze 200 fresh same-family questions at seeds 4009100000-4009100199
(offset 100000), disjoint from training offsets 0-499 and development offset
300000. Name this split held_out_table. All arms and saved observations use
the same instances, greedy decoding, eight turns and the 4096-token trajectory
budget. Do not extend this sample after seeing outcomes. These instances are
the test set for this declared first contrast; future tuning must use development
data and requires fresh confirmation data for a new claim.

Pin the existing E0 development reference as the metric ruler. A frozen copy
maps its development_table samples to held_out_table solely to supply the same
P10/P75 thresholds. Record the source hash and alias; this is calibration data,
not an E0 result on the new questions. Evaluate E0 on all new questions too.
Use separate output IDs; never modify completed pilot reports or protocols.

## Ordered execution

1. table-first-e0-s4009: 200 held-out base-model episodes, then technical review.
2. table-first-e1-s4009: evaluate the saved pilot checkpoints at 100/200/300 on
   those 200 questions each, then technical review. Record original checkpoint
   paths, steps and hashes; no E1 retraining. Success saturation does not block.
3. table-first-e2-l040-s4009: 300 updates from the base, bounded group observations
   at 1,100,200,291-300, then all three scheduled 200-question evaluations.
4. Harvest and review the paired result. Step 300 is primary; 100/200 are declared
   sensitivity observations. Deliver the finding and stop before expansion.

The launcher binds all configs, the decision, this plan, the calibration ruler,
qualified source/stack, pilot review, and existing adapter hashes. Each stage
requires a source-bound technical review of its predecessor. Remote state stays
in table-diagnostic-s4008-ops/state.json, feeding the existing persistent watcher.
No new scheduler, automatic overwrite or model-phase retry is introduced.

## Declared analysis and precision

The first result is one trained seed, not training-seed replication. Pair by
question seed and verify identical initial observations. Use the existing paired
statistics for success transitions, exact McNemar, absolute token differences
and stop transitions. Report all success counts and intervals, both-correct
sample size, and inference/training time. Do not attribute E0-to-E1 savings to E2.

Success preservation means no more than 5 percentage points of loss. Use a
conservative paired test: count questions correct under E1 and wrong under E2,
and require the one-sided 95% exact binomial upper bound on that regression
probability to be at most 5%. This bounds net accuracy loss without giving
credit for compensating gains. Failure to meet the bound is inconclusive unless
other evidence establishes harm; it is not automatically a harmful result.
At n=200 this accepts at most four regressions (upper bound 4.52%). Under an
independent 1% regression probability its admission probability is 94.8%; at
2% it is 62.9%. These are conditional episode-level planning calculations,
not power for variation across training seeds.

The meaningful efficiency target is a 10% median paired token reduction on
jointly correct episodes: 100 * (E2 tokens / E1 tokens - 1). Report its paired
percentile-bootstrap 95% interval using 20,000 resamples and RNG seed 0, plus
the existing absolute-token comparison. An upper interval bound at or below
-10% supports that target; a point reduction of at least 10% with an upper bound
below zero is promising but does not establish the full target. With no jointly
correct pairs the efficiency comparison is undefined. Do not count more failures
or shortened failed trajectories as an efficiency improvement. There is no
credible E2 effect-size estimate yet; do not invent power for its token effect.

Report unconditional wrong submission, voluntary no-tool stopping, episodes
with invalid actions, and episodes with repeated actions. For each, bound newly
introduced events with one-sided exact binomial bounds at alpha 0.05/4. A bound
at or below 5 percentage points supports no material increase in that measured
indicator; otherwise label it unresolved or shifted if supported. Budget endings
remain a separate E3 warning. These are bounded indicators, not a safety claim.
Keep the legacy frozen-threshold short-response flag visible but do not call a
valid short solution harmful underthinking.

Audit every evaluation trajectory against visible table values, rewards and
stops, and the captured training trajectories/reward arithmetic. Inspect all
success regressions and a seed-17 stratified masked sample with efficient correct
controls; disclose if the reviewer already knows aggregate outcomes. Compare
active-group fractions between E1 and E2. Added correctness information and
activated groups are part of this intervention; the contrast alone does not
isolate cost assignment. The planned placebo remains required for that claim.

An interpretable negative, null or inconclusive outcome completes this tranche.
Freeze any next experiment separately after reviewing the result.
