# Final E0-E2 campaign on read-table-2

Declared: 2026-09-24, before any run in this campaign. Decision
[0021](../decisions/0021-final-e0-e2-from-base.md) fixes the independent
from-base comparison; decision [0022](../decisions/0022-e2-dose-grid-and-analysis-followups.md)
restores the full relative-cost grid and records the deferred interpretation work.
Status: prepared; no final-campaign run has started. Preparation includes no
further readiness runs. Await the user's instruction to launch.

## Conditions and fixed recipe

| Condition | Config | Initial policy | Training reward |
|---|---|---|---|
| E0 | `pipeline/configs/e0.yaml` | Original base, no adapter | No training |
| E1 | `pipeline/configs/e1.yaml` | Original base, new LoRA | Task success |
| E2 | `pipeline/configs/e2.yaml` | Original base, new LoRA | Task success - 0.1 * relative successful-response cost |
| E2, weak | `pipeline/configs/e2-l005.yaml` | Original base, new LoRA | Task success - 0.05 * relative successful-response cost |
| E2, strong | `pipeline/configs/e2-l020.yaml` | Original base, new LoRA | Task success - 0.2 * relative successful-response cost |
| E2 placebo | `pipeline/configs/e2-placebo-l020.yaml` | Original base, new LoRA | Task success - 0.2 * within-group uniformly shuffled relative cost |

The new dose/placebo run IDs begin with `e2l005`, `e2l020` and `e2placebol020`,
respectively, so existing short plot labels and filenames stay distinct.

The weight grid is fixed before results: E1 supplies zero, E2 uses 0.05/0.1/0.2.
The placebo shuffles only the shaped component, including its failure zeros;
it preserves the within-group cost multiset, not total-reward variance or
gradient noise. Against E1 only the shaped component differs; between E2 doses
only its weight differs, and against the matching placebo only its assignment
differs. No optimizer, initialization or data differences accompany the reward.

Use Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, bf16,
LoRA rank 16 / alpha 32, and the pinned dependency/OpenEnv stack. E1/E2 use the
same 300-update recipe: batch size 4, eight rollouts per question, micro-batch 1,
learning rate 0.00005, constant schedule with 10% warmup, weight decay 0.1,
KL coefficient 0, DAPO, one update per fresh rollout batch, `naive_sum` and
`scale_rewards: none`. Keep vLLM colocation at 0.3, token-truncate importance
sampling, Liger disabled and vLLM sleep disabled.

The family is BrowserGym MiniWoB `read-table-2`, with click/fill/noop and the
qualified native action-error feedback. Training and evaluation stop at
environment completion. Both have eight turns and a 4096-token whole-trajectory
budget, including tool observations; the length reward counts assistant tokens
only, including reasoning and tool-call arguments. No warm start, reward ramp,
non-termination penalty or other condition is mixed into this comparison.

## Fresh allocation and observations

Training seeds are 4016, 4017 and 4018. Execute all conditions at 4016 first,
review their technical integrity, then repeat every trained condition at 4017
and 4018 with matching E0 coverage. Replicate null/adverse outcomes too. There
are five trained conditions per seed: 15 training runs and three E0 evaluations,
with 45 trained checkpoint evaluations. These are actual research runs, not
readiness batches. The first seed supplies an informative result but not a
training-seed uncertainty estimate.

Each seed owns a disjoint block. Arms within a seed answer identical questions;
different training seeds also have different question sets. For seed 4016:

- Training: 500 questions at seeds 4016000000-4016000499.
- `held_out_table`: 200 questions at seeds 4016100000-4016100199 (offset 100000).
- E0: one greedy evaluation on those 200 held-out questions.
- Every trained condition: independent training from base; greedy evaluation at updates
  100, 200 and 300 on the same 200 questions. Update 300 is primary; earlier
  observations describe the trajectory, never select a best checkpoint.

E0 supplies the fixed P10/P75 token thresholds through
`runs/e0-read-table-2-s4016/eval_report.json`. E0 computes those thresholds from
its own samples; all E1/E2 observations read that same report. The reference
must remain unchanged. This threshold calibration does not tune the reward or
change the primary paired token measure. Do not extend the sample after seeing
results, reuse development questions, or add a shifted family to this run.
For seeds 4017/4018 use the same offsets and counts in their own million-seed
blocks. Before their launches, copy each active config and change only `seed`,
the experiment ID suffix and `eval.reference_report` to that seed's E0 report.
Do not use `training.batch --seeds` for this campaign: that helper does not
rewrite the reference-report path. Each seed must use its own E0 thresholds.
Shifted-family evaluation remains a separately declared extension after the
first read-table-2 result, with its own task/reward qualification.

## Execution after launch is requested

Read the existing BrowserGym operations checklist in LAB_NOTES.md. Sync the
whole current pipeline, preserve `runs/` and `configs/archive/`, and record the
Git commit, config hashes and stack with the launch evidence. Record each live
phase in RUNNING.md with box time. Use the existing CLI in this order, on the
GPU server from `/workspace/master-thesis-code/pipeline`:

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m eval.runner --config configs/e0.yaml --base-model
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m training.train --config configs/e1.yaml --observe-groups --eval
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m training.train --config configs/e2.yaml --observe-groups --eval
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m training.train --config configs/e2-l005.yaml --observe-groups --eval
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m training.train --config configs/e2-l020.yaml --observe-groups --eval
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m training.train --config configs/e2-placebo-l020.yaml --observe-groups --eval
```

E0 must finish before scheduled E1/E2 evaluation can load its reference. Do not
pass E0 to the training batch runner. Retain the existing bounded group
observations, train logs, adapter checkpoints, reports and episode traces. Do
not add parameter-gradient history. Harvest and inspect each phase for missing
outputs, nonfinite values, runtime errors and protocol mismatches. These are
result reviews, not a new readiness campaign. Success saturation alone does not
block the declared comparison. Do not overwrite failed or completed runs.

The old development watcher is retired. Operational review/delivery can use
the existing tools after launch; no new automation framework is a research
prerequisite. Nothing is queued by this preparation.

## Declared analysis

Carry forward the first contrast's success-preservation margin and efficiency
target. Pair each E2 dose versus E1 at update 300 by question seed and confirm identical
initial observations. Report success counts, Wilson intervals, gain/loss
transitions, exact McNemar, both-correct count, paired absolute token changes,
and the paired percentage change `100 * (E2 tokens / E1 tokens - 1)` on jointly
correct episodes. With no jointly correct pairs, efficiency is undefined.

Success preservation: the one-sided 95% exact binomial upper bound on
`P(E1 correct and E2 wrong)` must be at most 5 percentage points. This
conservative bound does not offset losses with gains. At 200 questions, at most
four regressions satisfy it. Not meeting the bound is inconclusive unless the
data establish harm; do not turn imprecision into a harmful-result claim.

Meaningful compression: the upper bound of the median paired percentage-change
95% percentile-bootstrap interval must be at most -10%. Use 20,000 resamples
and RNG seed 0. A median reduction of at least 10% with an upper bound below
zero is promising, but does not establish the full target. Token savings from
failed responses do not count as compression of successful behavior. Compare
E0 versus E1 separately to describe task training; do not attribute that change
to the E2 reward.

Retain the off-target panel: wrong submissions, voluntary no-tool stopping,
invalid actions, repeated actions and stop transitions. For the first four,
report one-sided exact bounds on newly introduced events at alpha 0.05/4,
against the existing 5-percentage-point margin. Treat budget exhaustion and
non-termination as separate descriptive warnings, not a hard compression gate.
The legacy frozen-threshold short-response flag is not proof that a valid short
solution is harmful. Inspect all evaluation trajectories against the visible
table and stops, and audit recorded training groups/reward arithmetic. Inspect
all success regressions plus the existing seed-17 stratified masked sample of
efficient correct controls; disclose any prior knowledge of aggregate results.

Report active-group fractions, zero-gradient updates and measured training and
inference costs. More active groups are part of the shaping intervention; this
contrast alone does not isolate length assignment from gradient activation.
The declared 0.2 placebo supplies the cost-assignment comparison against real
E2 at 0.2; also report it against E1.

Report every weight and every seed. Episode-level intervals within a seed do
not measure training-seed variation; checkpoints are repeated observations,
not independent replications. A positive, negative, null or inconclusive result
completes a condition. Review integrity before matched seed replication, without
requiring a positive result. Non-termination training is deferred until E0-E2
are finished and reviewed.

## Interpretation follow-ups, explicitly deferred

These are analysis tasks, not readiness or launch blockers (user, 2026-09-24):

- Quantify paired minimum detectable effects/precision using development
  discordance and the realized seed coverage. Report per-seed results and
  training-seed uncertainty separately from within-seed episode intervals.
  The individual bounds above do not establish simultaneous coverage across
  all doses and endpoints. Address multiplicity in the cross-condition verdict;
  do not select the best dose and quote its unadjusted interval as confirmatory.
  Keep the fixed 200-question allocation and margins. A later calculation is
  a sensitivity analysis, not prospective power or permission to extend n.
- Validate off-target labels against task-grounded trajectories when interpreting
  results, including efficient correct controls. Use arm-masked review where
  possible and disclose prior aggregate knowledge. Until validated, early
  termination and preceding-action counts are descriptive proxies. Resolve the
  equivalent/shifted/indeterminate verdict and any length-conditioned sensitivity
  analysis there; a nonsignificant difference does not establish equivalence.

## Cost accounting from the first run

`costs.jsonl` is an append-only phase ledger. Each attempt has UTC start/end
records, a unique ID, host, PID, visible GPU allocation, monotonic wall seconds,
status and allocated GPU hours. Sum end records once per attempt, including
failed attempts when reporting actual study expenditure. An unmatched start
(e.g. SIGKILL or power loss) means incomplete cost, not zero. Report it explicitly.

Training is timed after config validation/freezing, before model loading through
adapter saving. Evaluation is a separate phase. Scheduled checkpoint timing
includes its fresh worker's loading, environment startup, episodes, reporting
and process teardown. Direct E0 timing covers evaluation function entry through
report writing, including model loading and environment shutdown. Small CLI
startup and schedule-preflight overheads are outside these phase boundaries.
Scheduled workers write costs to their canonical result directory even on
failure. Their nested temporary timer is not published or counted twice.

Every completed episode is flushed with `episode_wall_seconds` (reset, policy,
tools and scoring, excluding output-file writes) and `inference_wall_seconds`
(sum of policy-call wall time: prompt preparation, HF generation and response
parsing, excluding environment calls). Generated IDs are transferred to CPU
before parsing, so asynchronous CUDA work has completed within this boundary.
The report retains individual timings and split totals with timed-sample counts.
First-use overhead is retained; no unrecorded warmup episodes are added.

These are wall-time/allocation measurements, not GPU kernel time, energy or
money. The current CLI assigns one L4; GPU-hours = wall seconds / 3600.
Unknown allocation is recorded as null. Compare latency on the same hardware,
engine and settings, paired by question. Report all-episode deployment cost
and successful-pair savings separately so failure-induced savings are visible.

At interpretation, calculate the break-even count as training cost in seconds
divided by mean per-episode inference seconds saved, for an explicitly named
reference (E0 for specialization, E1 for shaping). Use the same cost basis in
numerator and denominator, report total expert-training cost separately from
incremental E2-versus-E1 training cost, and keep research evaluation expenditure
separate. If measured savings are nonpositive, there is no finite break-even.
Earlier records with missing timings remain unknown, never zero-filled.
