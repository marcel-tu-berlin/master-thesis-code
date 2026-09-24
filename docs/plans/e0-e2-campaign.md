# Final E0-E2 campaign on read-table-2

Declared: 2026-09-24, before any run in this campaign. Decision
[0021](../decisions/0021-final-e0-e2-from-base.md) fixes the independent
from-base comparison and decision 0020 fixes the relative reward at weight 0.1.
Status: prepared; no final-campaign run has started. Preparation includes no
further readiness runs. Await the user's instruction to launch.

## Conditions and fixed recipe

| Condition | Config | Initial policy | Training reward |
|---|---|---|---|
| E0 | `pipeline/configs/e0.yaml` | Original base, no adapter | No training |
| E1 | `pipeline/configs/e1.yaml` | Original base, new LoRA | Task success |
| E2 | `pipeline/configs/e2.yaml` | Original base, new LoRA | Task success - 0.1 * relative successful-response cost |

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

Seed 4016 is unused by the archived development campaigns. It owns one fresh
seed block, shared across these three conditions:

- Training: 500 questions at seeds 4016000000-4016000499.
- `held_out_table`: 200 questions at seeds 4016100000-4016100199 (offset 100000).
- E0: one greedy evaluation on those 200 held-out questions.
- E1/E2: independent training from base; greedy evaluation at updates
  100, 200 and 300 on the same 200 questions. Update 300 is primary; earlier
  observations describe the trajectory, never select a best checkpoint.

E0 supplies the fixed P10/P75 token thresholds through
`runs/e0-read-table-2-s4016/eval_report.json`. E0 computes those thresholds from
its own samples; all E1/E2 observations read that same report. The reference
must remain unchanged. This threshold calibration does not tune the reward or
change the primary paired token measure. Do not extend the sample after seeing
results, reuse development questions, or add a shifted family to this run.

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
target. Pair E2 versus E1 at update 300 by question seed and confirm identical
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
Any such attribution needs a separately declared placebo comparison.

This is one training seed and one family. Episode-level intervals do not measure
training-seed variation; checkpoints are repeated observations, not independent
replications. A positive, negative, null or inconclusive result completes this
first campaign. Review it before matched seed replication or broader conditions.
Non-termination training is deferred until E0-E2 are finished and reviewed.
