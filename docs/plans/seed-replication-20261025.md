# Five-family seed replication

Authorized 2026-10-25. The user confirmed all five families and all four
conditions, with two additional seeds per family: 30 fresh training runs and
10 E0 evaluations. Automatic advancement is authorized. Only gpu-l4 physical
GPU 1 may be used. The desktop and L4 GPU 0 are excluded. A later desktop
release requires a separate allocation decision before any queue is moved.
Decision 0029 records this scope.

## Declared allocation

| Family | Existing seed | Round 1 | Round 2 |
|---|---:|---:|---:|
| read-table-2 | 4016 | 4017 | 4018 |
| email-inbox-noscroll | 4021 | 4240 | 4241 |
| click-checkboxes-soft | 4232 | 4242 | 4243 |
| tic-tac-toe | 4230 | 4244 | 4245 |
| click-tab-2 | 4231 | 4246 | 4247 |

One serial worker completes round 1 in this table order, then round 2 in the
same order. For each family and seed: E0 -> E1 -> E2 -> matched placebo.
E0 evaluates the unchanged base. E1, E2 and placebo each start independently
from the original pinned Qwen3-1.7B with fresh LoRA. E2 and placebo use weight
0.1. No additional dose, model, environment or E3 is included.

Each seed owns its existing disjoint million-question block: 500 training
questions at offset 0 and 200 held-out questions at offset 100000. Within a
seed, all four conditions share question seeds and initial observations.
The new E0 report supplies that seed's fixed token thresholds. No old E0
reference and no training.batch --seeds shorthand may substitute for it.
Seed changes affect training randomness and generated question sets; this
replicates the full seeded experiment, not optimizer randomness in isolation.

Keep each family's frozen recipe, 300 updates, batch four, eight rollouts,
naive_sum, scale_rewards none, 4096 trajectory tokens including tool feedback,
4096 prompt tokens, 8192 context and eight turns. Keep native accepted
batch/continuation behavior. Trained conditions each have 200-question
checkpoint evaluations at updates 100/200/300; update 300 is primary.
Total new evaluation work is 100 evaluations, or 20000 episodes.

## Deployment and advancement

Operations: pipeline/runs/seed-replication-20261025-ops/.
The exact 40 run IDs and order are in queue.json. Each family has a full,
separate runtime snapshot under /workspace/master-thesis-seeds-20261025/.
Historical runtime hashes determine the sources. Checkbox's original L4
GPU 0 routing changes to GPU 1; no experiment semantics change. Table and
inbox retain their historical runtime versions. Existing run directories,
controllers, source snapshots and raw results remain frozen.

Before launch, check all five snapshots against source hashes, frozen recipes,
package/OpenEnv/MiniWoB pins, GPU identity, three historical native reset
observations, and fresh CUDA/vLLM platform detection. Record the current
kernel and driver. Before each cell, check bound sources/config/stack, free
ports/GPU, all previous same-family receipts, and the new E0 reset reference
for trained arms. Every cell has an exclusive run directory and log. A
started.json marker and queue lock prevent automatic restarts.

Advance only after zero process exit, all expected updates/captures/checkpoints,
replayed reward arithmetic (including all placebo shuffle positions), assistant
token recounts, 200 episodes at every evaluation, paired initial observations,
metric recomputation, native task audits and complete cost attempts. Hash-bind
the technical receipt. Inbox audits pool E0 and the three saved checkpoints,
retaining the approved body-reference coverage. Tabs uses the corrected native
auditor. Missing evidence, unknown audit cases, runtime errors or integrity
failures stop advancement without retry or overwrite. Performance is not a gate.

The remote controller owns automatic execution and phase state. An hourly
chat heartbeat contacts only gpu-l4, verifies controller identity and progress,
harvests completed non-weight artifacts including checkpoint-evals, verifies
hashes and updates local RUNNING.md with box time. Preserve the last verified
state if SSH fails. Notify on meaningful changes, completion, failure or a
required user decision; stay quiet on healthy unchanged checks. The remote
queue continues independently of the Mac. Monitoring/harvest needs this app,
an awake Mac and a working TU VPN connection. Retire monitoring after all
harvests, scientific reviews and the report, or a reported non-actionable blocker.

## Analysis fixed before launch

Retain every declared outcome and checkpoint. Carry forward each family's
existing paired success transitions, successful-response token comparisons,
fixed preservation/compression margins, behavior panel and measured costs.
Primary contrasts are E2 versus E1 and E2 versus matched placebo, per seed.
Placebo versus E1 and E0 versus E1 describe mechanism and specialization.
Report each seed and its joint-success coverage before family-level summaries.
Use equal seed weight when summarizing across seeds; do not treat all episodes
or checkpoints as independent training replications. With only three seeds,
training-seed uncertainty remains weakly estimated. These are exploratory
replications declared after the original family outcomes, not retrospective
confirmation or selection of favorable seeds. No cross-family pooled verdict.

Native checkbox success and exact completion remain separate. Tic-tac-toe
native outcomes remain primary, with saved-prefix cutoff sensitivity retained.
Older checkbox E2/placebo and tic-tac-toe placebo ran on home hardware; the
new matched conditions all run on L4. Original kernel/driver differences also
remain explicit. Report new L4 seeds separately as a host-consistent subset;
adding seeds does not erase the original host confounds. Unknown action labels
remain unknown, and task 8's independent label validation stays separate.

## Rough scheduling allowance

Allow roughly 3-6 hours for each E0 and 1-3 days for each trained arm including
its three evaluations. Plan around 8-12 days per family for both new seeds,
with longer allowances for tic-tac-toe and tabs. The entire serial L4 queue
is approximately 6-9 weeks before interruptions or review blockers. These are
planning ranges from prior runs, not measured predictions or a deadline.
