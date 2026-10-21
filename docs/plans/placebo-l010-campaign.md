# Matched placebo campaign

Authorized 2026-10-18 after review of the five completed family comparisons.
Decision 0028 records the scope. Each host runs its own fixed queue. Stop after
these five cells or an unexpected runtime/integrity/audit failure; no retries.

| Host | Order | Family | Seed | Config |
|---|---:|---|---:|---|
| gpu-l4, GPU 1 | 1 | read-table-2 | 4016 | e2-placebo-l010.yaml |
| gpu-l4, GPU 1 | 2 | email-inbox-noscroll | 4021 | e2-placebo-inbox.yaml |
| gpu-l4, GPU 1 | 3 | click-tab-2 | 4231 | e2-placebo-click-tab-2.yaml |
| omarchy-home, GPU 0 | 1 | click-checkboxes-soft | 4232 | e2-placebo-click-checkboxes-soft.yaml |
| omarchy-home, GPU 0 | 2 | tic-tac-toe | 4230 | e2-placebo-tic-tac-toe.yaml |

All configs are under `pipeline/configs/`. The exact run IDs, original E2
comparators, source hashes and queues are bound in
`pipeline/runs/placebo-l010-20261018-ops/`. No E0 or E1/E2 arm is restarted.
The original E0 reports supply unchanged threshold references.

The initial home-inbox preflight reproduced seeds 4021100000 and 4021100002,
but seed 4021100001 exposed an additional email-row wrapper, subject and body
text. No training started. Inbox therefore stays on L4; checkboxes takes the
first home slot. The user fixed the final allocation at three cells on L4 and
two on home. L4 remains the primary GPU host; home is auxiliary.
Matching package/source pins alone did not
establish identical browser observations across hosts. Preserve the failed
preflight as evidence; do not relax paired-observation checks.

## Fixed experiment

Start independently from the original pinned Qwen3-1.7B, with fresh LoRA.
Keep each family's frozen E2 configuration except experiment metadata,
`successful_length.placebo: true`, and the host's OpenEnv installation path.
Weight remains 0.1. Keep 300 updates, batch four, eight rollouts, `naive_sum`,
`scale_rewards: none`, and the original optimizer/generation settings. Preserve
4096 trajectory tokens including feedback, 4096 prompt tokens, 8192 context
tokens and eight turns with the accepted batch/continuation semantics.

Evaluate 200 identical held-out questions at updates 100/200/300. Update 300
remains primary. Retain all directions of effect and all checkpoints; do not
select a favorable checkpoint. The original native outcomes remain primary,
including the checkbox native/exact-goal distinction and tic-tac-toe boundary
sensitivity. This is a subsequent exploratory mechanism comparison after the
original outcomes were known, not a retrospectively prespecified confirmation.

## Frozen deployment and checks

Use separate full pipeline snapshots per family under
`/workspace/master-thesis-placebo-20261018/` on L4 and
`~/experiments/master-thesis/placebo-20261018/` on home. Never sync over a running
cell or a completed campaign. Reconstruct historical sources by their original
manifest hashes, not by assuming the current checkout is equivalent. The only
runtime-source adaptation is physical GPU 0 routing in the two home entry points
where the original used GPU 1. Record exact before/after hashes. Use the existing
pinned environments, model cache, OpenEnv and MiniWoB installations.

Before freezing a manifest: validate the config against its E2 comparator, verify
source/package/OpenEnv/MiniWoB pins and physical GPU identity, check three native
resets against original held-out observations, and exercise CUDA/vLLM platform
detection. Check free environment/distributed ports and GPU occupancy before
each launch. Permit the known desktop voxtype process on home only.

The controller starts `training.train --observe-groups --eval` once per cell.
Its capture audit recomputes the original unshuffled costs and replays the seeded
shuffle stream across every update, including uncaptured updates. The existing
observer records unshuffled component values and the actual composed reward;
do not mislabel those raw component values as shuffled assignments. A mismatch
stops the current worker and its queue.

Before advancing: require all 300 finite optimizer records, all 416 captured
rollouts, tokenizer recounts, checkpoint 100/200/300 files, final/300 adapter
equality, all 600 evaluation episodes with paired initial observations, metric
recomputation, existing task-grounded native auditors, unchanged frozen
sources/stack/config, and four complete cost attempts. Bind the technical
receipt to the artifacts. A newly encountered unsupported audit case stops
advancement for diagnosis; never weaken a check just to advance.

## Monitoring and analysis

The host controllers own automatic advancement and update host-local live state.
An hourly chat heartbeat checks both process identities, progress and failures,
harvests completed non-weight artifacts including checkpoint evaluations,
hash-verifies them, and updates root RUNNING.md using both host timestamps.
After home completed, the user authorized stopping its dedicated HTTP service
on 2026-10-21 (Europe/Berlin) so the desktop can be shut down. Remaining
monitoring contacts L4 only and uses its box timestamp; retain the completed
home snapshots and harvested evidence without requiring home to be online.
Keep the last verified snapshot if a host is unreachable. Notify on meaningful
changes, completion, failure or required action; stay quiet on routine healthy
checks. Retire monitoring after all five scientific reviews and reports, or a
reported blocker leaves no actionable monitoring.
When L4 and all five scientific and counterpart reviews are complete, report
the combined placebo findings and recommend next steps before retiring the
heartbeat. Recommendations do not authorize further runs.

Compare placebo against E2 at the matched dose and against E1 as a secondary
reference. Carry forward paired success transitions, jointly successful token
changes, task-grounded behavior panels, costs and fixed margins from each
original protocol. Report within-seed intervals as such; no training-seed
robustness or host equivalence claim follows. A placebo does not control total
reward variance or gradient noise exactly. No new dose, seed or E3 follows
automatically from this campaign.

## Approved inbox audit recovery and tabs continuation, 2026-10-21

The user approved fixing the saved-result audit, checks and counterpart review,
then starting tabs on L4 if every gate passes and restoring hourly monitoring.
Inbox training and evaluation must not be rerun. The original controller stays
stopped. The audit-only sidecar restores the original inbox E1/E2 body-reference
scope (manifest-bound E0 plus all three saved checkpoint evaluations), with
identical initial observations and conflict rejection. It reuses the frozen
cell review and native auditor, redirects only audit outputs into an exclusive
recovery directory, and proves the original files unchanged. This corrects audit
evidence coverage; it changes no outcome, measurement or experiment semantics.

Recovery implementation and receipts live in
`pipeline/runs/placebo-l010-20261018-ops/inbox-audit-recovery-20261021/`.
A passing full technical recovery, verified canonical recovery harvest, passing
independent counterpart review, launch preflight and restored hourly monitoring
are required before the frozen tabs cell may start once on GPU 1. No old queue
is restarted. Unexpected execution or integrity failures stop work without retry.
Inbox scientific review remains required. All earlier scope limits remain.
