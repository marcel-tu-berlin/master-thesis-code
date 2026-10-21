# Run dose-matched placebos across the five completed families

Decided: 2026-10-18, explicitly authorized by the user.

Run one independent from-base cost-assignment placebo at weight 0.1 for each
completed E0/E1/E2 family. This matches the retained E2 dose and supersedes the
deferred 0.2 placebo allocation for this batch. The historical 0.2 template stays
deferred; no 0.2 run, dose sweep, seed replication or E3 is authorized here.

The user approved two concurrent workers, one on L4 GPU 1 and one on the home
RTX 4090, and accepted cross-host execution. Keep host/driver differences explicit
in interpretation; acceptance does not establish numerical or timing equivalence.
L4 remains the primary GPU host, with three placebos (table, inbox, tabs).
The private home desktop is auxiliary, with two (checkboxes, tic-tac-toe).
L4 GPU 0 remains excluded.

Each host automatically starts its next declared placebo after successful training,
all three checkpoint evaluations and a passing technical receipt. Stop that queue
on runtime, integrity or unrecognized audit failures. No retries or overwrites.
Negative scientific outcomes alone are valid results, not a performance gate.
Check both hosts hourly, harvest completed evidence, and report findings.

Each cell retains its original family seed, model, questions, reward definition,
300 updates, batch four, eight rollouts, 4096 trajectory / 4096 prompt / 8192
context / eight-turn limits, and accepted batch/continuation semantics. Shuffle
only the length-cost assignments within each eight-response prompt group,
including failure zeros. Keep task rewards attached to their responses.

The placebo preserves the shaped-component multiset within its own sampled
group. It does not match total-reward variance, gradient noise, or the evolving
trajectory distribution of the original E2 policy. Existing single-seed and
host limitations remain. All earlier results and reviews stay unchanged.

The fixed allocation and deployment checks are in
[the placebo protocol](../plans/placebo-l010-campaign.md).

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
