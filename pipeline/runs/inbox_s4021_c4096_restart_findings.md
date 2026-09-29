# Inbox restart at the retained agent budgets

Reviewed: 2026-09-29. Protocol correction: decision 0025 and
`docs/plans/inbox-family-extension.md`. Source/config commit: `44207d2`.

## Why the restart was necessary

The family qualification increased the trajectory allowance to 5,120 tokens,
context to 9,216 and turns to 12, then carried those settings into the inbox
campaign without explicit approval for a budget change. The user rejected that
change and authorized a complete restart, including E0, at the retained
seed-4016 E0/E1/E2 budgets.

All nine active configs now match the actual three retained frozen configs:
8,192 context tokens, 4,096 prompt tokens, 4,096 training and evaluation
trajectory tokens, a 4,096 successful-length reference maximum and eight turns.
The comparison is recorded in
`inbox-campaign-s4021-c4096-ops/budget-reference-audit.json`.
Tool feedback remains charged to the trajectory allowance. Assistant-token
measurement, reward code, model precision and pinned runtime are unchanged.

The original read-table trio remains retained. The old inbox E0 and failed E1
remain separate, superseded-budget diagnostics; their 18 and 11 hash-bound
review inputs were checked unchanged. Never use the old E0 as the corrected
baseline. Fresh `-c4096` IDs avoid overwriting evidence. Seed 4021 and question
allocation are retained, with no outcome-driven sample selection. Observation
and prompt differences from the read-table study remain explicit in the plan.

## Feasibility evidence

Run: `inbox-native-check-s4021-c4096`, physical GPU 1 only. Three ordinary
`training.train` updates used four questions by eight rollouts, micro-batch one,
bf16, vLLM reservation 0.24, original token/turn limits and the task-only reward.
No smoke overrides, readiness capture, replay, Liger or vLLM sleep were used.
The first batch's questions and initial observations match the failed inbox E1
attempt exactly. This is a disposable engineering check, not a campaign arm.

| Update | Successes in each eight-rollout group | Trajectories at 4,096 | Gradient norm | Learning rate |
| --- | --- | --- | --- | --- |
| 1 | 8, 7, 0, 7 | 6 | 0.043202 | 0 |
| 2 | 8, 0, 8, 0 | 9 | 0 | 0.00005 |
| 3 | 1, 8, 4, 0 | 8 | 0.049100 | 0.00005 |

All three optimizer steps completed with finite metrics and no memory/runtime
failure. The step-3 and final adapter bytes match. Update 2 had no within-group
task-reward variation and no learning signal. The short run resolves warmup to
one step, while the campaign has 30 warmup steps; update 3 supplies a real,
nonzero-gradient update. No diagnostic weights are used in E0, E1 or E2.
This check does not guarantee memory safety for every later batch or establish
the family's learning curve.

The complete run cost was 1,650.623 seconds, or 0.458506 allocated GPU-hours.
Logs, all 96 captured trajectories, stack/config evidence and costs are harvested;
adapters remain on the box. `inbox-native-check-s4021-c4096/review.json` binds the
reviewed files and records integrity pass. The review normalizes the post-run
collector's `pipeline/../.venv/bin/python` spelling and compares frozen YAML
values independently of key order; neither difference changes the experiment.

A CPU recount of eight previously saved successful scripted inbox paths covers
reply, forward, trash and star. Their longest minimal path, including rendered
actions and tool feedback, is 1,320 tokens and four turns. This supports task
feasibility at the original allowance, not a policy success estimate or an
exhaustive bound on all generated tasks. Evidence is in
`inbox-campaign-s4021-c4096-ops/oracle-fit.json`.

## Verification and relaunch

The budget regression test failed on all three inbox configs before correction
and passed all nine active configs afterward. The project gate passed: 485 tests,
7 skipped, and all 17 setup checks; formatting, lint and types passed. CI for
`44207d2` passed. The controller/watcher check rejects all 15 deferred launch
indices and verifies the corrected scope, phase transitions, exits, queue
acceptance handling, deduplication and retry without launching GPU work.

The full pipeline was synchronized. Preflight verified all 41 runtime/lockfile
hashes, the pinned stack, clean OpenEnv/MiniWoB clones, an idle GPU 1, a free
port 8000, the served task and three unused run IDs. The corrected E0 launched
at 21:01:30 UTC as `e0-email-inbox-noscroll-s4021-c4096` from the original base
model with a 4,096-token evaluation allowance. Live state belongs in RUNNING.md.

Operations are in `inbox-campaign-s4021-c4096-ops/`. The new dedicated watcher
submitted a continuation for the live E0; the old watcher remains unloaded.
Queue acceptance is verified; consumption is checked on the subsequent turn.
Each completed run must be harvested and reviewed before the next launch.
Only corrected E0 -> E1 -> E2 weight 0.1 at seed 4021 is authorized. Stop after
E2 and the initial comparison review, or for an unresolved runtime/integrity
failure or new scientific issue. No further dose, seed, placebo or E3 is admitted.
