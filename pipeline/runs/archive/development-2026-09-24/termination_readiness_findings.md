# Episode-boundary requalification

Decision: [0014](../../docs/decisions/0014-stop-at-environment-completion.md).
Status: E1, E2 and E3 pass; harvested and reviewed on 2026-09-17.

The deployed controller passes every contract, including the two-hour diagnostic
ceiling. Recorded runtime totals 5128.837 seconds (85.5 minutes). Its generic
next-phase label is `gate4_e1_pilot`; decision 0014 limits the actual next research
step to the bounded read-table-2 screen. The earlier menu-family rejection stands.

The production trainer now stops each episode at environment completion,
including terminal failures, and does not generate an extra assistant turn
after the final allowed tool iteration. Other batch slots continue. Terminal
feedback remains available to the audit and reward functions but is not fed
into another model generation. Historical runs and snapshots are unchanged.

Five CPU regression cases failed against the unchanged native loop and passed
after the boundary guards. They cover mixed finished/live slots with and without
sampling log probabilities, same-turn extra calls, the last allowed turn and
overlong feedback. The installed-stack check uses the actual production wrapper,
pinned tokenizer, TRL training template and native tool-suffix formatting with
32 independent environments. Only unfinished slots regenerate; IDs, masks and
log probabilities remain aligned. A nonterminal voluntary-stop path is identical
to unmodified native TRL. Native source drift is rejected by a method hash.

The local gate passed: formatting, lint, types, 422 tests and six optional-stack
skips, plus the setup harness. The new native test passed on the box with all
six termination tests. A read-only native review found no concrete defects.
It inspected code and the upstream method; the execution evidence comes from
the separate checks above.

The box is a deployed runtime tree, without a local CPU test environment. As in
the previous readiness review, the deployed controller consumes the passing Mac
gate only after matching its source digest to the deployed source. Native
trainer replay and runtime-stack checks still execute on the box. The fresh
controller report is `readiness_envdone_v1.json`; old Gate 3 reports are not
overwritten or reinterpreted as passes of this correction.

Fresh configs are `configs/readiness/g3-envdone-e1/e2/e3.yaml`, in that order.
Each retains seed 4001, three updates and 4 x 8 geometry. Readiness capture and
the existing group observer record all three small diagnostic batches. Each
arm requires its controller review before the next starts. The cumulative
diagnostic ceiling remains two hours. A pass permits the bounded harder-family
screen; it does not restore click-menu-2's failed research qualification.

## E1 review - 2026-09-17

`readiness-g3-e1-s4001-envdone-v1` completed in 1683.999 seconds, including
checkpoint reload and the four-episode evaluation (2/4 correct). The deployed
controller passes E1 and admits E2. Its arithmetic replay passes reward/dose,
advantage, mask, DAPO loss and selected-log-probability gradient checks.
All three recorded gradient norms are finite; the second update is nonzero and
the saved adapter parameters change. The first and third batches have no
within-group task-reward variation, so their zero gradients are expected.

All 96 observed episodes satisfy the eight-assistant-turn boundary and the
recorded E3/composed reward checks. All 63 completed episodes end at tool
feedback; none contains a rejected post-completion call. The independent
artifact check and input hashes are retained under
`termination-envdone-v1-review/`. This checks recorded structure together with
the native-loop regressions, not a per-action timestamp of environment completion.

The first controller invocation rejected an interpreter path spelling mismatch
(`pipeline/../.venv/bin/python` versus the captured `.venv/bin/python`). Running
the controller through the exact captured absolute executable passes the
unchanged stack comparison. No capture, source hash or gate was relaxed.

## E2 review - 2026-09-17

`readiness-g3-e2-s4001-envdone-v1` completed in 1786.850 seconds, including
checkpoint reload and four-episode evaluation (2/4 correct). The deployed
controller passes E2 and admits E3. Native replay covers the actual nonzero
loss/gradient and all eight captured zero/half/full/placebo dose cases.
All three optimizer-gradient norms are finite and nonzero, and saved adapter
parameters change. All twelve observed groups have token-length reward variation.

The 96 observed episodes pass the same structural and reward checks as E1.
All 63 completed episodes end at tool feedback without rejected post-completion
calls. E1 and E2 together used 3470.849 seconds of the two-hour diagnostic limit.
These four-episode evaluations verify checkpoint execution and reporting;
their accuracy is not evidence for a treatment effect.

## E3 review - 2026-09-17

`readiness-g3-e3-s4001-envdone-v1` completed in 1657.988 seconds, including
checkpoint reload and four-episode evaluation (2/4 correct). The deployed
controller passes E3. All three optimizer-gradient norms are finite and nonzero,
saved parameters change, and the E3 penalty varies in five of twelve observed
groups. All 96 observed episodes pass the structural and reward checks; all 63
completed episodes end at tool feedback without rejected post-completion calls.

Native replay passes the 32 actual loss shards and eight captured dose/placebo
cases. The independent scripted E3 cases at lambda 0, 0.5 and 1 also pass,
including budget exhaustion, voluntary stopping and completion precedence.
The largest actual E3 scalar-loss discrepancy is 2.85e-9 and the largest
selected-log-probability gradient discrepancy is 1.36e-12; the scripted E3
maximum loss discrepancy is 5.43e-8. These are below the declared 2e-5 loss
and 5e-7 gradient tolerances.

## Combined verdict and limits

All three captures match the deployed source and pinned stack, use the accepted
4 x 8 geometry and record `env_done_or_budget_v1`. The combined observation
audit covers 288 episodes, including 189 completed episodes. The source digest is
`bccf65f81204384b566fada6813aa9ac0f9dd45be2eb983447157fffbbc41aa3`.
`readiness_envdone_v1.json` contains the final controller verdict;
`termination-envdone-v1-review/audit.py` and `all.json` retain the additional
structural/reward check and input hashes. Checkpoints remain on the box.

The fresh local gate passed formatting, lint, types, 422 tests with six
optional-stack skips, and all 17 setup-harness cases. The earlier native
tokenizer/tool-loop checks remain applicable because no training source changed
during these passes. The deployed native replays supply the stack-dependent
arithmetic evidence.

These three-update runs establish the corrected execution and measurement
contract on the tested stack. Their short warmup, four training instances and
four-episode evaluations do not establish long-run stability, treatment effects
or harder-family suitability. The read-table-2 screen must still exercise its
opt-in fill tool through real sampled training trajectories before a pilot.
