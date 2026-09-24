# Decision records

One file per standing decision, numbered in the order the decisions were made.
Each record states what was decided, when, and why, and is never edited to say
something different later - a reversal gets its own record that supersedes the
old one. Minor notes and traps stay in `LAB_NOTES.md`; only decisions broad
enough to shape the pipeline or the thesis land here.

Historical run paths in these records now resolve under
`pipeline/runs/archive/development-2026-09-24/`. Original records keep their text;
the [archive map](../../pipeline/runs/archive/development-2026-09-24/README.md)
explains the move. Decisions 0021/0022 and the E0-E2 campaign are the current protocol.

| # | Decision | Date |
|---|---|---|
| [0001](0001-batch-size-4-everywhere.md) | batch_size 4 everywhere | 2026-08-06 |
| [0002](0002-environment-selection.md) | Environment selection: finqa, repl, textarena removed | 2026-08-07 |
| [0003](0003-truncation-is-a-reported-outcome.md) | Truncation is a reported outcome, not a confound to remove | 2026-08-07 |
| [0004](0004-train-click-menu-2-only.md) | The campaign trains click-menu-2 only | 2026-08-13 |
| [0005](0005-reasoning-gym-is-calibration-only.md) | No poly re-runs; reasoning_gym is calibration only | 2026-08-17 |
| [0006](0006-pinned-dependency-stack.md) | The dependency stack is pinned | 2026-08-22 |
| [0007](0007-recipe-defaults-2026-08-24.md) | Recipe defaults since 2026-08-24 | 2026-08-24 |
| [0008](0008-no-unsloth.md) | No unsloth; training stays on plain TRL | 2026-09-04 |
| [0009](0009-planned-checkpoint-thirds.md) | Planned checkpoint observations at training thirds | 2026-09-10 |
| [0010](0010-final-training-recipe.md) | Final training recipe for new experiments | 2026-09-11 |
| [0011](0011-keep-liger-disabled.md) | Keep Liger disabled | 2026-09-11 |
| [0012](0012-one-update-per-rollout-batch.md) | Use one optimizer update per rollout batch | 2026-09-11 |
| [0013](0013-keep-vllm-sleep-disabled.md) | Keep vLLM sleep mode disabled | 2026-09-11 |
| [0014](0014-stop-at-environment-completion.md) | Stop training and evaluation at environment completion | 2026-09-16 |
| [0015](0015-surface-native-action-errors.md) | Surface native action errors through domain adapters | 2026-09-17 |
| [0016](0016-development-screen-success-floor.md) | Lower the development-screen success floor to 30% | 2026-09-17 |
| [0017](0017-budget-exhaustion-is-secondary.md) | Budget exhaustion warns; learning and compression govern admission | 2026-09-18 |
| [0018](0018-read-table-first-contrast.md) | Keep read-table-2 for the first E1/E2 contrast; success saturation is allowed | 2026-09-21 |
| [0019](0019-successful-response-compression.md) | Compare successful-response costs, then compress the competent E1 policy | 2026-09-23 |
| [0020](0020-relative-length-main-candidate.md) | Continue with relative successful-response length cost as the main candidate | 2026-09-24 |
| [0021](0021-final-e0-e2-from-base.md) | Separate the final E0-E2 from-base comparison from competent-policy continuation | 2026-09-24 |
| [0022](0022-e2-dose-grid-and-analysis-followups.md) | Restore the E2 dose grid and defer interpretation checks | 2026-09-24 |
