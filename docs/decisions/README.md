# Decision records

One file per standing decision, numbered in the order the decisions were made.
Each record states what was decided, when, and why, and is never edited to say
something different later - a reversal gets its own record that supersedes the
old one. Minor notes and traps stay in `LAB_NOTES.md`; only decisions broad
enough to shape the pipeline or the thesis land here.

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
