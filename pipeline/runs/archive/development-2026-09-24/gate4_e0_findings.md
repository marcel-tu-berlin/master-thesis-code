# Gate 4 E0 calibration findings

E0 completed and was harvested on 2026-09-14. This is a quality review of the
base-model calibration reference, not a Gate 4 pass. Gate 4 is decided from the
final E1 pilot checkpoint after training.

## Provenance and completeness

The non-smoke base-model report records `checkpoint: null`, `checkpoint_step: 0`,
and seed 4002. The frozen run config exactly matches
`configs/readiness/g4-e0.yaml` (SHA-256
`09370f6e159ec5d09588bcfd9952a4989062671c26f28ba06dda6ee6789f8ce9`). Its
launch source digest is
`9cb8ac80183cb2329b2ec789d61d3349110b8e1af823ed4f2eb6439436a062cd`.
The environment stamp pins OpenEnv at `024eedc90305cc8bd7a5b44f44d1b987102e957b`,
which matches its recorded pin.

All 160 required JSONL records are present: 100 menu episodes and 20 each for
dialog, tree, and transfer. Each split has the exact contiguous seed sequence
`4002000000 + split_offset + index`, with no duplicates. Rewards and token counts
are finite. Every record has a nonempty initial observation, saved turns, and a
one-to-one action and tool-result trace. Recomputed counts, success, token means,
mean steps, non-termination, stop reasons, and serialized report samples all match
`eval_report.json`. `load_reference_thresholds` returns finite per-split thresholds:
menu 943.0/2401.5, dialog 230.6/772.5, tree 254.7/1187.0, and transfer
251.5/684.5 (underthinking/overthinking).

## Calibration rates

Wilson 95% intervals are shown for each rate. Budget endings include only
`max_turns` and `hit_generation_cap`.

| Family | Success | Budget ending |
|---|---|---|
| menu | 67/100 = 67.0% [57.3%, 75.4%] | 4/100 = 4.0% [1.6%, 9.8%] |
| dialog | 18/20 = 90.0% [69.9%, 97.2%] | 0/20 = 0.0% [0.0%, 16.1%] |
| tree | 12/20 = 60.0% [38.7%, 78.1%] | 0/20 = 0.0% [0.0%, 16.1%] |
| transfer | 18/20 = 90.0% [69.9%, 97.2%] | 0/20 = 0.0% [0.0%, 16.1%] |

Menu stop reasons were 79 `env_done`, 17 `no_tool_call`, and 4
`hit_generation_cap`. Dialog had 20 `env_done`; tree had 12 `env_done` and 8
`no_tool_call`; transfer had 18 `env_done` and 2 `no_tool_call`.

## Trace review and verdict

I reviewed a fixed stratified sample from saved records: menu seeds 4002300000
(correct), 4002300023 (budget ending), and 4002300018 (voluntary stop); dialog
4002400001 (correct) and 4002400000 (incorrect); tree 4002500002 (correct) and
4002500000 (voluntary stop); transfer 4002600000 (correct) and 4002600007
(voluntary stop). The initial observations, selected actions, and tool results
support their stored stop reasons and outcomes. The eval log contains no traceback
or infrastructure failure.

Quality verdict: accepted as the E0 calibration reference for E1 threshold
pinning. The menu rate and the four observed budget endings are evidence, not an
implementation failure. The shifted 20-episode slices are calibration references
only; their wide intervals do not support powered cross-family comparisons.

## E1 handoff

The E0 review left bounded late-group observation as the remaining prerequisite.
That qualification is recorded in `gate4_observer_findings.md`; E1 can use this
reference after the observer and deployment checks pass.
