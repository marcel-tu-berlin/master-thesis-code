# e30 gate: E1 (menu-only, fixed trainer) vs e0m base - PASSED

Date: 2026-08-14. Runs: `e0m-browsergym-base-menu-qwen3-1_7b` (base model,
no adapter) and `e30-browsergym-e1-menu-qwen3-1_7b` (task-success-only GRPO,
click-menu-2, 150 steps, lr 5e-5, `token_truncate` ISR). Identical splits and
seeds; e30's thinking-rate thresholds pinned to e0m's report. Gate protocol
from `docs/plans/no-arm-beats-e0-audit.md`: e30 must beat e0m on training-reward
slope plus paired held-out eval before the E2/E3 arms run on the recipe.

## Verdict

Both halves pass, decisively. The first trained arm of the project that beats
its own base model - the "no arm beats e0" symptom is resolved by the recipe
(menu-only task set + 5e-5 + `token_truncate`), on one seed.

## Held-out (click-menu-2, n=200, paired seed-for-seed)

| | e0m | e30 |
|---|---|---|
| success | 0.595 [0.525, 0.661] | 1.000 [0.981, 1.000] |
| non-termination | 0.295 | 0.000 [0.000, 0.019] |
| mean tokens (correct eps) | 1908 | 1541 [1470, 1610] |
| mean steps | 3.09 | 2.61 |
| stop reasons | 141 env_done / 48 no_tool_call / 11 cap | 200 env_done |

- Discordant pairs: e30-only-correct 81, e0m-only-correct 0. Exact McNemar
  two-sided p = 8.3e-25. Zero regressions.
- On the 119 jointly-correct episodes: paired median token diff -643, paired
  mean -442.6, e30 shorter on 82/119. Compression is incidental - this arm has
  no length reward - and sets the bar E2 must beat to claim the cosine adds
  anything.

## Training slope (150 steps, buckets of 10)

reward 0.562 0.591 0.569 0.634 0.841 0.803 0.691 0.769 0.772 0.803 0.762
0.847 0.884 0.875 0.912. KL 0.001 -> 0.101 (settling, kl_beta 0.001);
grad_norm flat. Both mid-run watch-items resolved on their own: completion
mean length peaked at 2750 (bucket 4) and fell to 1818 by step 150;
trajectory-cap clipping peaked 0.19 and ended 0.016. frac_reward_zero_std
rose from ~0.5 to ~0.725 as the task saturated.

## Shifted (untrained families, n=50 each, paired)

| family | e0m acc | e30 acc | e0m term | e30 term | McNemar p |
|---|---|---|---|---|---|
| click-dialog-2 | 0.780 | 0.600 | 1.000 | 0.740 | 0.064 |
| navigate-tree | 0.620 | 0.640 | 0.640 | 0.760 | 1.0 |
| click-checkboxes-transfer | 1.000 | 0.940 | 1.000 | 0.940 | 0.25 |

Suggestive substitution cost on click-dialog-2, not significant at n=50: the
trained arm stops terminating on the family it never saw (terminated 1.000 ->
0.740; 27 of the 28 shifted failures beyond the cap hit are no_tool_call).
This is the RQ2 substitution read the C4 decision moved dialog-2 to the
shifted split to expose, and it is the sharpest thing E3 can act on: the
non-termination penalty prices exactly this failure.

## Caveats for reading E2/E3 against e30

- Single seed (42), greedy eval. Nothing here is seed-replicated yet.
- Held-out accuracy is at ceiling. E2/E3 cannot beat 1.000 - the informative
  axes are token counts, training dynamics, and the shifted split.
- e30 ends with ~72% zero-std prompt groups. A length reward varies where env
  reward is constant, so the E2 arm will train on more live groups - check
  `frac_reward_zero_std` in both logs before attributing any difference to the
  reward's shape (the "reward plus more gradient" confound, CLAUDE.md).
- The e27bs4/e28bs4/e29bs4 numbers do not carry over: two-task mix, lr 5e-6,
  and the `sequence_mask` ISR filter (run-mean ISR 0.28) that starved long
  completions of gradient.

## Timing (L4)

e0m eval 3h50m (350 episodes). e30 train 19h30m (150 steps, ~492 s/it minus
startup), e30 eval 5h12m. Full gate 28h32m.
