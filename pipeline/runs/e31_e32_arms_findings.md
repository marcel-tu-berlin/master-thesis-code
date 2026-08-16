# e31 (E2 cosine) and e32 (E3 non-termination) vs e30 (E1) - the campaign read

Date: 2026-08-16. Both arms are single-knob copies of e30 (verified by
key-level config diff): e31 adds `token_length` (cosine, w1.0, max_len 4096),
e32 adds `non_termination` (w1.0). Same seed (42), geometry, lr 5e-5,
`token_truncate`, splits, and e0m-pinned thinking thresholds. All paired
seed-for-seed; exact McNemar throughout. Single seed, greedy eval.

## Headline

Both efficiency rewards work as designed AND produce a measurable off-target
cost, in opposite directions. Both still beat the base model.

| held_out (menu-2, n=200) | e0m | e30 (E1) | e31 (E2) | e32 (E3) |
|---|---|---|---|---|
| success | 0.595 | 1.000 | 0.930 | 0.730 |
| mean tokens (correct) | 1908 | 1541 | 487 | 1400 |
| non-termination | 0.295 | 0.000 | 0.000 | 0.000 |

| shifted acc (n=50/family) | e0m | e30 | e31 | e32 |
|---|---|---|---|---|
| click-dialog-2 | 0.78 | 0.60 | 0.56 | **0.90** |
| navigate-tree | 0.62 | 0.64 | 0.74 | 0.70 |
| click-checkboxes-transfer | 1.00 | 0.94 | 0.98 | 0.92 |

## E2 (cosine): compression is real, and it buys some of it with correctness

- vs e30 held_out: 14 losses / 0 wins, p=1.2e-4. The 7pt drop is the Wu/Yeo
  overshoot, directly observed for the first time in this project.
- Compression is massive: paired median -1130 tokens on the 186
  jointly-correct episodes (487 vs 1541 mean; -68%). Underthinking rate 1.0 -
  every correct episode sits below e0m's P10. Training mean length 2257 ->
  643; train ran 13h vs e30's 19.5h because completions shrank.
- vs e0m: 67/0 flips, p=1.4e-20. Better AND 4x shorter than base.
- Shifted: no dialog-2 rescue (0.56, p=0.8 vs e30); navigate-tree up 0.64 ->
  0.74 (p=0.06, suggestive).
- Confounds to carry: `frac_reward_zero_std` = 0.000 every bucket (e30
  ~0.5-0.7) - the cosine gives every group live gradient, so E2 = "reward +
  more gradient", per the standing caveat. KL 0.368 vs e30's 0.101 - the
  policy moved 3.6x further; some of the 7pt may be drift, not the reward's
  length preference per se. env-component slope (0.544 -> 0.934) tracked
  e30's throughout, so task learning was not visibly starved during training.

## E3 (non-termination): transfers the intended behavior, taxes the task

- vs e30 held_out: 54 losses / 0 wins, p=1.1e-16. A 27pt cost on the trained
  family relative to task-reward-only. Still beats e0m (44/17, p=7.3e-4).
- The penalty's own target saturates fast: NonTerminationPenalty contrib_l1
  0.272 -> ~0.03 by bucket 5 - non-termination is trained away in the first
  ~50 steps, after which the component is nearly inert while env learning
  continues weaker than e30's (final env contrib 0.863 vs e30 reward 0.912).
- The striking result is shifted: dialog-2 0.90 vs e30's 0.60 (18/3 flips,
  p=1.5e-3) and above e0m's 0.78; terminated 0.98. The penalty prevented -
  and reversed - the substitution collapse E1 induced on the unseen family.
  Pooled shifted 0.840, best of all four arms. E3 appears to train "act and
  terminate" as a family-general behavior rather than menu-specific policy.
- RQ2's token-side prediction (buying termination with tokens) shows up as
  absence-of-compression: e32 length ends ~2165 vs e30's 1818, correct-ep
  tokens 1400 (vs 1541) - no compression beyond noise, unlike every other
  trained arm.

## What this means for the thesis

The three-arm contrast is exactly the RQ2 structure the study was built for:
task reward alone (E1) maximizes the trained family and quietly degrades an
unseen one; the length reward (E2) converts tokens into a small correctness
tax; the termination penalty (E3) trades trained-family headroom for
robustness of the terminate-behavior across families. One seed; none of this
is replicated yet.

## Caveats

- Single seed (42), greedy eval, one environment family trained.
- Held-out ceiling in e30 means E2/E3 could only tie or lose there by
  construction; the paired flip counts (14/0, 54/0) are the honest readout.
- e31's gradient-share and KL confounds above.
- e32's held-out drop conflates the penalty's effect with ~5% weaker env
  learning during training - a lambda sweep (0.25/0.5) would separate
  penalty-induced behavior change from gradient dilution.
- e0m's shifted dialog-2 (0.78) and the probe-era 0.875 differ by draw and
  temperature; comparisons here use the paired n=50 split only.

## Timing (L4)

e31 train 13.0h + eval 2.2h (short completions); e32 train 19.4h + eval
5.6h. Batch wall 40h35m (`batch_summary_20260816_092835.md`).
