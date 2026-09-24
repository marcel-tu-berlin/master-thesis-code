# e33-e36 lambda sweep: E2 and E3 at w0.25 / w0.5, read against e30 / e31 / e32

Date: 2026-08-21. Four new arms on the identical e30 recipe (seed 42, geometry,
lr 5e-5, naive_sum, splits): e33/e34 add `token_length` (cosine) at w0.25/w0.5,
e35/e36 add `non_termination` at w0.25/w0.5. Together with e31 (E2 w1.0), e32
(E3 w1.0) and e30 (lambda 0) this is a 3-point dose-response per reward. All
seven arms run `compose_method: naive_sum`, so the weight is a true penalty
coefficient in every arm. Every comparison below is paired seed-for-seed
(identical 200 held-out and 150 shifted question seeds, verified by
intersection); exact McNemar on flips, sign test + paired bootstrap on the
jointly-correct token diffs. Single seed, greedy eval. Batch wall 74h24m
(`batch_summary_20260820_135932.md`); e31/e32 numbers from their own campaign
(`e31_e32_arms_findings.md`).

## Headline

E2's compression has no dose-response in 0.25-1.0: it saturates at the
smallest lambda. E3 is dose-responsive on its target metric (termination),
with the held-out tax minimized at w0.5, not w0.25.

## held_out (menu-2, n=200; e30 = 1.000 acc, 1695 median tokens on correct)

| arm | acc | flips vs e30 (L/W) | McNemar p | paired median dtok (both-correct) | sign p | non-term |
|---|---|---|---|---|---|---|
| e33 E2 w0.25 | 0.870 | 26/0 | <1e-4 | -1273 [-1298,-1224] | <1e-4 | 0.000 |
| e34 E2 w0.5 | 0.930 | 14/0 | 1e-4 | -1234 [-1235,-1198] | <1e-4 | 0.000 |
| e31 E2 w1.0 | 0.930 | 14/0 | 1e-4 | -1130 [-1161,-986] | <1e-4 | 0.000 |
| e35 E3 w0.25 | 0.815 | 37/0 | <1e-4 | -444 [-444,-373] | <1e-4 | 0.060 |
| e36 E3 w0.5 | 0.930 | 14/0 | 1e-4 | -151 [-151,-14] | 0.067 | 0.000 |
| e32 E3 w1.0 | 0.730 | 54/0 | <1e-4 | -70 [-151,+61] | 0.68 | 0.000 |

e30 is at ceiling, so arms can only tie or lose here; the flip counts are the
honest readout.

## shifted (3 unseen families, n=150 pooled; e30 = 0.727, non-term 0.187)

| arm | pooled acc | flips (L/W) | McNemar p | non-term | dialog-2 acc |
|---|---|---|---|---|---|
| e33 E2 w0.25 | 0.807 | 9/21 | 0.043 | 0.147 | 0.86 (p=0.011) |
| e34 E2 w0.5 | 0.700 | 11/7 | 0.48 | 0.240 | 0.50 (ns) |
| e31 E2 w1.0 | 0.760 | 11/16 | 0.44 | 0.120 | 0.56 (ns) |
| e35 E3 w0.25 | 0.633 | 23/9 | 0.020 | 0.227 | 0.44 (ns) |
| e36 E3 w0.5 | 0.727 | 15/15 | 1.00 | 0.107 | 0.56 (ns) |
| e32 E3 w1.0 | 0.840 | 6/23 | 0.002 | 0.073 | 0.90 (p=0.001) |

Per-family dialog-2 flips are vs e30's 0.60 on the same 50 seeds.
navigate-tree and checkboxes-transfer move nowhere significant in any arm.

## E2: a plateau, not a dose-response

- Compression saturates at the smallest lambda: -1273 / -1234 / -1130 median
  tokens (all -67% to -75%, all p<1e-4). Turning the coefficient up 4x buys
  nothing - the cosine shape, not its weight, sets the equilibrium length.
- The accuracy cost is not monotone: w0.5 and w1.0 are identical (14/0
  flips), w0.25 is worst (26/0). A weaker length signal did not mean a
  smaller overshoot. With one seed the 0.870-vs-0.930 gap (12 extra losses)
  is not distinguishable from noise, but the direction rules out "smaller
  lambda, safer arm" as a working rule.
- e33's shifted read (pooled 0.807, best E2; dialog-2 0.86) is the one
  lambda-dependent E2 result, and it is anomalous: neither w0.5 nor w1.0
  shows any rescue. Single-seed noise is the default reading.
- The gradient confound applies to every E2 lambda equally:
  `frac_reward_zero_std` = 0.000 in all three arms vs 0.600 in e30. Even at
  w0.25 the cosine gives every prompt-group live gradient. There is no lambda
  small enough to escape this confound; only naive_sum-vs-composer or a
  matched-gradient design could.

## E3: dose-responsive on target, non-monotone on tax

- The target metric responds monotonically to lambda on shifted: non-term
  0.227 / 0.107 / 0.073 (e30: 0.187), dialog-2 0.44 / 0.56 / 0.90. The
  substitution rescue that e32 showed is a genuine dose effect - it needs
  full strength; at w0.5 the arm merely matches e30.
- The held-out tax is NOT monotone: 37 / 14 / 54 losses. w0.5 is the sweet
  spot - same 14-loss cost as E2's best arms, non-term 0.000 held-out and
  0.107 shifted, at the price of losing the w1.0 robustness gain.
- This breaks the gradient-dilution story from `e31_e32_arms_findings.md`
  (which predicted smaller lambda -> smaller tax). w0.25 taxes MORE than
  w0.5 and also degrades shifted (0.633, significantly below e30) and even
  hits the generation cap on 12 held-out episodes - the only arm in the
  campaign with held-out non-termination. A weak penalty appears to be worse
  than none or full: enough to perturb training, not enough to install the
  terminate behavior. One seed; treat as a hypothesis.
- `frac_reward_zero_std` 0.51-0.53 across E3 arms vs e30's 0.60 - gradient
  exposure comparable, the E3 contrast stays clean on this axis at every
  lambda.
- Token side: mild compression at w0.25 (-444), fading to nothing by w1.0
  (-70 ns) - consistent with the penalty buying termination with tokens, as
  in the e32 read.

## What this means for the thesis (4.3 / 7.2)

The "several penalty strengths" requirement is met with a 3-point sweep per
reward. The shape of the two dose-responses is itself the finding: the cosine
reward acts like a switch (any tested lambda buys full compression and a
7-13pt tax), the termination penalty like a dial (target behavior scales with
lambda, cost does not scale down with it). Practically: E2 has no
lambda-tuning story in this range; E3's operating point is a real choice -
w0.5 for cheap on-distribution safety, w1.0 for off-distribution robustness
at a 27pt on-distribution price.

## Caveats

- Single seed (42), greedy eval, one trained family. The non-monotone points
  (e33's held-out dip, e33's dialog-2 rescue, e35's across-the-board
  degradation) are exactly the kind of result seed replication exists for.
- Held-out ceiling in e30 (see above).
- E2-vs-e30 remains "reward + more gradient" at every lambda (zero-std
  panel); E3-vs-e30 does not.
- e35's 12 held-out cap-hits mean its token numbers mix truncations into the
  wrong-episode pool; its correct-episode numbers are unaffected.

## Timing (L4)

e33 13.6h train (compression accelerates it), e34 ~14h, e35/e36 ~23-25h each;
batch wall 74h24m including evals.
