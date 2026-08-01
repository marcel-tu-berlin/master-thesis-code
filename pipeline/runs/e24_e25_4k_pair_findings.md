# e24 / e25: the cosine length reward at a 4096 completion budget

Harvested 2026-07-30. `polynomial_equations`, Qwen3-1.7B + LoRA, seed 42, 300 GRPO
steps, 100 held-out eval episodes. The two configs differ only in
`rewards.token_length`: off in e24, `enabled: true, weight: 16.0, max_len: 4096` in
e25. Both evals ran `seed_base=100042, max_new_tokens=4096`, and eval iterates
`seed_base + i`, so **both arms answered the same 100 questions** and every
comparison below is paired.

## Headline

| | e24 (w0) | e25 (cosine w16) |
|---|---|---|
| accuracy | 0.910 [0.838, 0.952] | 0.930 [0.863, 0.966] |
| correct-episode mean tokens | 876.6 | 839.9 |
| correct-episode median tokens | 667.0 | 510.0 |
| episodes at the 4096 cap | 8 | 7 |
| wrong episodes below the cap | 1 | 0 |
| underthinking / overthinking | 0.110 / 0.187 | 0.108 / 0.194 |

The cosine arm is weakly better on both axes at once - shorter and no less accurate.
That is the first time in this campaign the length reward has moved anything in the
intended direction. It is **not** statistically significant.

## Paired statistics (91 questions both arms answered correctly)

| statistic | value | 95% CI (paired bootstrap, 20k) |
|---|---|---|
| difference of medians | -157.0 | [-232.0, +3.0] |
| median of per-question differences | -37.0 | [-61.0, 0.0] |
| mean of per-question differences | -54.3 | [-151.0, +43.8] |
| e25 shorter / longer / tied | 52 / 34 / 5 | sign test p = 0.066 |

Every interval touches or crosses zero and the sign test misses 0.05. The direction is
consistent across all four statistics, which the earlier weight sweep never managed,
but a single seed at n=100 cannot call it.

Quartiles of the same 91 questions show where the compression lives:

| | Q1 | median | Q3 |
|---|---|---|---|
| e24 | 356 | 667 | 1121 |
| e25 | 343 | 510 | 999 |

Q1 is flat (-13) while the median and Q3 move (-157, -122). The reward acts on the
long half of the distribution and leaves already-short episodes alone - which is what
a correctness-gated cosine should do.

Note the gap between "difference of medians" (-157) and "median of differences" (-37):
completion lengths cluster, only 24 of 91 pairs land within 50 tokens of each other,
and the two statistics answer different questions. The -157 figure is the one
comparable to the earlier w-sweep numbers; the -37 figure is the typical per-question
effect. Quoting only the first overstates the result.

## Accuracy did not pay for it

McNemar on the discordant pairs: 0 questions that e24 got right and e25 got wrong, 2
the other way (exact two-sided p = 0.50). e25's correct set is a strict superset of
e24's on this seed. No sign of the Wu/Yeo overshoot the config was watching for, at
this weight and budget.

## What the raised cap changed (e12 -> e24, same arm, cap 2048 -> 4096)

| | e12 (cap 2048) | e24 (cap 4096) |
|---|---|---|
| accuracy | 0.840 | 0.910 |
| wrong episodes at cap | 16/16 (100%) | 8/9 (89%) |
| correct-ep median | 567 | 667 |

Doubling the budget converted 7 of 16 "wrong" episodes into correct ones without any
reward change. This is the truncation confound, measured: the e9-e21 null was scored
on rollouts cut mid-reasoning and labelled wrong, then fed to the cosine wrong-branch.

Two consequences worth keeping:

1. **Almost every remaining error is a length failure, not a reasoning failure.** e24
   has exactly one wrong episode that terminated below the cap; e25 has none. There is
   effectively no wrong-and-short cell for the cosine's wrong-branch to act on.
2. **Correct episodes got longer with more room** (median 567 -> 667). Extra budget is
   spent, not banked - which is the slack a length reward is supposed to reclaim, and
   e25 reclaims part of it.

## Residual limits

- 7-8% of episodes still terminate at 4096, and e22's 16k probe had 0/20 at cap with
  accuracy 1.00, so the confound is reduced, not eliminated. 4096 is the L4 ceiling
  under accelerate's mixed precision (fp32 logits upcast), so removing the rest needs
  either different hardware or dropping mixed precision and losing comparability with
  every earlier run.
- Single seed. The w-sweep already produced one seed pair whose medians swapped under
  replication, so seeds 43/44 are the minimum before this direction is claimed.
- 100 eval episodes. The paired design buys real power over the old unpaired
  comparison (diff-of-medians CI narrows from [-290, +93] unpaired to [-232, +3]
  paired), but not enough.

## Reproducing the numbers

`/tmp/harvest.py` and `/tmp/paired.py` on the box (copies in the session scratchpad)
read the two `eval_report.json` files; bootstrap RNG is seeded at 0.
