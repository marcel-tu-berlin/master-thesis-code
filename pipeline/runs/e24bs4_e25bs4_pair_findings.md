# e24bs4 / e25bs4: the cosine pair re-run at batch_size 4

Harvested 2026-08-07. `polynomial_equations`, Qwen3-1.7B + LoRA, seed 42, 150 GRPO
steps at `batch_size: 4`, `n_rollouts: 8`, 100 held-out eval episodes. The frozen
configs differ in exactly one key, `rewards.token_length` (off in e24bs4;
`enabled: true, weight: 16.0, max_len: 4096` in e25bs4) - verified by diffing the two
`config.yaml` files. Both evals ran `seed_base=100042, max_new_tokens=4096` and both
arms answered the same 100 questions, so every comparison below is paired.

This re-runs the 2026-07-30 e24/e25 pair after the batch-size confound was found: the
originals trained at an implicit `batch_size: 1`, one prompt-group per optimizer step.

## Headline

| | e24bs4 (w0) | e25bs4 (cosine w16) |
|---|---|---|
| accuracy | 0.860 [0.779, 0.915] | 0.910 [0.838, 0.952] |
| episodes at the 4096 cap | 14 | 9 |
| wrong episodes below the cap | 0 | 0 |
| correct-episode mean tokens | 720.0 | 783.7 |
| correct-episode median tokens | 459.5 | 427.0 |
| all-episode mean tokens | 1192.7 | 1081.8 |

The cosine arm is more accurate and shorter at the same time. Read the two token
columns with care: the correct-episode **mean** rises (720.0 -> 783.7) while the paired
median falls, because e25bs4 solves 5 questions e24bs4 did not and those 5 were the
long ones. That is a composition change, not a regression, and it is exactly why the
paired statistics below are the valid contrast.

## Paired statistics (86 questions both arms answered correctly)

| statistic | value | 95% CI (paired bootstrap, 20k) |
|---|---|---|
| difference of medians | -65.0 | [-118.5, -3.5] |
| median of per-question differences | -47.0 | [-81.0, -1.0] |
| mean of per-question differences | -51.1 | [-133.4, +32.1] |
| e25bs4 shorter / longer / tied | 53 / 30 / 3 | sign test p = 0.0152 |

Both median-based intervals exclude zero and the sign test clears 0.05. That is new:
in the 300-step bs=1 pair the same two intervals were [-232, +3] and [-61, 0], and the
sign test came in at p = 0.066. The mean-based interval still crosses zero, as before -
the per-question differences have a heavy tail, so the mean is the wrong summary here.

Quartiles over the same 86 questions:

| | Q1 | median | Q3 |
|---|---|---|---|
| e24bs4 | 339.8 | 459.5 | 825.2 |
| e25bs4 | 289.5 | 394.5 | 696.0 |
| delta | -50.3 | -65.0 | -129.2 |

The compression grows with length - a correctness-gated cosine acting hardest on the
long tail, which is the intended shape.

## Accuracy did not pay for it

McNemar on the discordant pairs: 5 questions e24bs4 got wrong and e25bs4 got right, 0
the other way, exact two-sided p = 0.0625. e25bs4's correct set is a strict superset of
e24bs4's on this seed. No sign of the Wu/Yeo overshoot at this weight and budget.

Mechanically the accuracy gain **is** the compression: all 5 flips were episodes that
previously ran into the 4096 cap and now finish inside it. See the next section.

## Accuracy and termination are still the same measurement

Both arms have zero wrong-and-terminated episodes: 86/86 and 91/91 of the correct
episodes stopped with `env_done`, and every wrong episode stopped with
`hit_generation_cap` at exactly 4096 tokens. `non_termination_rate` equals `1 - accuracy`
to the digit in both reports.

So "accuracy" on this task at this budget measures whether the model finished inside
the token budget, not whether it can solve the problem. The e22 16k probe already showed
these episodes terminate correctly given room (16/16, poly 0.80 -> 1.00). Any accuracy
number quoted from this pair carries that caveat, and a length reward improving accuracy
by shortening completions is a coherent mechanism rather than a contradiction - but it is
not evidence about reasoning quality.

## The arms are not gradient-matched

The training logs (150 steps each) show a difference the reward config implies but the
design did not intend:

| | e24bs4 | e25bs4 |
|---|---|---|
| live prompt-groups (non-zero within-group reward variance) | 40.7% | 99.5% |
| fully-dead steps (all 4 groups zero-variance) | 14.7% | 0.0% |
| `EnvReward` raw mean | 0.9442 | 0.9476 |
| `EnvReward` contrib_l1 | 0.2934 | 0.3084 |
| `CosineLengthReward` contrib_l1 | - | 12.0261 |
| train-time mean completion length | 1331.8 | 1220.5 |
| train-time terminated-only length | 1174.3 | 1084.1 |
| same, last 30 steps | 1199.2 | 1026.4 |
| `clipped_ratio` | 0.0585 | 0.0487 |

The task is near-saturated (`EnvReward` raw mean 0.94), so most prompt-groups are
all-correct, the env component z-scores to zero under `advantage_weighted`, and the
baseline arm trains on 40.7% of its groups. Adding the cosine gives every group
within-group variance - rollout lengths differ even when all 8 are correct - so the
treatment arm trains on 99.5%. The treatment arm receives about 2.4x the live gradient
of its own control, and cosine outweighs env in the advantage by 39x.

This is inherent to adding any component with within-group variance, not a bug, but it
means the contrast is "cosine reward plus more gradient" rather than "cosine reward
alone". A gradient-matched control would need a length-independent component with
comparable variance, which does not obviously exist.

The train-time length gap also widens over training (-8.4% over all 150 steps, -14.6%
over the last 30), so the effect is still growing when the budget runs out.

## What batch_size 4 changed (e24 -> e24bs4, same arm)

Both are the env-only control. The bs4 run trades optimizer steps for prompts: 300
steps x 1 group = 300 prompt-groups before, 150 x 4 = 600 now.

| | e24 (bs 1, 300 steps) | e24bs4 (bs 4, 150 steps) |
|---|---|---|
| accuracy | 0.910 | 0.860 |
| correct-episode median tokens | 667.0 | 459.5 |
| episodes at cap | 8 | 14 |
| fully-dead steps | 57.0% | 14.7% |
| live prompt-groups over the run | 129 | 244 |

Dead steps fall from 57.0% to 14.7% as intended, and live groups nearly double. Accuracy
is lower, which at 100 episodes and one seed is inside the noise the intervals allow
([0.838, 0.966] vs [0.779, 0.915] overlap heavily) and is not attributable without more
seeds.

## Residual limits

- **Single seed.** The bs=1 w-sweep produced a seed pair whose medians swapped under
  replication, so seeds 43/44 remain the minimum before this direction is claimed. The
  paired intervals here exclude zero on one seed; that is a stronger starting point than
  any earlier arm, not a result.
- **The cap still binds.** 9-14% of episodes terminate at 4096. 4096 is the L4 ceiling
  under accelerate's mixed precision, so removing the rest needs different hardware or
  dropping mixed precision and losing comparability with every earlier run.
- **Under/overthinking rates are not comparable across arms.** Each report derives its
  thresholds from its own token distribution, so e25bs4's higher overthinking rate
  (0.176 vs 0.128) is measured against a different ruler.
- **The gradient asymmetry above** is not controlled for.

## Provenance

Wall time 21h42m total: e24bs4 train 9h57m + eval 1h23m, e25bs4 train 9h05m + eval
1h15m. Batch summary `batch_summary_20260807_093614.md`. Both run directories are in
this folder minus checkpoints; the paired statistics read `episodes_agentic.jsonl` from
each, bootstrap RNG seeded at 0.
