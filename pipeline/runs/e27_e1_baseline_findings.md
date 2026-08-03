# e27: the E1 task-success baseline on BrowserGym/MiniWoB

Task-success-only GRPO (`env_reward` alone, `naive_sum`), 300 steps, Qwen3-1.7B
+ LoRA on an L4. The specialized baseline that E2 (cosine length) and E3
(non-termination penalty) are shaped on top of, and the first run in this
project whose off-target axis is measured rather than inferred.

Run at `runs/e27-browsergym-e1-baseline-qwen3-1_7b/`. Task mix click-menu-2 +
click-dialog-2, selected by four probes
(`browsergym_difficulty_correction.md`). Trained 2026-08-01 17:25 to 2026-08-03
05:05 UTC, then 200 eval episodes across two splits.

## Result

| split | acc | 95% CI | term | non-term | mean tokens | mean steps |
|---|---|---|---|---|---|---|
| held_out | 0.780 | [0.689, 0.850] | 0.900 | 0.100 | 1711 | 1.78 |
| shifted | 0.790 | [0.700, 0.858] | 0.850 | 0.150 | 1230 | 1.94 |

Per family, since each split is an even 50/50 mix and the aggregate hides
which half moves:

| split | family | n | acc | term | gap | median tok | steps | terminated-but-wrong |
|---|---|---|---|---|---|---|---|---|
| held_out | click-menu-2 | 50 | 0.70 | 0.80 | +0.10 | 2160 | 2.54 | 5 |
| held_out | click-dialog-2 | 50 | 0.86 | 1.00 | +0.14 | 313 | 1.02 | 7 |
| shifted | navigate-tree | 50 | 0.62 | 0.74 | +0.12 | 363 | 1.54 | 6 |
| shifted | click-checkboxes-transfer | 50 | 0.96 | 0.96 | 0.00 | 991 | 2.34 | 0 |

Training was stable and never saturated: reward quartiles [0.628, 0.717, 0.632,
0.738], `frac_reward_zero_std` [0.507, 0.453, 0.400, 0.480]. It rises 0.375 ->
about 0.72 inside the first 30 steps and then holds. Completion length is flat
at 1300-1700 across all 300 steps, so E1 does not compress on its own past that
first transient - which is what leaves room for E2 to be measured against it.

## The axis separation the environment was chosen for is real

Success sits strictly below termination on three of the four families (+0.10,
+0.14, +0.12). Ninety of a hundred held-out episodes terminated and 78 were
correct, so twelve terminated with a wrong answer. Task performance and the
off-target axis move independently, which is the property finqa and the
saturated MiniWoB families lacked and the reason a non-termination penalty can
be told apart from a task reward here at all.

## Non-termination is premature stopping, not truncation

Every non-terminating held-out episode stopped on `no_tool_call` - the model
ended its turn without acting. Zero hit the generation cap. Across all 200
episodes there is exactly one `hit_generation_cap`.

This is the first clean read of this axis in the project. The e9-e21 sweep was
uninterpretable precisely because non-termination there was truncation wearing
a behavioral label; here the 4096 budget is large enough that the two are
separated, and the behavior that remains is the one the thesis names.

## click-dialog-2 is a control family, not a treatment family

It terminates in 50 of 50 episodes and solves in 1.02 steps and 313 median
tokens. There is no non-termination for E3 to remove and no length for E2 to
compress - the cosine's within-group reward spread at that length is about
0.01 against env_reward differences of 1.0.

Its failure mode is 7 terminated-but-wrong episodes: pure task error with no
off-target component at all.

So the mix is one treatment family and one control. click-dialog-2 earns its
place as gradient insurance - it is what kept `frac_reward_zero_std` off 1.0
when click-menu-2 sampled near 0.10 in the first steps - but **both efficiency
conditions will act on the click-menu-2 half of the training set only, and E2
and E3 results have to be read against that half.** The same holds on the
shifted split: click-checkboxes-transfer is at 0.96/0.96 with a zero gap and
zero terminated-but-wrong, while navigate-tree carries all 13 of that split's
premature stops.

## Against E0: no measurable task gain, a significant off-distribution loss

`e0-browsergym-base-qwen3-1_7b` ran the identical splits at the identical seeds
and budget with no adapter, so every episode pairs. Paired data calls for a
paired test; comparing the two runs' independent Wilson intervals would discard
most of the power, and on this data it would have called everything null.
McNemar exact, two-sided:

| comparison | E0 -> e27 | lost | gained | p |
|---|---|---|---|---|
| held_out accuracy | 0.750 -> 0.780 | 4 | 7 | 0.549 |
| click-menu-2 | 0.66 -> 0.70 | 0 | 2 | 0.500 |
| click-dialog-2 | 0.84 -> 0.86 | 4 | 5 | 1.000 |
| **shifted accuracy** | **0.870 -> 0.790** | **9** | **1** | **0.021** |
| navigate-tree | 0.74 -> 0.62 | 7 | 1 | 0.070 |
| click-checkboxes-transfer | 1.00 -> 0.96 | 2 | 0 | 0.500 |

**Three hundred steps of task-success-only GRPO bought no measurable accuracy on
the families it trained on, and cost a significant amount on families it did
not.** The loss is carried by navigate-tree (7 of 8 flips) but reaches
significance only when both shifted families are pooled.

The probe-based version of this comparison overstated it. navigate-tree probed
0.85 at n=20 on different seeds; paired against E0 on these seeds the base rate
is 0.74, so the drop is 0.12 rather than the 0.23 the probes implied. This is
the second time a 20-episode browsergym estimate has moved by more than a tenth
under re-measurement.

### Length inflated on the treatment family

On the 33 click-menu-2 seeds **both** models answered correctly, e27's episodes
are longer: median delta +247 tokens, 23 longer against 10 shorter, sign test
p=0.035. click-dialog-2 does not move (+10 tokens, p=0.296), nor does either
shifted family.

So the E1 recipe inflates length on exactly the family that has length to
inflate, which is the family E2 has to compress. That is headroom rather than a
problem: E2 has 247 tokens of training-induced inflation to remove before it
even returns to base.

Two cautions on this number. Marginal per-family token statistics are not usable
here - the correct-episode sets differ between runs (33 against 35 on
click-menu-2), so a marginal mean moves with composition; only the both-correct
paired subset is sound. And the pooled per-split median is meaningless
regardless: each split is an even mix of a roughly 330-token family and a
roughly 2200-token family, so the pooled median lands in the empty space between
the two clusters and moves arbitrarily. Report per family, and report the mean
rather than the median when pooling.

### Why the training reward rose so much more than greedy accuracy

Training reward went 0.375 -> about 0.72 while greedy held-out accuracy moved
0.750 -> 0.780. The two are not measuring the same thing: training reward is
over rollouts sampled at temperature 1.0, greedy eval reads the mode. The
consistent reading is that GRPO sharpened the sampling distribution onto answers
the greedy decode was already finding, which raises sampled reward a great deal
and greedy accuracy barely at all.

Stated as the interpretation it is, not a measurement - confirming it would take
an eval at temperature 1.0, which the protocol does not run.

### What this does and does not change

It does not change the E2 and E3 designs. Both are shaped on top of this exact
recipe with e27 as their lambda=0 control, and both act within prompt-groups, so
neither needs E1 to have improved task success in order to move length or
termination.

It does change how their results are read. The baseline they sit on is a policy
that traded a significant amount of off-distribution accuracy for no measurable
on-distribution gain. Any substitution E2 or E3 shows is on top of a transfer
cost the task reward alone already imposed - and RQ2's question of whether
reducing one inefficiency shifts the agent toward another now has a documented
precedent in the baseline itself.

Whether 300 steps at 5e-6 is the right E1 to build on is a separate design
question. Strengthening it would break the matched budget every arm now shares,
so it is not a change to make quietly mid-campaign.

## Two panel metrics that do not apply here

`unsupported_claim_rate` (0.54 / 0.51) and `mean_verification_depth` (0.76 /
0.89) are computed but carry no information in this domain. They are defined
over a terminal tool called with nothing before it, which needs a multi-tool
environment to mean anything; browsergym exposes `click` and `noop`, where every
call is the same kind of act. Do not report them.

## Reproducing

```
python -m training.train --config configs/e27-browsergym-e1-baseline-qwen3-1_7b.yaml --eval
python -m eval.runner --config configs/e0-browsergym-base-qwen3-1_7b.yaml --base-model
```

Per-episode records with `seed`, `stop_reason` and the ordered tool calls are in
`episodes_held_out.jsonl` and `episodes_shifted.jsonl` under the run directory.
Family per episode is `tasks[seed % 2]`.
