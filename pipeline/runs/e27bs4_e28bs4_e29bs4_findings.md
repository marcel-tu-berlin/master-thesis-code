# e27bs4 / e28bs4 / e29bs4: the browsergym E1/E2/E3 arms

Harvested 2026-08-10. MiniWoB via browsergym (`click-menu-2`, `click-dialog-2`),
Qwen3-1.7B + LoRA, seed 42, 150 GRPO steps at `batch_size: 4`, `n_rollouts: 8`,
`compose_method: naive_sum`, 100 held-out + 100 shifted eval episodes per arm.

| arm | condition | reward config |
|---|---|---|
| e27bs4 | E1 | `env_reward` only (the lambda=0 control) |
| e28bs4 | E2 | `env_reward` + `token_length` (cosine, w 1.0, max_len 4096) |
| e29bs4 | E3 | `env_reward` + `non_termination` (w 1.0, i.e. lambda=1) |

Diffing the frozen `runs/*/config.yaml` files outside the `description` block leaves
exactly the added reward key in each case, so the one-knob rule holds mechanically and
not just by intent. All three arms answered the identical 100 held-out seeds
(42100000-42100099) and the identical 100 shifted seeds (42200000-42200099), so every
comparison below is paired.

## Headline (held_out)

| | e27bs4 (E1) | e28bs4 (E2) | e29bs4 (E3) |
|---|---|---|---|
| accuracy | 0.670 [0.573, 0.754] | 0.650 [0.553, 0.736] | 0.640 [0.542, 0.727] |
| non-termination rate | 0.140 [0.085, 0.221] | 0.220 [0.150, 0.311] | 0.090 [0.048, 0.162] |
| correct-episode mean tokens | 930.8 | 776.3 | 845.4 |
| correct-episode median tokens | 680.0 | 435.0 | 630.0 |
| stop: env_done | 86 | 78 | 91 |
| stop: hit_generation_cap | 5 | 18 | 0 |
| stop: no_tool_call | 9 | 3 | 9 |
| stop: max_turns | 0 | 1 | 0 |

Accuracy is flat: every Wilson interval overlaps every other, and no pairwise McNemar
comes close (E1-E2 p = 0.83, E1-E3 p = 0.65, E2-E3 p = 1.00). Nothing about task
success is claimed from this campaign.

The arms separate on the last four rows, and they move in opposite directions. That
is the result.

## E0: training bought no accuracy on this task

`e0-browsergym-base-qwen3-1_7b` re-ran on 2026-08-10 against the identical seeds
(42100000-42100099 and 42200000-42200099, `--base-model`, no adapter, same 4096-token
whole-trajectory budget), so it pairs question for question with all three arms. The
2026-08-03 e0 run does not - it scored the pre-fix seed bases 100042.. and 200042.. -
and is archived on the box rather than quoted here.

| held_out | e0 | e27bs4 (E1) | e28bs4 (E2) | e29bs4 (E3) |
|---|---|---|---|---|
| accuracy | 0.680 [0.583, 0.763] | 0.670 | 0.650 | 0.640 |
| non-termination rate | 0.150 | 0.140 | 0.220 | 0.090 |
| correct-episode mean tokens | 1035.6 | 930.8 | 776.3 | 845.4 |

Paired McNemar against e0 on held_out: E1 8 vs 7 discordant (p = 1.00), E2 15 vs 12
(p = 0.70), E3 14 vs 10 (p = 0.54). Not one trained arm beats the untrained model, and
E0's point estimate is nominally the highest of the four. On `shifted` the same holds
(E0 0.810 against 0.800 / 0.820 / 0.790, every McNemar p >= 0.62).

So 150 GRPO steps of env-reward training did not make this policy better at MiniWoB.
Everything below is a comparison between behaviours at equal task competence. That is
still the right question for RQ1/RQ2 - the reward conditions are about *how* the agent
spends tokens and turns, not about whether it wins more - but no sentence in the thesis
should imply the E1 recipe taught the model the task.

The discordant counts are large relative to the differences (15 and 12 flips between
E0 and E2 for a net of -3), so per-question outcomes are noisy while the aggregate is
stable. That is the signature of a policy whose competence did not move and whose
tie-breaking did.

## Training did change length, and it changed it most on the hard family

The pooled accuracy above hides two family-level moves in opposite directions.
`held_out` alternates by seed parity (`tasks[seed % len(tasks)]`), 50 episodes each.

| click-menu-2 (n=50) | e0 | e27bs4 | e28bs4 | e29bs4 |
|---|---|---|---|---|
| accuracy | 0.58 | 0.48 | 0.42 | 0.44 |
| correct-episode median tokens | 1963.0 | 1876.0 | 1622.0 | 1463.0 |
| stop: env_done / cap / no_tool | 36 / 3 / 11 | 36 / 5 / 9 | 28 / 18 / 3 | 41 / 0 / 9 |

| click-dialog-2 (n=50) | e0 | e27bs4 | e28bs4 | e29bs4 |
|---|---|---|---|---|
| accuracy | 0.78 | 0.86 | 0.88 | 0.84 |
| correct-episode median tokens | 275.0 | 301.0 | 271.0 | 261.5 |
| stop: env_done | 49 | 50 | 50 | 50 |

(e28bs4's `click-menu-2` column also carries the single `max_turns` episode.)

Two things fall out.

**`click-dialog-2` is fully solved-and-terminating in every trained arm** - 50 of 50
`env_done`, no truncation, no silent stopping, against 49 of 50 for the untrained
model. So the entire non-termination and
truncation story of this campaign is a `click-menu-2` story, and any RQ2 statement
should say so rather than quoting a pooled rate over a set that is half degenerate.

**Training trades the hard family for the easy one.** Accuracy falls 0.58 to 0.42-0.48
on `click-menu-2` and rises 0.78 to 0.84-0.88 on `click-dialog-2`. Pooled, those cancel
to the flat number in the headline. Whether that is the reward or 150 steps of GRPO
sharpening whatever was already easier is not separable here, but it is a real
behavioural change hiding inside a null.

The strongest single length effect in the campaign is E1 against E0 on `click-menu-2`:
on the 24 questions both answered correctly, median difference **-286 tokens**, 23
shorter against 1 longer, sign test p < 0.0001. The env reward alone compresses the
long family hard. E2 and E3 against E0 on the same family are -341 (n=16, p = 0.077)
and -172 (n=17, p = 0.14) - the same direction, smaller n, no added effect that this
seed can resolve.

This **reverses** the corresponding claim in `e27_e1_baseline_findings.md`, which
reported the E1 recipe *inflating* `click-menu-2` length by +247 tokens against E0.
That comparison was between two batch_size 1 runs on the pre-fix seed scheme; under the
bs4 geometry and the current seeds the sign flips. The old number should not be carried
forward.

## E2 compresses correct episodes, and the compression is paired-significant

55 questions both E1 and E2 answered correctly:

| statistic | value | 95% CI (paired bootstrap, 20k) |
|---|---|---|
| difference of medians | -86.0 | [-321.0, +29.0] |
| median of per-question differences | -60.0 | [-160.0, -7.0] |
| mean of per-question differences | -67.9 | [-130.0, -0.6] |
| E2 shorter / longer / tied | 36 / 19 / 0 | sign test p = 0.0300 |

Quartiles over the same 55 questions:

| | Q1 | median | Q3 |
|---|---|---|---|
| e27bs4 | 268.5 | 398.0 | 1379.0 |
| e28bs4 | 243.0 | 312.0 | 1074.0 |
| delta | -25.5 | -86.0 | -305.0 |

Two of the three intervals exclude zero and the sign test clears 0.05, on a
correctness-matched set. The compression again grows with length, which is the shape a
correctness-gated cosine is supposed to produce. The difference-of-medians interval is
wide because the correct-episode distribution is two clusters, not one: `click-menu-2`
correct episodes run about 1900 tokens and `click-dialog-2` about 300, so which family
the bootstrap happens to straddle moves the median a long way. The per-question
differences do not have that problem.

Split by family, neither half carries the result alone at these sample sizes:

| | n both-correct | median diff | CI | shorter/longer | sign test |
|---|---|---|---|---|---|
| click-menu-2 | 16 | -254.0 | [-305.0, +16.0] | 12 / 4 | p = 0.077 |
| click-dialog-2 | 39 | -33.0 | [-73.0, +6.0] | 24 / 15 | p = 0.200 |

Pooling gets to p = 0.030 because both families lean the same way, which is a
legitimate combination of consistent evidence, not a subgroup fished out after the
fact. But it is worth stating plainly that the significance is a pooled effect and the
per-family n is 16 on the family where the tokens actually are.

This is the second arm in a row, on a different environment, where the paired
statistics point the same way as e25bs4 did on `polynomial_equations`. It is still one
seed each.

## E2 pays for it in truncation, not in accuracy

Stop-reason transitions, per question, E1 to E2 on held_out:

| E1 | E2 | n |
|---|---|---|
| env_done | env_done | 73 |
| env_done | hit_generation_cap | 12 |
| no_tool_call | hit_generation_cap | 6 |
| hit_generation_cap | env_done | 5 |
| no_tool_call | no_tool_call | 3 |
| env_done | max_turns | 1 |

E2 rescues all 5 of E1's truncations and creates 18 new ones. Paired McNemar on the
`terminated` flag: 13 questions non-terminating only in E2 against 5 only in E1,
p = 0.096.

The mechanism is in the reward definition, not in a bug. The cosine pays
wrong-and-long more than wrong-and-short by construction (`r_wrong_long = -0.5`,
`r_wrong_short = -1.0`), so on any episode the policy is going to fail, the reward
gradient points at generating more. On `polynomial_equations` that incentive had
nowhere to go: every wrong episode already sat at the cap. Here roughly a third of
episodes are wrong and they started with headroom, so the reward spent it. E2 shortens
the episodes it solves and lengthens the ones it does not.

Read as an efficiency result that is a genuine cost, not a wash: mean tokens over
*all* episodes actually rise (1347.9 to 1452.4) while correct-episode tokens fall. This
is exactly the confound `mean_token_count_correct` exists to separate, and here it
separates in the direction that makes the pooled number misleading in E2's favour if
you read it the other way round.

Note that six of the new truncations came from `no_tool_call` episodes. Those are
episodes where E1's policy gave up silently and E2's kept generating. Whether that is
better behaviour is a judgement, not a measurement; it is not more correct.

## E3 removes truncation, not premature stopping

Stop-reason transitions, E1 to E3 on held_out:

| E1 | E3 | n |
|---|---|---|
| env_done | env_done | 86 |
| no_tool_call | no_tool_call | 9 |
| hit_generation_cap | env_done | 5 |

That is the whole table. Every single episode either kept its stop reason or was one of
the 5 truncations that became `env_done`. Paired McNemar on `terminated`: 5 to 0,
p = 0.0625.

The e29bs4 config states the target explicitly: "the target behaviour is premature
stopping rather than truncation wearing a behavioural label". Premature stopping is
`no_tool_call`, and it did not move at all - 9 before, 9 after, and not merely 9 in
aggregate but the same 9 questions. What the penalty removed is the category the config
argued was *not* what it was aimed at.

That is not a null, it is a specific finding about the training signal. During training
the non-termination indicator is `1[env never reported done]`, which covers truncation
and silent stopping identically. The policy found the cheaper of the two: finishing
before the cap is a length adjustment, whereas emitting a tool call on a question you
cannot parse is a capability change. Under a penalty that cannot tell them apart, the
first is what gets optimised.

Two consequences worth carrying into the writeup:

- The E3 non-termination number should not be quoted as evidence about premature
  stopping without this decomposition beside it.
- A version of E3 that penalises `no_tool_call` specifically, leaving truncation to the
  cap, is the experiment that would actually test the RQ2 claim. That is a reward
  change, not a knob.

## RQ2's substitution prediction did not appear in E3

The prediction to test was the reverse of E2's: if premature stopping falls, does
completion length rise, the agent buying termination with tokens?

56 questions both E1 and E3 answered correctly:

| statistic | value | 95% CI (paired bootstrap, 20k) |
|---|---|---|
| difference of medians | -83.5 | [-143.5, +142.0] |
| median of per-question differences | +7.0 | [-57.0, +27.0] |
| mean of per-question differences | +6.4 | [-82.9, +97.1] |
| E3 shorter / longer / tied | 26 / 30 / 0 | sign test p = 0.6889 |

Flat in both directions. No substitution, and no compression either, which is expected
since E3 carries no length term. E3's arm-level `mean_token_count_correct` is *lower*
than E1's (845.4 vs 930.8) purely because E1's five 4096-token truncations are gone
from the pool.

## The gradient is matched for E3 and not for E2

`frac_reward_zero_std` over the 150 training steps (a prompt-group with zero
within-group reward variance produces zero advantage and trains on nothing):

| | e27bs4 | e28bs4 | e29bs4 |
|---|---|---|---|
| mean `frac_reward_zero_std` | 0.477 | 0.000 | 0.420 |
| live prompt-groups | 52.3% | 100.0% | 58.0% |
| fully-dead steps | 3.3% | 0.0% | 2.0% |
| same, first 30 / last 30 steps | 0.508 / 0.442 | 0.000 / 0.000 | 0.417 / 0.492 |

E2 receives 1.91x the live gradient of its own control, for the same reason as
e25bs4: rollout lengths differ inside a prompt-group even when all eight agree on the
answer, so the cosine gives every group variance (its `frac_reward_zero_std` is exactly
0.0 at every one of the 150 steps). The E1-E2 contrast is therefore "cosine reward plus
roughly twice the live gradient", and the compression result above inherits that
caveat.

E3 is close to matched at 1.11x, because the non-termination indicator is binary and
constant within most groups. The E1-E3 contrast is clean, and the finding it produces -
the penalty targets truncation - is not explainable by a gradient difference.

Other training-side numbers, all three arms over 150 steps:

| | e27bs4 | e28bs4 | e29bs4 |
|---|---|---|---|
| `EnvReward` raw mean | 0.7081 | 0.7106 | 0.7079 |
| `EnvReward` contrib_l1 | 0.7081 | 0.7106 | 0.7079 |
| `CosineLengthReward` contrib_l1 | - | 0.8455 | - |
| `NonTerminationPenalty` contrib_l1 | - | - | 0.1317 |
| `NonTerminationPenalty` raw mean | - | - | -0.1317 |
| mean completion length | 1464.4 | 1459.4 | 1482.6 |
| terminated-only length | 1424.3 | 1417.5 | 1429.0 |
| same, last 30 steps | 1371.0 | 1333.8 | 1377.2 |
| `clipped_ratio` | 0.0208 | 0.0221 | 0.0292 |
| `tools/call_frequency` | 2.150 | 2.135 | 2.158 |
| kl | 0.00224 | 0.00280 | 0.00232 |
| step time | 350.4 s | 351.0 s | 355.7 s |

Train-time task reward is identical to three decimals across the arms, so no arm
learned the task better than another; whatever the rewards did, they did it to
behaviour and not to competence. The cosine's contribution is comparable in magnitude
to the env reward (1.19x) rather than dominating it as it did at w=16 in the poly
campaign (39x). The non-termination penalty is a fifth of the env reward's magnitude:
13.2% of training rollouts were non-terminating, so lambda=1 buys it a small voice by
construction.

Train-time lengths barely differ, including in the last 30 steps (E2 is 2.7% below E1).
The eval-time compression is much larger than the train-time one, which is worth a
second look at some point - eval is greedy and training samples, so they are not the
same distribution, but the gap is bigger than that alone comfortably explains.

## The shifted split shows nothing

| | e27bs4 | e28bs4 | e29bs4 |
|---|---|---|---|
| accuracy | 0.800 [0.711, 0.867] | 0.820 [0.733, 0.883] | 0.790 [0.700, 0.858] |
| non-termination rate | 0.200 | 0.170 | 0.200 |
| correct-episode mean tokens | 478.3 | 510.5 | 472.6 |

Every paired statistic on `shifted` is null, in all three pairings and in both
directions: median of per-question differences +3.5 (E1-E2), +0.0 (E1-E3), sign tests
p = 0.64 and p = 1.00. Stop-reason transitions are near-diagonal for both treatments.

The straightforward reading is that there is nothing to compress. Correct episodes on
`navigate-tree` / `click-checkboxes-transfer` run 470-510 tokens against 780-930 on the
held-out families, and 20% of episodes are `no_tool_call` in every arm - the shifted
families fail by not acting, which no length reward addresses. So E2's compression does
not transfer out of distribution, but this split cannot distinguish "does not transfer"
from "has no room to transfer into".

## Unlike the poly campaign, accuracy here is a real measurement

On the e24bs4/e25bs4 pair, every wrong episode sat at the cap and
`non_termination_rate` equalled `1 - accuracy` exactly, so accuracy measured whether the
model finished inside the budget. That is not the case here. Wrong episodes on held_out
split as:

| | env_done (wrong but finished) | hit_generation_cap | no_tool_call |
|---|---|---|---|
| e27bs4 | 19 | 5 | 9 |
| e28bs4 | 13 | 18 | 3 (+1 max_turns) |
| e29bs4 | 27 | 0 | 9 |

Most failures are episodes that acted, terminated, and got it wrong. Correct episodes
are 100% `env_done` in all three arms. So on browsergym, accuracy and termination are
genuinely separate quantities and the E3 condition has something of its own to measure -
which was the qualification bar the config set for this task mix.

All five of the truncations E3 converted came back *correct*, so that conversion is not
what drives the wrong-and-terminated count from 19 to 27. That rise is 11 questions E1
answered correctly and E3 did not, against 3 the other way, all of them `env_done` in
both arms - correctness churn inside the already-terminating set, and null by McNemar.

## Residual limits

- **One seed per arm.** The poly w-sweep produced a seed pair whose medians swapped
  under replication. Seeds 43/44 are the minimum before E2's compression is claimed as
  a direction rather than an observation.
- **E2 is not gradient-matched** (1.91x live groups). E3 is.
- **Under/overthinking rates in the three reports are not comparable.** Each derives
  its thresholds from its own token distribution. Use `load_reference_thresholds`
  against a fixed reference before quoting them.
- **E0 is paired but also single-seed**, and it is the arm that reframes the campaign:
  no trained arm beats it. Read every result here as a behavioural contrast at equal
  competence, never as an improvement over no training.
- **E3's non-termination result reads differently against E0 than against E1.** Against
  its own control the penalty removed 5 truncations and touched no `no_tool_call`
  episode. Against E0 it removed 6 `no_tool_call` episodes (paired non-termination
  McNemar 6 vs 0, p = 0.031) and 3 of E0's truncations turned *into* `no_tool_call`.
  Which of the two comparisons is the honest one depends on whether E1 or E0 is the
  reference for RQ2; the E1 control is the pre-registered one and is the one the
  headline uses, but the disagreement is seed-level churn of the size the accuracy
  flips already showed, and it should not be resolved by picking the flattering arm.
- **The unsupported-claim and verification-depth panel is weak here.** The rate is
  `len(tool_calls) == 1` among terminated episodes, and browsergym exposes only `click`
  and `noop` with no distinguished terminal call, so it fires on every one-click
  success. Read those two numbers as a step-count proxy, not as verification behaviour.

## Provenance

Wall time 52h39m: e27bs4 train 14h56m + eval 2h30m, e28bs4 14h57m + 2h43m, e29bs4
15h09m + 2h22m. Batch summary `batch_summary_20260810_000818.md`. E0 re-eval 2026-08-10
10:58-12:29 UTC, 1h31m, 200 episodes, no adapter, clean exit with no errors in the log.
Run directories are in this folder minus checkpoints; the paired statistics read
`episodes_held_out.jsonl` and `episodes_shifted.jsonl` from each, bootstrap RNG seeded
at 0.
