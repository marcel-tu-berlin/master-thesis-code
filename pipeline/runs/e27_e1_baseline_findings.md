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

## What E0 has to settle before any transfer claim

Against the base-model probes the trained policy looks like it gained where it
trained and lost where it did not: click-menu-2 0.60 -> 0.70, click-dialog-2
0.80 -> 0.86, but navigate-tree 0.85 -> 0.62 and click-checkboxes-transfer 1.00
-> 0.96.

**That comparison is not sound and must not be reported as one.** The probes ran
20 episodes per family at `seed_offset` 300000; these splits are 50 per family
at 100000 and 200000. Different seeds, different n, and the correction doc puts
the noise floor on a 20-episode browsergym point estimate at roughly one episode
even before that.

`configs/e0-browsergym-base-qwen3-1_7b.yaml` runs the identical splits at the
identical seeds and budget with no adapter, which makes it paired episode for
episode. If navigate-tree comes back near 0.85 there, task-success-only training
cost 0.23 accuracy on an unseen family, and that is a specialization result the
efficiency conditions then have to be read on top of. If it comes back near
0.62, there is no transfer loss and the probe number was the outlier.

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
