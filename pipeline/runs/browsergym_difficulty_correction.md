# BrowserGym difficulty: correction to the feasibility findings

`browsergym_feasibility_findings.md` selected the e27 task mix from a probe whose
numbers do not transfer. This records what replaced them, and a hardware ceiling
that the corrected numbers exposed.

Everything below was measured base-model, greedy, through `eval.runner
--base-model`, which drives the same `BrowserGymEnvAdapter` and prompt that
training uses. Twenty episodes per family, paired seeds (`seed_offset: 300000`)
across all three probes, so any family can be compared against itself between
budgets.

## What went wrong with the original probe

`pipeline/probes/bg_probe.py` built its tool list from `_BROWSERGYM_TOOLS` - all
five browsergym actions - and used its own prompt. The adapter exposes exactly
two, `click` and `noop`, and the domain supplies its own lead-in. Difficulty
measured under a five-tool menu does not survive contact with a two-tool one:
every family the two probes share came in about 0.45 higher under the real
conditions.

| family | original probe | e27probe (real adapter) |
|---|---|---|
| click-option | 0.50 | 1.00 |
| click-checkboxes | 0.35 | 0.95 |
| click-widget | 0.60 | 0.90 |
| navigate-tree | 0.70 | 0.80 |

e27 was launched on the left column and killed at step 21 of 300 with
`frac_reward_zero_std` 0.70 - 14 of 20 prompt groups had all 8 rollouts score
identically - and a mean reward of 0.931. No gradient, and worse, success and
termination had collapsed onto one axis, which is the condition the task set was
chosen to avoid.

The lesson generalises past this bug: a difficulty number is only valid under
the tool surface, prompt, and token budget it was measured with. Two of those
three have now each invalidated a measurement in this project.

## The prompt hint was not the cause

The domain's `_LEAD_IN` ended "Most tasks need a selection followed by clicking
Submit", which is the solution to both training families stated outright. e27probe2
removed that sentence and changed nothing on paired seeds: click-option 1.00 ->
1.00, click-checkboxes 0.95 -> 1.00. The two-tool surface accounts for the
inflation on its own. The hint-free lead-in stays - handing over the plan is not
a fair measurement - but it is not a fix.

## Half the difficulty ranking was truncation

At a 2048-token budget the hard end of the ranking was mostly episodes cut off
mid-generation, not episodes that failed. e27probe3 re-ran the four survivors at
4096 on the same seeds.

| family | acc 2048 -> 4096 | term 2048 -> 4096 | gap | cap-hits |
|---|---|---|---|---|
| navigate-tree | 0.80 -> 0.85 | 0.95 -> 0.85 | 0.15 -> 0.00 | 0 -> 0 |
| click-checkboxes-soft | 0.75 -> 0.85 | 0.80 -> 0.90 | 0.05 -> 0.05 | 3 -> 2 |
| click-checkboxes-large | 0.45 -> 0.60 | 0.45 -> 0.75 | 0.00 -> 0.15 | 8 -> 1 |
| click-tab-2 | 0.10 -> 0.05 | 0.35 -> 0.35 | 0.25 -> 0.30 | 11 -> 2 |

click-checkboxes-large is the only family that clears both selection bars, and
it clears them only after the budget stops lying: five of the six seeds that
flipped to correct had stopped on `hit_generation_cap` at 2048. The mechanism is
attributed per seed, not inferred from the aggregate.

click-tab-2 is the counter-example that keeps this from being "a bigger budget
lifts everything". Its cap-hits fell 11 -> 2 and its accuracy did not move,
because the freed budget went into looping instead: 5 of 20 episodes now run to
`max_turns` and 6 stop without calling a tool. It stays disqualified, at 0.05
success and a 5327-token median.

navigate-tree moves the other way. Its 0.15 gap at 2048 was itself truncation;
at 4096 it reads 0.85 success against 0.85 termination, with all 17 terminated
episodes correct. Fully collapsed axes.

### How much of this is noise

The controls moved too, which needs accounting for. Comparing trajectories per
seed between the two budgets:

| family | mean steps | identical trajectories | net accuracy change |
|---|---|---|---|
| navigate-tree | 1.6 | 16/20 | +1 episode |
| click-checkboxes-soft | 3.6 | 9/20 | +2 episodes |
| click-checkboxes-large | 5.9 | 3/20 | +3 episodes |

Greedy decoding on a fixed seed should reproduce exactly, so the non-identical
trajectories are environment nondeterminism, and it compounds with episode
length - once one turn diverges, every later turn does. That sets a noise floor
of roughly one episode (0.05) on short families and more on long ones. It is a
standing limit on every 20-episode browsergym point estimate here, and it argues
for more episodes per family, not more families, in any future probe.

click-checkboxes-large's termination moved 6 episodes with the mechanism visible
in the stop reasons, which is well clear of that floor. Its accuracy moved 3.

## Why the winning family cannot be trained on this box

TRL applies `max_completion_length` to the entire multi-turn trajectory rather
than per turn. `grpo_trainer.py:1605` drops tool results once `len(pct) -
len(prompt_ids[i])` passes it, and the loss at `grpo_trainer.py:2628` divides by
it, so the logits tensor is sized to it and memory scales with it directly. With
the fp32 logits upcast at roughly 10 bytes per token per vocabulary entry and a
151,936-entry vocabulary, 4096 completion tokens costs about 6.2 GB and 8192
about 12.5 GB, against the roughly 12 GB left on a 24 GB L4 after the policy and
the vLLM engine.

Measured against that ceiling, using assistant tokens alone as a lower bound on
trajectory length (page observations count against the same budget on top):

| family | median assistant tokens | episodes over 4096 |
|---|---|---|
| click-checkboxes-large | 6066 | 16/20 |
| click-checkboxes-soft | 3390 | 9/20 |
| navigate-tree | 331 | 3/20 |

So click-checkboxes-large would train truncated in 16 of 20 episodes. That is
the e9-e21 artifact again, moved from the eval side to the training side, and it
would be harder to detect there.

Across every family probed so far, difficulty and trajectory length are
entangled: the trainable ones are saturated, and the interesting one is too
long. Qwen3's thinking blocks are the proximate cause - about 1030 tokens per
turn on click-checkboxes-large, with every prior turn's block retained in the
completion. Turning thinking off is not the cheap way out: the model then emits
around 24 tokens and passes the visible label instead of the element id.

## Greedy difficulty is not sampled difficulty

The mix chosen above was launched and the discrepancy showed up in the first six
training steps. click-menu-2, probed at 0.60, ran near 0.10 per rollout;
click-dialog-2, probed at 0.80, ran near 0.875. Tool failure frequency was 0 and
completion lengths matched the probe, so neither a broken tool nor truncation
explains it.

The probes ran greedy (`eval.temperature: 0.0`, `do_sample: false`). Training
samples at temperature 1.0 (`grpo_runner.py:93`, unset in the config so it takes
the default). GRPO's within-group variance - the thing the 0.40-0.80 band exists
to protect - depends on the sampled success rate, not the greedy one, and for a
2.6-step task the two differ by about 6x because each turn resamples an element
id and the errors compound.

So a selection probe has to run `do_sample: true` at the training temperature.
Greedy accuracy answers "can the model do this task at all", which is the wrong
question for choosing a GRPO training set; the right one is "how often do eight
sampled rollouts disagree".

This is the third instance of one error in this environment: difficulty measured
under conditions the training run does not use. The tool surface was first (five
tools versus two), the token budget second (2048 per turn versus 2048 for a
whole trajectory), and the decoding mode third. Each was invisible until
something downstream behaved impossibly.

It did not sink the run - the policy learned anyway, reward climbing 0.375 to
0.734 over the first 30 steps - but that is luck rather than method. Had
click-dialog-2 not been in the mix, every group would have been all-zero and the
run would have had no gradient at all, which is exactly how finqa failed.

## Train/eval budget mismatch

Independently of which family wins, `max_seq - max_prompt_length` currently
feeds two different quantities: TRL's whole-trajectory `max_completion_length`
in training (`grpo_runner.py:78`), and `max_new_tokens` per turn in eval
(`agentic_eval.py:36`, where generate is called once per turn with no total
cap). On a single-turn domain these coincide, which is why reasoning_gym never
exposed it. On an n-turn domain they differ by a factor of n: e27's original
geometry gave training 2048 tokens for an entire 8-turn episode while eval
allowed 2048 on every turn. `eval.max_new_tokens` already overrides the eval
side, so the two can be set apart without a code change.

## The competence boundary is about three decisions

e27probe4 probed seven click-only families chosen to be decided in one or two
clicks. Five had finished when this was written:

| family | acc | term | gap | median assistant tokens | steps |
|---|---|---|---|---|---|
| click-dialog-2 | 0.80 | 1.00 | 0.20 | 385 | 1.0 |
| click-menu-2 | 0.60 | 0.70 | 0.10 | 2569 | 2.6 |
| click-tab-2-hard | 0.10 | 0.20 | 0.10 | 6550 | 5.7 |
| click-collapsible-2 | 0.05 | 0.30 | 0.25 | 5638 | 4.8 |
| click-pie | 0.00 | 0.00 | 0.00 | 8929 | 6.3 |

The split is sharp and it falls on episode length, not on task family. Every
family averaging more than about four decisions collapses, and always the same
way: 7, 10 and 13 of 20 episodes loop until `max_turns`. Every family under
three decisions is solvable. Nothing sits in between.

That explains why difficulty and trajectory length looked entangled in every
earlier probe: they are the same variable. Qwen3-1.7B does not fail MiniWoB
families because their pages are harder to read, it fails them once the episode
needs more steps than it can keep coherent, and a failing episode then spends
the entire remaining budget looping. The apparent difficulty of
click-checkboxes-large, click-tab-2, click-pie and click-collapsible-2 is one
phenomenon with one cause.

The practical consequence is that the 0.40-0.80 band and the token budget are
not two independent bars for this model. Anything hard enough to be interesting
by looping its way there is untrainable by construction, so the only usable
families are the ones that are hard within two or three decisions.

## Where this leaves the environment choice

Three ways forward, in the order they should be tried:

1. **Find a short-hard family.** e27probe4 probes seven click-only families
   decided in one or two clicks, where difficulty comes from reading the page
   rather than acting many times: click-pie, click-menu-2, click-collapsible-2,
   click-dialog-2, click-tab-2-hard, grid-coordinate, click-checkboxes-transfer.
   A third selection bar applies - median assistant tokens low enough that the
   trajectory and its observations fit 4096. Costs about two hours and needs no
   new code.
2. **Lift the completion ceiling.** A chunked or fused GRPO loss (Liger, already
   in BACKLOG) removes the fp32 logits materialisation and would make
   click-checkboxes-large trainable at its natural length. This is the only
   option that rescues the family already known to clear both bars.
3. **Shorten the trajectories.** Stripping prior turns' thinking blocks from the
   running completion would cut length by roughly the number of turns. It is a
   change to TRL's multi-turn loop and would alter what the policy conditions
   on, so it needs its own justification rather than being slipped in as an
   optimisation.

Families excluded throughout, on the rule established for click-shape and
click-color: count-shape, count-sides, identify-shape and click-shades turn on
attributes an accessibility tree does not carry, so a text-only observation
cannot see the target at all.

## Reproducing

```
python -m eval.runner --config configs/e27probe-browsergym-difficulty-qwen3-1_7b.yaml --base-model
python -m eval.runner --config configs/e27probe2-browsergym-nohint-qwen3-1_7b.yaml --base-model
python -m eval.runner --config configs/e27probe3-browsergym-4k-qwen3-1_7b.yaml --base-model
python -m eval.runner --config configs/e27probe4-browsergym-short-qwen3-1_7b.yaml --base-model
```

Per-episode records, including `stop_reason` and the ordered tool calls each
claim above rests on, are in `episodes_<split>.jsonl` under each run directory.
