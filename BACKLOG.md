# Backlog

Deferred work, and only work that is still open. `RUNNING.md` is live state, this
is what has not started yet.

Delete an item the moment it is done or decided against. This file answers "what
could we pick up next", and a finished item answers nothing while still costing a
read. If the reasoning behind a dropped item is worth keeping, move it to the file
that owns it - `docs/decisions/` for standing decisions (environment choices
included), `LAB_NOTES.md` for traps, `pipeline/runs/*_findings.md` for numbers -
and then delete it here. Git holds what nobody moved.

## 1. Model scale x quantization sweep - larger Qwen3, 4-bit where needed

Supersedes the audit plan's D1, which was gated on "menu-2 stays flat" and
deleted when B3 refuted that premise. This is the ungated version: parameter
count and quantization are exploration axes in their own right - try the next
larger Qwen3 rung(s), quantized where bf16 does not fit the 24 GB L4.

Registry state today: the `qwen3-4b` slug points at the correct post-trained
`Qwen/Qwen3-4B` lineage counterpart. Its old eval-only candidate config is
archived; recreate it from the accepted template when the probe is scheduled.
An 8B rung needs a new entry with
`load_in_4bit: true` (bf16 8B is ~16 GB of weights before the vLLM colocate
copy and the logits path). Eval-only probes (HF greedy generate, no vLLM) fit
4B in bf16 easily; 4B training likely needs 4-bit or tightened budgets.

Method constraints, so the sweep produces comparable numbers:

- The difficulty band is model-relative. menu-2 at ~0.59 sampled for 1.7B may
  saturate for 4B, and families dead-by-looping for 1.7B (checkboxes-large,
  tab-2) may become trainable - the C4 elimination does not transfer, so the
  candidate pool reopens per model. Re-run the band check sampled at the
  training temperature before training any new scale.
- Quantization is a treatment, not free memory. Hold it fixed within any
  compared pair; a 4-bit arm against a bf16 arm confounds scale with precision.
- Each scale gets its own e0 base eval on identical splits before its E1.
- QLoRA + vLLM colocate has an open correctness question: if vLLM serves a
  bf16 copy while the trainer computes on nf4 weights, sampler and trainer
  diverge systematically per token - the same length-correlated drift the TIS
  audit just closed. Before training quantized, check what precision the vLLM
  copy runs at and confirm ISR stays ~1.0 under `token_truncate`.

Steps: create a final-recipe D1-style base eval of 4B on the e0m splits (cheap,
eval-only, answers "does scale alone move menu-2"), then decide whether a
trained 4B arm is worth the geometry re-run.

## 2. Criterion 3 - confirm the training family before the campaign

Gates the lambda campaign. The two environment bars recorded so far
(`probes/README.md`, `docs/decisions/0002-environment-selection.md`) are
pre-training: base accuracy in the
40-80% band and a success/termination gap. Both are measured on the base model.
Criterion 3 (thesis chapter 6, Environment Selection) is post-training and only
an E1 run can answer it: the cost a shaped reward targets must still be there
after task-success-only training. A penalty on a behavior the E1 policy no
longer shows is a silent no-op, and its sweep can only read as "cost without
effect". A success ceiling hides the other half of the verdict, because a
success drop under shaping cannot show against 1.0.

The check, on the final recipe (kl_beta 0.0, `compose_method: naive_sum`,
`scale_rewards: none`, current defaults), on the candidate family:

1. Train E1 (env_reward only), one seed. This is the stage 1 E1 cell of the
   campaign, not an extra run.
2. Evaluate on the held-out split (unseen seeds, same family) and the shifted
   split.
3. Require on held-out: `non_termination_rate` clearly above zero (E3's
   target), completion length with room to shrink (E2's target), and success
   clearly below 1.0. No numeric thresholds are set yet - decide them from
   the pilot numbers and record them here.
4. Fail any bar: pick a harder family or a mix, repeat.

Do not reuse the seed-42 campaign numbers (e0m, e30-e36): old recipe, group
scaling, old E3 semantics. Reference only, for orientation: the 300-step E1 on
click-menu-2 sat at 0.75-0.78 held-out success and inflated length (+247 median
tokens on both-correct seeds), so menu-2 had E2 headroom under the old recipe;
click-dialog-2 polished toward 0.97, which fails the ceiling bar as a training
family; E1 at a working lr compresses length on its own (LAB_NOTES, 4b rung),
so E2 is always read against the E1 arm, never against the base model.
Nothing is known about the E1 non-termination rate under the new recipe.

Campaign plan after the pilot (cells, lambda grid, placebo arms, staged seeds)
lives in the vault BACKLOG.md, Experiments section.
