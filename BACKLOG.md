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

## 2. Expand the campaign after the first read-table-2 result

First execute and review the final from-base E0-E2 campaign in
`docs/plans/e0-e2-campaign.md`: base evaluation, task-only training, and task
success plus relative length cost at weight 0.1. Preparation is complete; runs
have not started. Competent-policy continuation remains a separate future
option in LAB_NOTES.md, not an active arm.

After E0-E2 are complete and reviewed, declare the non-termination/E3 study.
Also review the relevant cost-assignment placebo, optional linear comparison,
at least three training seeds per cell, and additional families or environments.
Complete paired analysis, training-cost accounting
and thesis figures with claims bounded to the tested tasks and indicators.
Freeze the expansion protocol and reserve fresh confirmation data before using
new weights or making claims selected from the first contrast's test results.
