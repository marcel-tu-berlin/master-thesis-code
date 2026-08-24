# Backlog

Deferred work, and only work that is still open. `RUNNING.md` is live state, this
is what has not started yet.

Delete an item the moment it is done or decided against. This file answers "what
could we pick up next", and a finished item answers nothing while still costing a
read. If the reasoning behind a dropped item is worth keeping, move it to the file
that owns it - `DECISIONS.md` for environment choices, `LAB_NOTES.md` for traps
and standing decisions, `pipeline/runs/*_findings.md` for numbers - and then
delete it here. Git holds what nobody moved.

## 1. Liger fused GRPO loss - lift the 4096 completion-token training ceiling

Training completions are capped at 4096 tokens because the policy update
materialises four full `[T, 151936]` bf16 tensors per completion (logits, the
`log_softmax` output, and a gradient for each), about 4.6 GiB at T=4096 on top
of a ~14.4 GiB resident base. 6144 OOMs. The cap is what produces the
truncation confound that corrupted e9-e21 and still leaves 7-8% of e24/e25
episodes cut off mid-reasoning. That confound is real and independent of the
token-counting bug that separately voided those runs' cosine results - lifting
the cap is still worth doing, it just no longer rescues anything retroactively.

`use_liger_kernel` is a TRL `BaseConfig` field (default `False`). When set,
`GRPOTrainer` builds `LigerFusedLinearGRPOLoss` and routes the loss through
`compute_liger_loss` instead of `_get_per_token_logps_and_entropies`. The fused
path folds the LM head matmul into the loss and chunks over the sequence, so
the full `[T, vocab]` tensor is never allocated. transformers claims ~60% memory
reduction. If it holds here, 16k training becomes reachable on the L4 with no
new hardware and no precision change.

Done 2026-08-24: liger-kernel 0.8.2 installed and in the lock, `training.use_liger_kernel`
passes through `_grpo_config`, and a browsergym `--smoke` ran the fused path end to
end with a live env server (TRL's `compute_liger_loss` applies the multi-turn
`tool_mask`, the vLLM importance-sampling ratio and the ref logps itself, read at
trl 1.6 source). Qwen3 is supported (`apply_liger_kernel_to_qwen3` imports).

Still open:

4. `probe-p2-liger` vs `probe-p2-base` (50 steps, seed 42, running - see
   `RUNNING.md`): is the loss path equivalent (paired per-step reward, grad_norm,
   loss) and what does it save in step time. Read with `python -m probes.p2_compare`.
   Note the liger path logs `clip_ratio` only, not the `clip_ratio/{low,high,region}`
   family - check `eval/plots.py` reads before plotting a liger run.
5. Memcheck at 16k (`max_seq_length` 20480, `max_prompt_length` 4096, `micro_batch_size`
   1). If it fits, the cap question is closed and `micro_batch_size` 2-4 becomes a
   speed knob to probe next.

Dead end, do not retry: dropping mixed precision. `cast_lm_head_to_fp32`
defaults to `False` and we never set it, so there is no fp32 upcast to remove.
Forcing fp32 logits flips `selective_log_softmax` to its chunked-logsumexp
branch at 4 bytes for the logits plus 4 for the gradient - the same 8 bytes per
element. No saving.

## 2. Model scale x quantization sweep - larger Qwen3, 4-bit where needed

Supersedes the audit plan's D1, which was gated on "menu-2 stays flat" and
deleted when B3 refuted that premise. This is the ungated version: parameter
count and quantization are exploration axes in their own right - try the next
larger Qwen3 rung(s), quantized where bf16 does not fit the 24 GB L4.

Registry state today: the `qwen3-4b` slug points at `Qwen/Qwen3-4B-Base`,
which is the wrong lineage counterpart - the pipeline's 1.7B is the
post-trained `Qwen/Qwen3-1.7B` with the tool-calling chat template. A 4B rung
needs a `Qwen/Qwen3-4B` entry; an 8B rung needs a new entry with
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

Steps: fix the registry entries, D1-style base eval of 4B on the e0m splits
(cheap, eval-only, answers "does scale alone move menu-2"), then decide
whether a trained 4B arm is worth the geometry re-run.
