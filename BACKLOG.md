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

Not verified yet:

- `liger_kernel` is not installed on the box (`pip install liger-kernel`); TRL
  raises `ImportError` if the flag is set without it.
- Whether the released liger-kernel supports Qwen3.
- Whether `LigerFusedLinearGRPOLoss` composes with `environment_factory` and the
  multi-turn tool masking the agentic path depends on.

Steps:

1. Install liger-kernel on the box.
2. Add `use_liger_kernel` passthrough in `_grpo_config` (`training/grpo_runner.py`).
3. `--smoke` agentic run at the current `max_seq_length: 5120` - does the path
   run at all with a live env server.
4. A/B against e24 at 4096, same seed, to confirm the loss path is numerically
   equivalent and not just cheaper.
5. Memcheck at 16k. If it fits, the cap question is closed.

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
