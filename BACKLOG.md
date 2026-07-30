# Backlog

Deferred work. `RUNNING.md` is live state, this is what is not started yet.
Drop an item once it is done or decided against (record the decision in
`DECISIONS.md` if it is worth keeping).

## 1. Liger fused GRPO loss - lift the 4096 completion-token training ceiling

Training completions are capped at 4096 tokens because the policy update
materialises four full `[T, 151936]` bf16 tensors per completion (logits, the
`log_softmax` output, and a gradient for each), about 4.6 GiB at T=4096 on top
of a ~14.4 GiB resident base. 6144 OOMs. The cap is what produces the
truncation confound that corrupted e9-e21 and still leaves 7-8% of e24/e25
episodes cut off mid-reasoning.

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

## 2. Re-evaluate e24/e25 at 16k

Eval is inference-only and has no memory ceiling; e22/e23 already ran at 16384.
`agentic_eval` defaults `max_new_tokens` to the training budget, so the e24/e25
reports inherited a 4096 cap for no reason and 7-8% of their episodes are
truncations counted as wrong. Re-running the existing checkpoints removes that
from the headline without retraining. About 2-2.5h per arm. It will move the
numbers, including the paired median comparison.

## 3. Decide how findings survive outside `pipeline/runs/`

`pipeline/runs/.gitignore` ignores everything, so eval reports, batch summaries
and `e24_e25_4k_pair_findings.md` are not in git. Right now the only copies are
the local disk and the box. Either commit the write-ups (not the checkpoints) or
push them to the thesis wiki as the durable record.

## 4. Regenerate the poly plots

`runs/plots_poly_wsweep/` predates e22-e25 and is stale locally and on the box.
Note the rsync footgun: the box runs its own copy of `eval/plots.py`.

## 5. Fix the four `test_multiturn_eval` failures

Pre-existing since ace8954, `tool_names` kwarg. Not caused by the reward work.

## 6. File the e24/e25 note in the thesis wiki

Findings are in `pipeline/runs/e24_e25_4k_pair_findings.md`, not yet in the
Obsidian vault.
