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

## 2. Decide whether to re-run any poly cosine arm - needs a call

The cosine reward now counts the whole completion. Whether to spend GPU
re-running the e24bs4/e25bs4 pair on the fixed counter is a scope question, not a
correctness one: polynomial_equations is saturated and the study has moved to
browsergym, where e28 is the E2 arm.

- Cheap: re-run nothing, and stop citing the poly null entirely. The thesis then
  says the length reward was tested on browsergym, and nothing about poly.
- Thorough: one re-run of the pair, about 17h per seed, which converts "we never
  tested it there" into an actual result.

Not blocked any more: both configs are in `configs/` as of `9853a94`, recovered
from the frozen copies that were their only record. The thorough option is a
launch away.
