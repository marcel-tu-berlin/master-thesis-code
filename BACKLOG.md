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

## 2. Re-evaluate e24/e25 at 16k - DROPPED, the comparison it would sharpen is void

**Do not do this.** The cosine reward in e25 was reading 8% of the completion
length (`runs/cosine_token_count_bug_findings.md`), so e25 is not a cosine arm -
it is a second env-only run that differs from e24 by noise. Re-evaluating both
at 16k would sharpen a comparison between two copies of the same condition.

The truncation point it rested on is still true and still worth having: eval is
inference-only with no memory ceiling, `agentic_eval` inherited the 4096 training
cap for no reason, and 7-8% of those episodes are truncations counted as wrong.
That argument now belongs to whichever cosine arm is actually run, not to these
checkpoints.

## 3. Regenerate the poly plots - DROPPED with item 2

`runs/plots_poly_wsweep/` would plot the voided w-sweep. Regenerate only if the
campaign is re-run on the fixed counter. (If it ever is: the box runs its own
copy of `eval/plots.py`, so rsync the module you execute there.)

## 4. File the e24/e25 note in the thesis wiki - DROPPED, would file a void result

`pipeline/runs/e24_e25_4k_pair_findings.md` reports a directional compression
that the reward could not have produced. Filing it into the vault is worse than
leaving the vault incomplete. If the poly campaign is re-run, file that instead;
otherwise the entry to make is the bug write-up, not the result.

## 4b. Decide whether to re-run any poly cosine arm - OPEN, needs a call

The cosine reward now works. Whether to spend GPU re-running e17 or the e24/e25
pair on the fixed counter is a scope question, not a correctness one: the poly
environment is saturated and the study has moved to browsergym, where e28 is the
E2 arm. The cheap option is to re-run nothing and simply stop citing the poly
null. The thorough option is one re-run of the e24/e25 pair, which would convert
"we never tested it there" into an actual result at about 17h per seed.

## 5. Retarget the repl env at a reasoning_gym task source

Fallback if no off-the-shelf OpenEnv environment clears both bars the e26 run
established (base accuracy 40-80%, and at least two tools). Recorded because it
is cheap and reuses everything, not because it is the current plan.

`domains/repl/tasks.py` mints its own tasks, so repl's difficulty is ours to set.
The current family (sum / maximum / minimum of a list) is deliberately trivial;
its own docstring names the upgrade: "Upgrade target: a richer task source (e.g.
reasoning_gym)." Pointing `make_task` at a reasoning_gym family gives a multi-turn
env at a difficulty the poly campaign already showed how to dial.

What it buys for RQ2: repl exposes one tool (`execute`), but the panel does not
collapse the way reasoning_gym's does, because the model prints `FINAL(answer)`
from inside an execute call. Verification depth becomes the number of execute
calls before the final one, and an episode that prints `FINAL` on its first
execute is a genuine unsupported claim. That is two real off-target axes from one
tool.

What it does not buy: no action-instability axis, and a single-tool panel is
thinner than a 3-4 tool env's. Prefer an off-the-shelf env that clears both bars.

Steps:

1. Swap `make_task` to draw from a reasoning_gym family, keeping it a pure
   function of the seed (training and eval both call `reset(seed=N)`).
2. Base-model difficulty probe to land inside the 40-80% band.
3. Confirm the server-side exact-match rubric still arms correctly against the
   reasoning_gym answer format.
