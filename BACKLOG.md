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

## 3. Regenerate the poly plots

`runs/plots_poly_wsweep/` predates e22-e25 and is stale locally and on the box.
Note the rsync footgun: the box runs its own copy of `eval/plots.py`.

## 4. File the e24/e25 note in the thesis wiki

Findings are in `pipeline/runs/e24_e25_4k_pair_findings.md`, not yet in the
Obsidian vault.

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
