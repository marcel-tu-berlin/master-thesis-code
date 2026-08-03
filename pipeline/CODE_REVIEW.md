# Pipeline code review, 2026-08-03

Max-effort multi-agent review of `pipeline/` (training, eval, domains, configs,
tests) plus the uncommitted working-tree changes. 6 finder agents across
independent angles, 63 verifier agents, 76 candidates raised, 8 refuted, 68
verified and consolidated into 15 findings.

This document is the input to a fix plan. No fixes have been applied.

**Provenance note.** Every claim below carries a CONFIRMED verdict from an
independent verifier that read the code. Numeric claims attributed to
measurement (median answer calls per rollout, the 4-of-100 e27 episodes, the
identical over/underthinking rates under halved token counts) come from those
agents' own runs against the artifacts on disk. They have not been
independently re-derived here. Everything structural (what the code does, which
branch runs when) is directly checkable in the cited lines.

## Verdict

The pipeline is not fine. The dominant failure mode is the same one that
voided e9-e25: a path computes a plausible number from the wrong quantity and
nothing raises.

Three findings change what the numbers already on disk mean. Four more will
change what the next run means. The rest are robustness and cleanup.

The single most important consequence: **the browsergym eval loop measures the
policy under a different token budget and a different context than training
produced.** e27 is the E1 baseline that e28/e29 are paired against and that the
whole RQ1 claim rests on.

---

## Tier 1: affects numbers already on disk

### 1.1 reasoning_gym `answer` has no done-guard; training scores the last
answer, eval scores the first

`pipeline/domains/reasoning_gym/adapter.py:66-80`

Every other adapter guards re-entry after termination (`textarena:69`,
`finqa:161`, `repl:78`, `browsergym:94`). `ReasoningGymEnvAdapter.answer` does
not: it calls `self._client.step(...)` unconditionally and overwrites
`self.reward` on line 78.

Compounding it, `grpo_runner.py:125-127` sets `max_tool_calling_iterations`
only when `env_config.max_turns > 0`, and no reasoning_gym config sets
`max_turns` (checked across e5, e6, e9-e25 and the probes). TRL's tool loop
therefore runs at its `sys.maxsize` default, which is exactly the failure
CLAUDE.md warns about.

A rollout that answers correctly, reads the tool result, then answers again
and gets it wrong ends with `env.reward = 0.0`. `EnvReward` and
`CosineLengthReward`'s correctness gate (`cosine_length.py:75-78`) train it as
wrong; `agentic_eval._parse_answer` takes the first call and scores it correct.
Measured median: 2.25 answer calls per rollout across e24/e25.

The unbounded loop is also a mechanism for rollouts running to the completion
cap, which is the phenomenon the entire e22/e23 cap-probe line was built to
explain. Worth revisiting that conclusion.

### 1.2 Eval applies `max_new_tokens` per turn, training applies it per
trajectory

`pipeline/eval/agentic_eval.py:397` (also `configs/e26-...:52`)

e27/e28/e29 set `max_new_tokens: 4096` and `max_turns: 8`. Training caps the
whole trajectory at 4096 via TRL's `max_completion_length`. `_run_multiturn_episodes`
calls `model.generate(max_new_tokens=4096)` once per turn with no running total,
so an eval episode can generate up to 32768 tokens.

4 of e27's 100 held-out episodes score correct only because they received about
1.8x the trained budget. `mean_token_count` is summed over a budget regime the
policy never trained under, and `hit_generation_cap` can never fire for the
trajectory-level cap that actually bound training.

### 1.3 Multi-turn eval discards the model's own reasoning between turns

`pipeline/eval/agentic_eval.py:203` (test at `tests/test_multiturn_eval.py:100`
encodes the current behaviour)

`gen_turn` decodes the completion, extracts `(name, args, n_tokens)`, throws
the text away, and the loop appends an assistant message with
`{"content": ""}` plus a tool_calls stub.

Training keeps the full generated text in the running completion.
`browsergym_difficulty_correction.md` says so explicitly ("about 1030 tokens
per turn on click-checkboxes-large, with every prior turn's block retained in
the completion"), and that retention is why the 4096 ceiling binds at all.

At eval the model enters turn 2 having lost its own turn-1 think block, which
for Qwen3 is roughly 300 to 1000 tokens of the reasoning that selected the
element id. Every multi-turn eval number measures a policy with amnesia between
turns. The divergence is invisible because both paths report identical per-turn
`n_tokens`.

### Scope note for 1.2 and 1.3

Both defects apply uniformly to every arm evaluated by this loop. A paired
e27-vs-e28 comparison where both arms ran the same buggy eval is not
automatically invalid, because the bias is shared. What is invalid is any
absolute claim, and any comparison against a number produced by a different
eval version. **Fixing the eval loop obliges a re-eval of e27**, or the pairing
breaks.

---

## Tier 2: will corrupt the next run

### 2.1 The uncommitted untagged-JSON parser relabels truncated completions as
clean terminations

`pipeline/eval/agentic_eval.py:55-64`

A Qwen3 rollout that fills the budget and is cut after the closing brace of its
answer JSON but before `</tool_call>`: `_TOOL_CALL_RE` requires the closing tag
so it finds nothing, `tagged` stays False, and the unanchored `raw_decode` scan
then finds the complete object and yields it. `_parse_answer` returns an
answer, so `_run_episodes` sets `terminated=True`, `stop_reason="env_done"`
(lines 154-155) for an episode that hit the cap. Before this change the same
episode was `terminated=False`, `stop_reason="hit_generation_cap"`.

Same for any JSON object with a `"name"` key inside a `<think>` block or a
markdown fence when the real call never landed.

Net: non-termination rate drops, accuracy rises, the `hit_generation_cap`
bucket empties. That is the one distinction CLAUDE.md and `metrics.py:31-33`
say must never be collapsed, and it is what made e9-e21 uninterpretable.
Re-running eval on any existing Qwen checkpoint now produces different,
better-looking numbers with no marker that the parser changed.
`tests/test_agentic_episode_eval.py:84` only covers the case where a tagged
call is present, so this path is untested.

This is uncommitted, so nothing on disk is affected yet. Fix before the next
eval.

### 2.2 The cosine reward measures length with two different rulers on the two
sides of its own correctness gate

`pipeline/training/rewards/cosine_length.py:87,103`

`_n_tokens` uses `len(completion_ids_i)` (raw ids, including `<think>`,
`<tool_call>` and `<|im_end|>` framing) when `_is_multiturn` is False, and
`model_token_count` otherwise. `model_token_count` re-encodes only `content` +
`reasoning_content` + `json.dumps(arguments)` and drops all framing
(`utils.py:53-66`).

`_is_multiturn` is True exactly when TRL interleaved a `role == "tool"`
message, i.e. exactly when the model actually called a tool. So the branch is a
proxy for the correctness gate: rollouts that answered are counted one way,
rollouts that never answered (always scored wrong) are counted the other and
carry every framing token. The wrong arm is inflated by tens of tokens purely
by measurement.

Because `AdvantageWeightedComposer` z-scores per prompt-group, that constant
offset between the two subsets shifts every rollout's normalised cosine value.
Part of the correct-vs-wrong length structure the reward trains on is an
artifact of which counter ran.

Same class as the `reasoning_content` bug.

### 2.3 `--seeds` shifts the dataset seed base by 1, so replicates share ~99%
of their data

`pipeline/training/train.py:195`

`training.batch --seeds 42 43 44` materializes three configs differing only in
the top-level `seed` (`batch.py:80`). That scalar goes straight through as
`seed_base` to `build_seed_dataset`, which mints `seed..seed+size-1`. At
`size: 500` the three runs share 499, 498 and 499 of 500 training questions.
`run_agentic_eval` sets `seed_base = seed + 100000` (`agentic_eval.py:363`), so
the three eval sets share 99, 98 and 99 of 100 questions, and eval is greedy so
shared questions decode identically.

The replicates vary the optimizer RNG and nothing else. Pooling them (as the
poly seed-43 replication did) counts the same questions up to three times and
reports a CI far tighter than the data supports.

Fix: `seed_base = seed * size`, or an explicit disjoint offset.

### 2.4 Over/underthinking rates use a per-run percentile threshold, so they
cannot detect compression

`pipeline/eval/metrics.py:160`

Verified numerically: 100 samples at mean 1146 tokens, and the same samples
with every `n_tokens` halved, both return `overthinking_rate 0.2424` and
`underthinking_rate 0.0909`. Bit-identical, because the P75/P10 threshold moves
with the distribution.

`agentic_eval.py:424` calls `compute_metrics(results)` with no threshold
overrides, and `load_reference_thresholds` (`metrics.py:358`), the function
that exists to pin a fixed cross-run threshold, has zero callers anywhere.

So `eval_report.md` prints an overthinking rate per arm that cannot detect the
compression E2 exists to produce, and a reader comparing two arms is comparing
two different thresholds.

### 2.5 The entropy reward conditions on a prompt missing the tool schema

`pipeline/training/rewards/token_entropy.py:77`

Every rollout prompt in this pipeline is rendered with the adapter's tool spec
injected. `_prompt_to_text` calls `apply_chat_template` with no `tools=`
argument, then concatenates `prompt_ids + completion_ids` and slices logits at
`[p_len-1 .. p_len+c_len-2]`. The forwarded sequence is one the policy never
generated from: hundreds of tokens of tool-schema context are absent. H_t is
the entropy of the completion given a truncated context.

e19 (the only entropy arm) reported null-to-harmful on this quantity.

Worse in multi-turn: `comp_ids` comes from `kwargs['completion_ids']`, which
`CosineLengthReward._is_multiturn` documents as including injected tool-result
tokens, so entropy would average over text the env wrote. The cosine reward
guards against exactly that; this one does not.

---

## Tier 3: operational and robustness

### 3.1 `--base-model` writes into the trained run's output directory

`pipeline/eval/runner.py:51`

The deleted `run_eval(baseline=True)` path routed base-model artifacts to
`runs/_baselines/<slug>/` (surviving artifacts: `pipeline/runs/_baselines/qwen3-1.7b/`,
`.../qwen-7b/`). Commit `e1f1cdf` deleted `run_eval`; commit `37f9b74` re-added
the condition as `--base-model` but resolves `run_dir = os.path.join("runs", exp_id)`
unconditionally.

`python -m eval.runner --config configs/e27-... --base-model` overwrites e27's
`eval_report.json`, `eval_report.md`, `episodes_held_out.jsonl` and
`episodes_shifted.jsonl` with untrained-model numbers. No backup, no warning.
The only mitigation in the repo is a prose note in RUNNING.md's Traps section.

(`e0b-browsergym-base-llama3_2-1b.yaml` has its own `experiment_id`, so it is
not affected. The hazard is running `--base-model` against a trained run's
config.)

### 3.2 `_is_real_report` checks a `smoke` key nothing ever writes

`pipeline/training/batch.py:176`

`_is_real_report` returns False only for `status in (skipped, error)` or a
truthy `smoke` key. `run_agentic_eval` writes its report
(`agentic_eval.py:428-441`) with keys experiment_id / model_slug / seed /
compose_method / mode / checkpoint / results. No `status`, no `smoke`.

So a `--smoke` batch leaves a 4-episode report classified as real, and the
follow-up unattended `--train --eval` prints `skip: eval_report.json exists`
and never runs the 100-episode eval. The 4-episode numbers flow into
`eval.plots` and any comparison table.

The train side does have a marker (`train.py:204-205` writes
`checkpoint-final/.smoke`). `tests/test_batch_skips.py:14-16` passes only
because it hand-builds a `{"smoke": True}` dict the production writer never
produces.

### 3.3 The `.smoke` marker is never cleared, bricking the run for the batch
runner

`pipeline/training/train.py:205`

`save_lora` calls `save_pretrained`, which does not clear the directory, and
`--overwrite` only bypasses the FileExistsError without deleting anything.

After a smoke batch: `checkpoint-final/.smoke` is permanent. Next
`training.batch cfg --train --eval` without `--force`: `_is_real_checkpoint`
returns False so train is not skipped, `_run_train_phase` omits `--overwrite`
because `force` is False, `train.py:151-155` sees `checkpoint-final/` and
raises FileExistsError. Both attempts burn, eval is skipped, `_write_eval_stub`
declines because the stale smoke report is there. The run is unrunnable through
the batch runner without `--force`, and what is left on disk is a 3-step
checkpoint plus a 4-episode report under a real experiment_id.

### 3.4 `validate_config` has no whitelist for `training` keys

`pipeline/training/config_schema.py:270`

The file closes the silent-passthrough gap for top level, rewards, eval and
`training.env_config` (that last one added specifically to catch `datsaet`).
There is no equivalent set for the keys of `training` itself.

`max_prompt_lenght: 6144` in a max_seq 8192 config passes validation, is
ignored, and `_grpo_config` falls back to `max_seq // 2` = 4096, so
`max_completion_length` becomes 4096 instead of 2048. The run trains at a
geometry its own config and description contradict, and the frozen
`runs/<exp>/config.yaml` records the typo rather than the value used.

Same silent fallback for `n_rollouts` (the GRPO group size, hence the advantage
denominator), `micro_batch_size`, `kl_beta`, `save_steps`.

Worth grepping every existing config against the real key set before trusting
any frozen geometry.

### 3.5 `plots.load_report` raises on every multi-split report

`pipeline/eval/plots.py:72`, `:250`

With `split_name=None`, line 72 evaluates
`_SPLIT if _SPLIT in results or len(results) != 1 else next(iter(results))`.
e27/e28/e29 declare `held_out` and `shifted`, so `"agentic" in results` is
False but `len(results) != 1` is True, leaving `wanted = "agentic"`, and line
74 raises.

`make_figures` calls `load_report(p)` positionally and `main()` defines no
`--split`, so the parameter is unreachable from the CLI. No figure can be
produced for the browsergym campaign, and a mixed glob dies after the first
browsergym entry, losing the figures that would have worked.

### 3.7 `--smoke` overwrites a harvested run's results

Not found by the review. Found by hitting it: running
`eval.runner --config configs/e27-... --smoke` to verify the fixed eval loop
overwrote e27's `eval_report.json`, `eval_report.md` and both
`episodes_*.jsonl` with 4-episode smoke output.

Same hazard class as 3.1, and worse in one respect: the report can be
snapshotted beforehand, but the trajectory records cannot be reconstructed from
anything except that report's `samples` array. Here they were recoverable
because the snapshot existed and carried all 100 per-episode records; without it
they would be gone.

The 3.2 fix (write a `smoke` marker) makes the batch runner refuse to *read*
such a report as finished work. Nothing stopped it being *written*.

### 3.6 Only `TypeError` is caught around tool dispatch

`pipeline/eval/agentic_eval.py:197`

`_run_multiturn_episodes` wraps `getattr(env, name)(**(args or {}))` in
`except TypeError` only. An HTTP error from the env server, a
`ConnectionResetError` or a pydantic validation error propagates out of the
loop, past `finally: server.stop()`, and kills the process before
`_write_episodes` runs. `_write_episodes` is called only after the whole split
finishes, so a hiccup on episode 95 of 100 destroys all 95 trajectories and
writes no jsonl at all.

That contradicts the file's own durability rationale at lines 256-261, and
costs several GPU-hours per occurrence on the e27/e28/e29 geometry.

Secondary defect in the same handler: on a caught `TypeError`,
`calls.append(name)` still runs (line 201), so a call that never reached the
env counts toward `n_steps` and inflates `mean_verification_depth` in the RQ2
panel.

---

## Checked and refuted

Eight candidates were raised and killed by verification. Recorded so they are
not re-litigated:

- `compose.py:23,24` - two separate claims that per-prompt-group z-scoring
  groups incorrectly because seed rows share a prompt. Refuted.
- `agentic_eval.py:55` - that the untagged scan attempting a decode at every
  `{` is itself the defect. Refuted (the real defect is 2.1, the relabelling).
- `utils.py:12` - that `extract_content` still ignores `reasoning_content`.
  Refuted; `model_token_count` is the path that matters and it was fixed.
- `finqa/adapter.py:114` - that local csv re-read desyncs from the server's
  shuffle. Refuted.
- `textarena/adapter.py:75` - reward/done read off `result.observation`.
  Refuted.
- `agentic_eval.py:34` - that scoring the two lineages by different rules
  confounds e0b. Refuted.
- `grpo_runner.py:77` - that `training.max_prompt_length` is never forwarded to
  `GRPOConfig`. Refuted.

## Not included

The remaining verified-but-omitted findings are dead code, doc drift, and one
latent NaN path in `compose.py` (unbiased `std()` returns NaN for a
single-element group and the `std < 1e-6` guard cannot catch it, because every
comparison against NaN is False). TRL currently absorbs it; it becomes live if
a group of size 1 ever reaches the composer.

---

## Suggested fix order

Ordered by what unblocks the most work, not by severity:

1. **2.1** (parser relabelling) - uncommitted, cheapest, and blocks every
   future eval.
2. **1.2 + 1.3** (eval budget and context) - then re-eval e27. Nothing in the
   browsergym campaign can be compared across an eval-code change otherwise.
3. **3.1** (`--base-model` clobber) - one line, prevents losing e27.
4. **1.1** (done-guard + turn cap) - required before any further reasoning_gym
   run.
5. **2.2, 2.3, 2.4, 2.5** - correctness of the measurements the thesis reports.
6. **3.2 through 3.6** - operational, do in one cleanup pass.
