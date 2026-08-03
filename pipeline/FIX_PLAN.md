# Fix plan for the 2026-08-03 code review

> **Status: all 16 fixes applied and verified, 2026-08-03.**
> 236 tests locally (was 195), 246 on the box including the tokenizer-backed
> parser tests. The eval loop was exercised end to end against a real
> checkpoint. What remains is GPU work, not code: retrain + eval e27, then
> launch e28/e29. See "Verification results" at the bottom.
>
> One extra fix, C7, was added during the work: `--smoke` could overwrite a
> harvested run's results. It was found by hitting it (e27's episode files were
> overwritten and had to be reconstructed from a snapshot).


Input: `pipeline/CODE_REVIEW.md` (15 findings). Every finding below was
re-verified by reading the source directly, not taken from the review agents.

Three facts were measured on the box before writing this plan, because the fix
design depends on them:

1. `trl.chat_template_utils.parse_response(tokenizer, ids) -> dict` exists and
   works once `add_response_schema(tokenizer)` has been called. It is the exact
   function training uses.
2. It behaves correctly on all five cases that matter:

   | input | result |
   |---|---|
   | Qwen3, complete tagged call | `tool_calls` present, `reasoning_content` populated |
   | Qwen3, call truncated before `</tool_call>` | **no `tool_calls`**, raw text in `content` |
   | Qwen3, JSON quoted inside `<think>` | **no `tool_calls`**, quoted JSON stays in `reasoning_content` |
   | Llama 3.2, untagged call | `tool_calls` present, `parameters` normalised to `arguments` |
   | Llama 3.2, plain prose | no `tool_calls` |

3. Qwen3's chat template preserves `reasoning_content` when re-rendering a
   prior assistant turn, so appending the parsed message and re-rendering
   reproduces the context training produced.

Consequence: findings 2.1 and 1.3 collapse into one deletion. The hand-rolled
parser goes away entirely and eval scores by literally the same function
training does.

---

## Ordering and gates

Two hard constraints. Everything else is free to reorder.

**Gate 1 - all eval-measurement fixes land together, then e27 is re-evaluated.**
Fixes A1, A2, A3, A4, A5 all change what a number in `eval_report.json` means.
Landing them piecemeal produces reports that cannot be compared to each other.
Land all five, re-eval e27, then launch e28/e29. Nothing before that gate is
comparable to anything after it.

**Gate 2 - e27 is retrained, not just re-evaluated** (decision D1 below), so
the retrain must come after Group B as well as Group A. Group B changes what
the training reward measures; retraining before it would burn 12 hours on a
checkpoint that then needs redoing.

Sequence:

```
1. Group C (schema, tooling, robustness)   - no effect on numbers, land first, cheap
2. Group A (eval measurement)              - all five together, then smoke-check
3. Group B (training correctness)          - includes the seed-block change
4. Retrain + eval e27                      - ~12h GPU
5. Launch e28/e29                          - ~20h GPU
```

Group C first because it is risk-free and gets the test suite and the config
guard in place before the changes that matter.

---

## Group A: eval measurement

All five land together. Gate 1.

### A1 + A2. Replace the hand-rolled parser with TRL's `parse_response`

Fixes CODE_REVIEW 2.1 (truncation relabelling) and 1.3 (context amnesia).

**Delete** from `pipeline/eval/agentic_eval.py`: `_TOOL_CALL_RE` (line 17),
`_iter_tool_payloads` (20-64), `_call_args` (67-77). That removes the
uncommitted change in full.

**Replace** the two parsers with thin readers over the parsed message:

```python
from trl.chat_template_utils import add_response_schema, parse_response

def _first_tool_call(msg: dict) -> tuple | None:
    """(name, arguments) of the first tool call in a parsed assistant message."""
    for call in (msg.get("tool_calls") or []):
        fn = (call or {}).get("function") or {}
        name = fn.get("name")
        if name is not None:
            args = fn.get("arguments")
            return str(name), (args if isinstance(args, dict) else {})
    return None
```

`gen_fn` and `gen_turn` call `parse_response(tokenizer, comp_ids.tolist())`
instead of `tokenizer.decode(...)` plus a regex.

`add_response_schema(tokenizer)` goes next to the tokenizer load
(`agentic_eval.py:331`). It is what attaches the `response_schema` that
`parse_response` needs; without it the call raises `AttributeError`.

**In `_run_multiturn_episodes`**, append the parsed message itself:

```python
messages.append(msg)                                    # not {"content": "", ...}
messages.append({"role": "tool", "content": str(feedback)})
```

so `gen_turn` must return the parsed dict alongside `(name, args, n_tokens)`.

**Deletes** `tests/test_agentic_episode_eval.py` lines 60-97 (the 7 untagged
tests written this session) - they test a function that no longer exists.

Verification:
- New unit test asserting a truncated Qwen completion yields no tool call and
  `stop_reason == "hit_generation_cap"`. This is the regression guard for the
  finding. It needs a real tokenizer, so it runs on the box, not `.venv-test`.
- New unit test asserting a `<think>`-quoted JSON object yields no tool call.
- New unit test asserting the appended assistant message carries the
  reasoning text (guards 1.3 from regressing).
- Existing `test_multiturn_eval.py` must be updated for the `gen_turn` return
  shape - update, do not delete.

### A3. Make `max_new_tokens` a trajectory budget, not a per-turn budget

Fixes CODE_REVIEW 1.2.

In `_run_multiturn_episodes`, track the spend and pass what is left:

```python
budget = gen_cap
for _ in range(max_turns):
    if budget is not None and budget <= 0:
        stop_reason = "hit_generation_cap"
        break
    msg, name, args, n_tok = turn_fn(messages, budget)
    total_tokens += int(n_tok)
    if budget is not None:
        budget -= int(n_tok)
    ...
```

and `gen_turn(messages, budget)` passes `max_new_tokens=budget` to `generate`.

`_no_call_reason(n_tok, budget)` then compares the turn against the budget it
actually had, so a turn that consumed the whole remainder is correctly labelled
`hit_generation_cap`.

Verification:
- Unit test with a fake `turn_fn` returning 300 tokens per turn, `gen_cap=1000`,
  `max_turns=8`: asserts the episode stops after 4 turns with
  `stop_reason == "hit_generation_cap"` and `n_tokens <= 1000`.
- Unit test that the single-turn path is unchanged.

### A4. Report a correct-only token statistic

Fixes CODE_REVIEW finding 6.

`_mean_ci_on_correct` already exists in `eval/plots.py:97` with the right
rationale in its docstring, but lives where no report can reach it. Move it into
`eval/metrics.py`, add `mean_token_count_correct` (+ CI) to `EvalMetrics`,
serialize it in `_metrics_to_dict`, and print it in `_report_md` next to the
pooled mean. Keep the pooled mean - it is not wrong, it is just not the
efficiency number.

`plots.py` imports the moved function rather than keeping its own.

Verification:
- Unit test: 4 correct at 100 tokens, 4 wrong at 900 -> pooled mean 500,
  correct-only mean 100.
- Assert `mean_token_count_correct is None` when there are no correct episodes
  (not 0.0, which would read as "perfectly efficient").

### A5. Pin the over/underthinking thresholds to a reference run

Fixes CODE_REVIEW 2.4.

`load_reference_thresholds` (`metrics.py:358`) already does exactly this and has
zero callers - it is dead code that was written for this purpose.

- Add `reference_report` to `_KNOWN_EVAL_KEYS` in `config_schema.py`.
- In `run_agentic_eval`, when `eval.reference_report` is set, call
  `load_reference_thresholds(path)` once and pass the per-split thresholds into
  `compute_metrics`. Splits absent from the reference fall back to the
  percentile, and the report records which threshold was used (the field
  `underthinking_threshold` / `overthinking_threshold` already exists on
  `EvalMetrics` - just make sure it is serialized).
- Point e27/e28/e29 at the e0 report.

Verification:
- The numeric check from the review, as a test: 100 samples, and the same
  samples with every `n_tokens` halved. Without an override both return the
  same rate (documents the old behaviour); with an override the halved set
  returns a strictly higher underthinking rate. That test is the finding.

---

## Group B: training correctness

### B1. Add the done-guard to the reasoning_gym adapter and set a turn cap

Fixes CODE_REVIEW 1.1.

```python
def answer(self, answer: str) -> str:
    if self.done:
        return "Episode already finished; your answer was recorded."
    ...
```

Matching `textarena:69`, `finqa:161`, `repl:78`, `browsergym:94`.

Separately, `grpo_runner.py:125-127` leaves `max_tool_calling_iterations` unset
when no `max_turns` is configured, which TRL reads as `sys.maxsize`. Give
single-step domains an explicit cap of 1 rather than relying on the guard alone:
belt and braces, and it makes the unbounded-loop failure impossible rather than
merely harmless.

Verification:
- Unit test on the adapter with an injected fake client: two `answer` calls,
  assert the client was stepped once and `self.reward` is the first answer's
  score.
- Grep every reasoning_gym config for `max_turns`; add it where missing.

### B2. Make the cosine reward use one ruler

Fixes CODE_REVIEW 2.2.

Delete the branch. `_n_tokens` becomes:

```python
def _n_tokens(self, completion) -> int:
    return model_token_count(completion, self.tokenizer)
```

and the `completion_ids` plumbing in `__call__` (lines 79-86) goes with it. One
counter, both arms, no correlation with correctness.

The `len(completion_ids_i)` path exists to keep "reasoning_gym numbers
identical" to earlier runs. Those runs are void, so the continuity it protects
is worth nothing.

Verification:
- Unit test: a completion that called a tool and one that did not, both with
  identical assistant content, must produce the same token count. Under the
  current code they differ by the framing tokens - that test fails before the
  fix and passes after, which is the point.

### B3. Give each seed a disjoint block of questions

Fixes CODE_REVIEW 2.3. **See "Decisions needed" - this one costs GPU hours.**

```python
_SEED_BLOCK = 1_000_000   # must exceed every eval seed_offset

def seed_block(seed: int) -> int:
    return int(seed) * _SEED_BLOCK
```

- `train.py:195`: `seed_base=seed_block(seed)`
- `agentic_eval.py:363`: `seed_base = seed_block(seed) + split["seed_offset"]`

Seed 42 gets train `42_000_000..42_000_499`, held_out `42_100_000..`, shifted
`42_200_000..`; seed 43 gets the 43-million block. No overlap anywhere.

Verification:
- Unit test asserting the train and eval seed sets of seeds 42, 43, 44 are
  pairwise disjoint, for both split offsets.
- One-off check that reasoning_gym and browsergym accept 8-digit seeds and still
  produce distinct questions (browsergym does `tasks[seed % len(tasks)]`, which
  is fine, but the instance seed goes to MiniWoB and should be confirmed rather
  than assumed).

### B4. Entropy reward: delete it

Fixes CODE_REVIEW 2.5.

The reward conditions on a prompt rendered without `tools=`, so H_t is the
entropy given a context the policy never saw. Fixing it properly needs two
changes: thread the domain's tool spec into `_prompt_to_text`, and mask
`completion_ids` to assistant-only spans for multi-turn domains (where it
currently averages over text the env wrote).

That work buys nothing the thesis uses. The exposé's conditions are E0, E1, E2
(length), E3 (non-termination), optional E4 (combined). Entropy is not among
them. e19 is the only run that used it and its numbers are void twice over -
wrong conditioning, plus the truncation confound.

Delete `training/rewards/token_entropy.py`, its `REWARD_REGISTRY` entry, its
`_KNOWN_REWARD_KEYS` entry, its `_KNOWN_REWARD_SUBKEYS` entry, its
`warn_inert_scalars` branch, and `configs/e19-*`.

Alternative if it should stay: fix `tools=` and add the assistant-span mask,
roughly a day including tests. Say so and I will do that instead.

Verification:
- Test suite passes with the registry entry gone.
- Grep confirms no config references `token_entropy`.

---

## Group C: schema, tooling, robustness

Land first. None of these change a number.

### C1. Whitelist `training` keys in `validate_config`

Fixes CODE_REVIEW 3.4.

Add `_KNOWN_TRAINING_KEYS` alongside the four whitelists that already exist,
and check it the same way. From the code that reads them: `mode`, `env`,
`env_config`, `env_server`, `max_prompt_length`, `max_steps`, `save_steps`,
`n_rollouts`, `batch_size`, `micro_batch_size`, `learning_rate`, `kl_beta`,
`temperature`, `weight_decay`, `warmup_ratio`, `dataset_size_limit`,
`gradient_accumulation_steps`.

Also: `eval/runner.py:29` loads the config and never calls `validate_config`.
Add the call, so an eval-only invocation gets the same guard as training.

Verification:
- Test that `training: {max_prompt_lenght: 6144}` raises.
- Run the new validator over every config in `configs/` and every frozen
  `runs/*/config.yaml`. Any hit is a run whose recorded geometry is not the
  geometry it trained at - report those, do not silently fix them.

### C2. Mark smoke eval reports

Fixes CODE_REVIEW 3.2.

`run_agentic_eval` writes the report without the `smoke` key that
`_is_real_report` looks for. One line: `"smoke": bool(config.get("_smoke"))` in
the report dict.

Verification:
- Test that a report written with `_smoke` set is rejected by `_is_real_report`.
- The existing `test_batch_skips.py` hand-builds the dict the writer never
  produced; rewrite it to go through the real writer.

### C3. Clear the `.smoke` marker on a real train

Fixes CODE_REVIEW 3.3.

In `train.py`, after `save_lora`, remove `checkpoint-final/.smoke` when not in
smoke mode. Currently it survives forever and bricks the run for the batch
runner.

Verification:
- Test: write a `.smoke` marker, run the non-smoke save path, assert the marker
  is gone and `_is_real_checkpoint` is True.

### C4. Guard `--base-model` against overwriting a trained run

Fixes CODE_REVIEW 3.1.

`runner.py:36` computes `run_dir = runs/<exp_id>` regardless of `--base-model`,
so pointing it at a trained run's config destroys that run's harvested report.

Minimal fix: when `--base-model` is set and `run_dir` already contains a
`checkpoint-final/`, refuse and tell the user to use a dedicated E0 config
(`e0b-...` is the pattern). Not a new directory scheme - just a refusal.

Verification:
- Test that `--base-model` against a run dir containing `checkpoint-final/`
  raises before any model loads.

### C5. Make `plots` handle multi-split reports

Fixes CODE_REVIEW 3.5.

Add `--split` to `main()` and thread it through `make_figures` to
`load_report`. Default: if the report has multiple splits and none was named,
draw a figure per split rather than raising. Also catch per-run failures inside
`make_figures` so one bad run does not kill the batch.

Verification:
- Run `python -m eval.plots runs/e27-browsergym-e1-baseline-qwen3-1_7b -o /tmp/p`
  and confirm it writes figures for both splits. It currently raises.

### C6. Do not lose a split's episodes on an env error

Fixes CODE_REVIEW 3.6.

Two changes in `_run_multiturn_episodes`:

- Catch `Exception`, not just `TypeError`, around the tool dispatch. A
  `ConnectionResetError` from the env server becomes feedback and ends that
  episode, rather than killing the process.
- Move `calls.append(name)` so a call that never reached the env is not counted
  in `n_steps` (it currently inflates `mean_verification_depth`).

And in `run_agentic_eval`, write `episodes_<split>.jsonl` incrementally rather
than after the whole split returns, so a hard crash keeps what was collected.

Verification:
- Test with a fake env whose tool raises `RuntimeError`: assert the episode
  ends, is recorded, and the remaining episodes still run.
- Test that a `TypeError`-ing call does not increment `n_steps`.

---

## Decisions (resolved 2026-08-03)

**D1. B3 lands now, and e27 is retrained.** Everything ends up on one seed
scheme with no special cases and no caveat to write into the thesis. Cost:
~10h GPU to retrain e27 on top of the ~2h re-eval.

This changes the sequence - e27 is retrained rather than only re-evaluated, and
that retrain must happen after both Group A and Group B land, since Group B
changes what the training reward measures:

```
1. Group C          - no effect on numbers, cheap
2. Group A          - eval measurement, all five together
3. Group B          - training correctness, including B3
4. Retrain + eval e27  (~12h GPU)
5. Launch e28/e29      (~20h GPU)
```

The Group A smoke check (layer 2) still runs right after Group A, against the
current e27 checkpoint, so the eval loop is proven before the retrain consumes
12 hours on it.

**D2. B4 deletes the entropy reward.** Confirmed.

---

## Verification plan

Three layers.

**Layer 1 - unit tests, `.venv-test` on the Mac.** Every fix above lists its
test. Roughly 20 new tests. The suite is at 195 today; target ~215 with none
removed except the 7 that test deleted code.

**Layer 2 - a smoke run on the box.** Group A changes the generation loop, and
no CPU test exercises a real tokenizer plus a real env server. After Group A
lands:

```
python -m training.batch configs/e27-browsergym-e1-baseline-qwen3-1_7b.yaml --eval --smoke
```

Check in the resulting report: `stop_reasons` contains `hit_generation_cap` for
at least one episode if any episode ran long, `mean_token_count_correct` is
present, and no episode has `n_tokens` above the configured budget.

**Layer 3 - a provenance check on e27, run twice.** e27 is being retrained *and*
re-evaluated, so a single diff against the old report would mix two causes.
Split it:

1. Copy `runs/e27-.../eval_report.json` to `eval_report_pre_fix.json`.
2. After Group A lands, re-evaluate the **existing** e27 checkpoint on the fixed
   eval loop (~2h). Diff against the snapshot. This isolates the eval-side
   changes with the policy held fixed. Expected: accuracy down slightly (the 4
   episodes that got 1.8x budget lose it), `hit_generation_cap` appears in
   `stop_reasons`, `mean_token_count_correct` is a new field.
3. After Group B lands, retrain e27 and evaluate again. Diff against step 2.
   That difference is attributable to the training-side fixes alone.

The 2h in step 2 is what makes the 12h retrain interpretable. Anything moving
in an unexpected direction means a fix is wrong, and these diffs are the only
place it will show.

Layers 1 and 2 prove the code does what I wrote; layer 3 proves the code does
what the thesis needs.

---

## Effort

| Step | Work | GPU |
|---|---|---|
| Group C (6 fixes) | ~half a day | none |
| Group A (5 fixes) | ~a day | ~2h smoke |
| e27 re-eval on old checkpoint (layer 3 step 2) | - | ~2h |
| Group B (4 fixes, incl. entropy deletion) | ~half a day | none |
| e27 retrain + eval | - | ~12h |
| e28/e29 launch | - | ~20h |

---

## Verification results (2026-08-03)

### Layer 1 - unit tests

236 local (`.venv-test`), 246 on the box, zero failures. Was 195. Every fix
carries a test that fails without it; several encode the defect explicitly so it
cannot come back quietly:

- `test_same_ruler_for_answered_and_unanswered_rollouts` - the cosine two-rulers bug
- `test_percentile_thresholds_cannot_detect_compression` next to
  `test_pinned_thresholds_do_detect_compression` - the threshold defect and its fix
- `test_old_scheme_would_have_overlapped` - the 499-of-500 seed overlap
- `test_budget_is_spent_across_turns_not_renewed` - asserts the exact budget
  sequence [1000, 700, 400, 100] and `hit_generation_cap`
- `test_second_answer_does_not_overwrite_the_reward` - the done-guard
- `tests/test_response_parsing.py` (box only, 8 tests) - the wire-format
  behaviour the whole parser fix rests on, against real Qwen3 and Llama
  tokenizers. Chief among them: a Qwen call truncated before `</tool_call>`
  yields no tool call, and a JSON object quoted inside `<think>` is not executed.

### Layer 2 - smoke on the box

`eval.runner --smoke` against the e27 checkpoint, 4 episodes per split. Every
changed behaviour visible in the output:

| what | observed |
|---|---|
| trajectory budget (A3) | no episode above 4096; per-episode 1296/504/1384/349 |
| correct-only mean (A4) | held_out 745.7 vs pooled 883.2; shifted 239.0 vs 423.2 |
| pinned thresholds (A5) | held_out P10=243.9 P75=2484, shifted P10=261.7 P75=2017.5, from e0 |
| seed blocks (B3) | seed_base 42100000 / 42200000 |
| smoke marker (C2) | `"smoke": true` in the report |
| stop reasons | `{env_done: 3, no_tool_call: 1}` - a 368-token 0-step episode
  labelled `no_tool_call`, not `hit_generation_cap`, because it stopped well
  under budget. That distinction is the point of the whole exercise. |

Multi-turn episodes (2 steps) ran, so the `parse_response` path and the
message-append both work against a real model, not just hand-built dicts.

### Layer 3 - provenance

Step 1 done: `runs/e27-.../eval_report_pre_fix.json` holds the pre-fix report
(100 episodes per split, held_out 0.780, shifted 0.790). Steps 2 and 3 are GPU
work and remain open.

### Damage note

The layer-2 smoke was first run against e27's own config, which overwrote its
report, `.md` and both `episodes_*.jsonl`. The report and both episode files
were reconstructed exactly from the pre-fix snapshot (all 100 per-episode
records were in its `samples` array); the `.md` was deleted rather than
regenerated, since a re-eval rewrites it and a wrong one is worse than none.
C7 now refuses that command.

## What remains

Code is done. The rest is GPU time, in this order:

1. Re-eval the existing e27 checkpoint on the fixed loop (~2h), diff against
   `eval_report_pre_fix.json`. Expected: accuracy down slightly, a
   `hit_generation_cap` bucket appears, `mean_token_count_correct` is new.
2. Retrain e27 under the fixed training path, then eval (~12h).
3. Launch e28/e29 (~20h).

Step 1 before step 2: it isolates the eval-side changes with the policy held
fixed, which is what makes the retrain interpretable.
