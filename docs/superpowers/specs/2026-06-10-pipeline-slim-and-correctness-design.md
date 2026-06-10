# Pipeline: slim and lock to the research goal

Date: 2026-06-10
Source: `PIPELINE_REVIEW.md` (149 verified issues: 5 HIGH, 46 MEDIUM, 98 LOW)

## Research goal

Can an efficiency reward (token length, and token entropy) make a reasoning model
produce shorter outputs without losing accuracy, and does the reward-composition
method matter?

Two measurable contrasts carry the thesis:

- **e0 vs e1**: does an efficiency reward shorten reasoning without dropping accuracy?
- **e2 vs e3**: does advantage-weighted composition beat a naive weighted sum on the
  same reward stack?

Every line of the pipeline must serve one of these. The review confirmed the core
math is correct (per-group z-scoring, the cosine reward shape against Wu/Yeo 2025,
Wilson CIs, the two-level paired bootstrap, Holm correction). The 149 issues sit at
the edges: string parsing, config plumbing, file-existence logic, doc drift. So this
work is mostly deletion, then a focused correctness pass, then locking the docs to
the code.

## Locked decisions

| Decision | Verdict |
|---|---|
| `effort_proxy` reward | Delete entirely (code, configs, schema, tests, thesis framing) |
| vLLM twin configs | Delete; vLLM becomes a `--vllm` runtime flag |
| `CodingDomain` | Delete entirely |
| Linear length ablation | Cut configs and the now-dead linear code path |
| `token_entropy` | Keep as a first-class studied signal |

## Target experiment matrix

Five configs, one per experiment, no backend twins:

| Config | Reward stack | Compose | Role |
|---|---|---|---|
| `e0-baseline-math-qwen-7b` | accuracy + format | advantage_weighted | baseline |
| `e1-token-length-qwen-7b` | + cosine length | advantage_weighted | length treatment |
| `e1-token-entropy-qwen-7b` | + entropy | advantage_weighted | entropy treatment |
| `e2-multi-signal-qwen-7b` | + length + entropy | advantage_weighted | combined |
| `e3-ablation-naive-sum-qwen-7b` | + length + entropy | naive_sum | composition ablation |

`--vllm` flips the training-rollout backend at runtime. The eval block is identical
across all five and uses the graded GSM8K-tail capability floor (the T2.6 mode), which
resolves the HIGH issue where that floor was missing from most configs.

The fork-mask entropy variant ships as a knob on the entropy reward, not as a separate
config. Drop the dedicated `e1-token-entropy-forkmask` config (open to revision).

## Plan, in execution order

Slim before fixing, so no effort goes into a bug inside code that is about to be
deleted.

### Phase 1: slim

Deleting this code also deletes every bug inside it (~35 issues), with no fix, test,
or doc change needed for any of them.

- **effort_proxy**: remove `training/rewards/effort_proxy.py`, its `REWARD_REGISTRY`
  entry and `_build_effort_proxy`, the schema whitelist entry and the two inert-knob
  warning branches, the `enabled: false` blocks in every config, the `test_guardrail`
  references, and the README and CLAUDE.md sections.
- **coding domain**: remove `domains/coding/`, its dispatch in `build_domain`, and the
  README and CLAUDE.md stub notes. `Domain` keeps one concrete subclass.
- **vLLM twins**: delete the eight `-vllm` configs. Add `--vllm` to `training.train`
  (and pass it through `training.batch`) so it injects `use_vllm`,
  `gpu_memory_utilization`, and `enforce_eager` at runtime. The backend can no longer
  drift from the reward config it tests.
- **linear ablation**: delete the two linear-ablation configs and the dead linear path:
  the `-alpha * n` `TokenLengthReward`, the `shape` / `schedule` / `alpha` knobs, and
  `_RewardStepCallback` (its only job was advancing the linear anneal). `token_length`
  becomes cosine-only; `CosineLengthReward` is the single implementation with no `shape`
  switch. Keep `_ComponentMetricsCallback`.
- **dead knobs**: drop `format_exact.reward` (whitelisted but ignored) and the inert
  `reward_scale` / `schedule` boilerplate from the surviving configs.

### Phase 2: fix the training signal

These distort the reward whose effect the experiments measure. All live in
`domains/base.py` and the math loader. Pure correctness.

- `score_numbers` strips commas from the model answer but not the ground truth, so
  comma-formatted truths score the answer wrong even when it is right. Strip both sides.
- A ground truth of 0 sends correct answers to the worst penalty (-4.5) through a
  divide-by-zero in the ratio branch. Handle zero truth directly.
- `_number_re` captures bare punctuation, leaks past `</SOLUTION>`, and matches
  `<SOLUTION>` mentions inside the chain of thought. Anchor it to `reasoning_end` and
  require a digit, the same way `_solution_re` is anchored.
- An empty extraction against an empty truth scores +5.0, the maximum reward for
  emitting nothing, on exactly the rows with broken labels. Drop unparseable-truth rows
  at load time and guard the empty-extraction case.
- `rewards.<name>: false` silently re-enables the reward with defaults, then crashes
  after the model has loaded. Validate that each reward value is a dict or null.

### Phase 3: fix eval comparability

These shape the numbers that go into the thesis.

- Use one identical eval block across all five configs, with the graded capability
  floor. Resolves the HIGH floor-gap.
- The MMLU letter fallback grabs the first bare A/B/C/D in the text, so the article "A"
  satisfies it and biases baseline scores. Require a SOLUTION block or a stricter letter
  match.
- The capability-floor matcher scans the whole chain of thought when the SOLUTION tags
  are absent, and marks sentence-final-period answers wrong. Tighten both.
- Per-probe generation-budget floors silently override `eval.max_new_tokens`, so mean
  token count is measured under different truncation budgets across splits. Make the
  budget consistent.
- `vs_base_model` scores the base model with the tag-dependent grader, so the delta
  credits capability for what is mostly format acquisition. Give the baseline a
  format-agnostic extractor (accept a final number or boxed answer without the full tag
  structure) so the delta measures capability. Document that the baseline uses a lenient
  extractor where the trained model uses the tag-anchored one.

### Phase 4: batch robustness

Saves GPU-nights, not validity.

- **Crash recovery (HIGH)**: make `checkpoint-final/` the sole clobber trigger so a
  mid-training OOM does not deadlock every retry on the frozen `config.yaml`.
- **Stub-aware skips**: the three skip predicates test only file existence, so they
  cannot tell a real artifact from a stub or a smoke run. Switch to content/status
  checks. This fixes the stub-baseline-ownership HIGH, the stub-blocks-eval-on-resume
  bug, and the smoke-run-skips-real-batch bug.
- **Baseline dedup**: replace the symlink-sharing mechanism. Run the baseline once per
  `model.slug` into a single canonical location, and have every report reference that
  path directly. Keeps the one-baseline GPU saving and drops every symlink failure mode
  (write-through, dangling links, cross-dataset mismatch).

### Phase 5: tests for survivors and the central mechanism

- Fix the failing test (`test_default_shape_is_linear`, stale since the cosine default).
  With cosine-only, rewrite or delete it.
- Add tests the review flagged as missing on load-bearing logic:
  `AdvantageWeightedComposer`'s z-scoring math (the thesis's central mechanism, only
  smoke-tested today), the core reward grading (where the Phase 2 bugs lived), and
  `validate_config`.

### Phase 6: lock docs to code

Fix CLAUDE.md, the README, and docstrings so thesis text written from them is correct:
underthinking is the P10 or reference threshold, not "<=50 tokens"; the accuracy CI is
Wilson, not bootstrap; the capability floor has 6 prompts, not 5; MMLU is zero-shot, not
5-shot; z-scoring cancels any global *positive* scalar. Propagate every Phase 1 deletion.

## Resolved at spec review

- **`vs_base_model`**: lenient format-agnostic baseline extractor (folded into Phase 3).
- **Batch baseline dedup**: single canonical baseline path, no symlinks (folded into
  Phase 4).

## Deliberately out of scope

Low-severity cosmetic and doc-only items the review flagged that do not touch the
research goal, the training signal, eval comparability, or batch robustness. Examples:
plot bin-edge cosmetics, the policy-loss panel label, registry generation-naming. Leave
them unless a phase above touches the same code.

## Success criteria

- `pytest pipeline/tests/ -q` passes, with new tests covering z-scoring, reward grading,
  and `validate_config`.
- `effort_proxy`, `domains/coding/`, the linear path, and all `-vllm` configs are gone;
  no dangling references remain.
- Five configs train and eval through `training.batch`, with and without `--vllm`, and
  produce comparable capability-floor numbers.
- A smoke run over the matrix does not leave artifacts that a later real batch mistakes
  for finished work.
- CLAUDE.md, the README, and docstrings match the code on every metric definition and
  every deleted component.

## Net effect

15 configs to 5. ~40 source files to ~33. The `effort_proxy`, `coding`, vLLM-twin,
linear-shape, and schedule-callback surfaces are gone. The surviving training signal is
correct, the eval is comparable across the matrix, the docs match the code, and the
central composition mechanism is tested.
