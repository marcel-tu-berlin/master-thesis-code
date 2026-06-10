# Pipeline slim and correctness — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the pipeline to the five experiments the thesis needs, fix the reward and eval correctness bugs the review found, and lock the docs to the code.

**Architecture:** Delete first (effort_proxy, coding domain, vLLM twins, the linear length path), so no fix lands in code that is about to go. Then fix the training signal and eval comparability on what survives. Then lock docs. Each task is one commit.

**Tech stack:** Python 3.12, TRL GRPOTrainer, Unsloth, vLLM (GPU rollouts), HuggingFace datasets, numpy/scipy. Tests use pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-pipeline-slim-and-correctness-design.md`

## Environment and verification

This repo trains on a Linux/RTX-4090 box. The current dev box is a Mac with no GPU and no Unsloth/vLLM. An isolated CPU venv at `.venv-test` (datasets, numpy, scipy, pytest, CPU torch — no unsloth/vllm) runs the full logic suite.

- **Local verify (every task that touches logic):**
  `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
  Baseline before this plan: **1 failed, 57 passed** (the stale `test_default_shape_is_linear`).
- **GPU-box verify (training/eval/batch end to end), run by the user on the Linux box:**
  `cd pipeline && python -m training.batch configs/e0-*.yaml configs/e1-*.yaml configs/e2-*.yaml configs/e3-*.yaml --train --eval --baseline --smoke`
  Tasks that can only be checked this way are marked **[GPU-box]**.

Do not claim a task passes without running its verify command and reading the output (superpowers:verification-before-completion).

## File-change map

| File | Change |
|---|---|
| `training/rewards/effort_proxy.py` | delete |
| `training/rewards/token_length.py` | delete (linear path) |
| `domains/coding/` | delete dir |
| `training/rewards/__init__.py` | drop effort + linear; cosine-only `_build_token_length` |
| `training/config_schema.py` | drop effort + linear knobs; reject non-dict reward values |
| `training/train.py` | drop coding import + `_RewardStepCallback`; add `--vllm` |
| `training/batch.py` | `--vllm` passthrough; content-aware skips; canonical baseline path |
| `domains/base.py` | fix score_numbers, score_answer, number extraction, empty-answer guard |
| `domains/math/loader.py` | drop empty-answer rows; route `extract_number` through `extract_answer` |
| `eval/ood_probes.py` | MMLU letter fix; capability-floor match fix; consistent budget |
| `eval/runner.py` / `eval/report.py` | lenient baseline extractor for `vs_base_model` |
| `configs/*.yaml` | delete 10; rewrite 5 survivors; update `_template.yaml` |
| `tests/*` | fix stale test; add z-scoring, grading, validate_config tests |
| `CLAUDE.md`, `pipeline/README.md` | lock to code |

---

## Phase 1 — Slim

### Task 1: Delete the effort_proxy reward

**Files:**
- Delete: `pipeline/training/rewards/effort_proxy.py`
- Modify: `pipeline/training/rewards/__init__.py` (import line 13, `_build_effort_proxy` 98-111, registry entry 122)
- Modify: `pipeline/training/config_schema.py` (`_KNOWN_REWARD_KEYS` line 29, `_KNOWN_REWARD_SUBKEYS` line 53, the `effort_proxy` block in `warn_inert_scalars` 143-155, the docstring mention line 87)
- Modify: `pipeline/tests/test_guardrail.py` (lines 12 comment, 59, 70 effort cases)

- [ ] **Step 1: Read** `tests/test_guardrail.py` in full to see how the two `effort_proxy` cases assert. They test the "inert under advantage_weighted" warning. Re-point them at `token_entropy.reward_scale` (which stays) or delete them if `token_entropy` already has equivalent coverage.

- [ ] **Step 2: Delete the file and remove all references.** Remove the import, builder, and registry row in `__init__.py`; the key in both schema sets and the warning block in `config_schema.py`.

- [ ] **Step 3: Verify no references remain**

Run: `grep -rn 'effort_proxy\|EffortProxy' pipeline/ ; echo "exit: $?"`
Expected: only matches inside `docs/` or none in `pipeline/`. (`grep` exit 1 = clean.)

- [ ] **Step 4: Run the suite**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
Expected: still `1 failed, 57 passed` (the stale test is fixed in Task 6; effort tests now pass or are gone).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(rewards): remove unused effort_proxy reward"
```

### Task 2: Collapse token_length to cosine-only (remove the linear path)

**Files:**
- Delete: `pipeline/training/rewards/token_length.py`
- Modify: `pipeline/training/rewards/__init__.py` (import line 16; `_build_token_length` 40-66)
- Modify: `pipeline/training/config_schema.py` (`_KNOWN_REWARD_SUBKEYS["token_length"]` 41-47; `warn_inert_scalars` token_length block 99-131; docstring 79-93)
- Modify: `pipeline/training/train.py` (`_RewardStepCallback` class 21-29; its wiring 178-181)
- Modify: `pipeline/tests/test_reward_builders.py` (rewrite — see Task 6)

- [ ] **Step 1: Simplify the builder.** Replace `_build_token_length` (lines 40-66) with a cosine-only version:

```python
def _build_token_length(domain, runner, training_cfg, cfg):
    # Cosine length reward (Wu/Yeo 2025): correct -> prefer shorter, wrong ->
    # prefer longer. Non-linear and correctness-gated, so it survives the
    # advantage_weighted per-group z-scoring. This is the only length shape.
    return CosineLengthReward(
        runner.tokenizer,
        domain,
        max_len=int(cfg.get("max_len", 256)),
        r_correct_short=cfg.get("r_correct_short", 1.0),
        r_correct_long=cfg.get("r_correct_long", 0.5),
        r_wrong_short=cfg.get("r_wrong_short", -1.0),
        r_wrong_long=cfg.get("r_wrong_long", -0.5),
    )
```

Remove the `from training.rewards.token_length import TokenLengthReward` import (line 16). Note `max_len` default is now `256` (matches every shipped config; closes the 256-vs-512 drift).

- [ ] **Step 2: Drop the linear knobs from the schema.** In `_KNOWN_REWARD_SUBKEYS["token_length"]` remove `"alpha"`, `"schedule"`, `"shape"`; keep `"max_len"` and the four `r_*` endpoints. A config that still sets `shape`/`alpha`/`schedule` now fails validation with "Unknown sub-keys" — the desired signal.

- [ ] **Step 3: Gut the token_length branch of `warn_inert_scalars`.** Remove lines 99-131 (the whole `tl = rc.get("token_length")` block); the knobs it warned about no longer exist. Update the docstring (79-93) to drop the "Wrong shape" paragraph. Keep the `token_entropy.reward_scale` advantage_weighted warning.

- [ ] **Step 4: Remove the now-dead step callback.** `step()` lives only on the deleted `TokenLengthReward` (confirmed: `grep -rn 'def step' training/rewards/` returns only `token_length.py`). Delete `_RewardStepCallback` (train.py 21-29) and its wiring (178-181: the `step_fns`/`callbacks.append(_RewardStepCallback(...))` block). Keep `_ComponentMetricsCallback`.

- [ ] **Step 5: Verify no references remain**

Run: `grep -rn 'TokenLengthReward\|_RewardStepCallback\|shape.*linear\|RewardStepCallback' pipeline/training pipeline/configs ; echo done`
Expected: no live references in `training/` (configs handled in Task 5).

- [ ] **Step 6: Run the suite** (test_reward_builders still references the deleted class — expect import error here; fixed in Task 6, which should be done together with this task).

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`

- [ ] **Step 7: Commit** (squash with Task 6 if done together)

```bash
git add -A && git commit -m "refactor(rewards): drop linear length path, cosine is the only shape"
```

### Task 3: Delete the coding domain

**Files:**
- Delete: `pipeline/domains/coding/` (dir: `loader.py`, `__init__.py`)
- Modify: `pipeline/training/train.py` (import line 13; `build_domain` coding branch 62-63)

- [ ] **Step 1: Remove the import and dispatch branch.** In `train.py`, delete `from domains.coding.loader import CodingDomain` (line 13) and the `if name == "coding": return CodingDomain()` branch (62-63). `build_domain` keeps `math` plus the `NotImplementedError` fallback.

- [ ] **Step 2: Delete the directory**

```bash
git rm -r pipeline/domains/coding
```

- [ ] **Step 3: Verify**

Run: `grep -rn 'coding\|Coding' pipeline/ --include=*.py | grep -v test ; echo done`
Expected: no `domains.coding` references.

- [ ] **Step 4: Run the suite + commit**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
```bash
git add -A && git commit -m "refactor(domains): remove coding stub, math is the only domain"
```

### Task 4: Make vLLM a `--vllm` flag

**Files:**
- Modify: `pipeline/training/train.py` (argparse + injection in `main`)
- Modify: `pipeline/training/batch.py` (passthrough)

`GRPORunner.__init__` already reads `config["model"]["use_vllm"|"gpu_memory_utilization"|"enforce_eager"]`. The flag injects those keys at runtime so configs carry no backend state.

- [ ] **Step 1: Add the flag and inject.** In `train.py` `main()`, add `parser.add_argument("--vllm", action="store_true", help="Route GRPO rollouts through vLLM fast inference")`. Immediately after `config = load_config(args.config)` (and after `apply_smoke_overrides`), before `validate_config`:

```python
if args.vllm:
    config.setdefault("model", {})
    config["model"]["use_vllm"] = True
    config["model"].setdefault("gpu_memory_utilization", 0.6)
    config["model"]["enforce_eager"] = True
    print("⚠  vLLM fast inference ON (--vllm): gpu_memory_utilization=0.6, enforce_eager=True")
```

- [ ] **Step 2: Read** `batch.py` to find where it builds the `training.train` / `eval.runner` subprocess argv (search for `"--smoke"`). Add `--vllm` to the parser and append it to each subprocess command whenever set, exactly as `--smoke` is threaded.

- [ ] **Step 3: Verify the flag parses** (no GPU needed for argparse)

Run: `(cd pipeline && ../.venv-test/bin/python -m training.train --help 2>&1 | grep -A1 vllm)`
Expected: the `--vllm` help line prints.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(train): add --vllm flag, replacing per-config vLLM knobs"
```

### Task 5: Rebuild the config matrix to five experiments

**Files:**
- Delete: 8 `-vllm` configs + 2 `linear-ablation` configs (10 files)
- Modify: the 5 survivors + `_template.yaml`

Survivors (final names): `e0-baseline-math-qwen-7b`, `e1-token-length-qwen-7b`, `e1-token-entropy-qwen-7b`, `e2-multi-signal-qwen-7b`, `e3-ablation-naive-sum-qwen-7b`.

- [ ] **Step 1: Delete the unwanted configs**

```bash
cd pipeline && git rm configs/*-vllm.yaml configs/e1-token-length-linear-ablation-qwen-7b.yaml
```
(The `-vllm` glob removes 8; the linear-ablation non-vllm removes the 9th; total 10 with the linear-vllm already caught by the glob. Verify with `ls configs/`.)

- [ ] **Step 2: Normalize each survivor.** In all five: remove the `effort_proxy:` block, remove inert `token_entropy.reward_scale` lines under `advantage_weighted`, remove any `token_length.shape/alpha/schedule` lines (cosine is implicit; keep `max_len` + `r_*`). Give every survivor the **same** eval block with the graded capability floor (port from the old `-vllm` configs):

```yaml
eval:
  temperature: 0.0
  do_sample: false
  reference_run: runs/e0-baseline-math-qwen-7b   # omit in e0 itself
  id_split: openai/gsm8k
  id_split_hf_split: test
  ood_probes:
    near: EleutherAI/hendrycks_math
    far: MMLU
    capability_floor: simple
  capability_floor_dataset: openai/gsm8k
  capability_floor_hf_split: test
  capability_floor_limit: 50
  capability_floor_take: tail
```

- [ ] **Step 3: Update `_template.yaml`** — remove the `effort_proxy` and linear `token_length` knobs, soften the "All fields shown" header to "Common fields shown", and show the graded capability-floor keys.

- [ ] **Step 4: Verify every config validates against the new schema** (CPU — `validate_config` is torch-free):

```bash
cd pipeline && for c in configs/e*.yaml; do
  ../.venv-test/bin/python -c "import yaml,sys; from training.config_schema import validate_config; validate_config(yaml.safe_load(open('$c'))); print('ok', '$c')"
done
```
Expected: `ok` for all five; no `Unknown sub-keys` / range errors.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor(configs): five-experiment matrix, no vLLM twins, uniform graded floor"
```

---

## Phase 2 — Fix the training signal (TDD; all CPU-local)

### Task 6: Fix the stale builder tests (unblocks the suite)

**Files:** `pipeline/tests/test_reward_builders.py`

- [ ] **Step 1: Rewrite the test file** for cosine-only:

```python
"""The token_length builder always returns the cosine length reward."""
from training.rewards import _build_token_length
from training.rewards.cosine_length import CosineLengthReward


class StubTok:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


class StubDomain:
    def is_correct(self, completion, ground_truth):
        return False


class StubRunner:
    tokenizer = StubTok()
    config = {"model": {"max_seq_length": 2048}}


def test_builds_cosine_reward_with_defaults():
    fn = _build_token_length(StubDomain(), StubRunner(), {"max_steps": 500}, {})
    assert isinstance(fn, CosineLengthReward)
    assert fn.max_len == 256  # builder default now matches the configs


def test_max_len_is_configurable():
    fn = _build_token_length(StubDomain(), StubRunner(), {"max_steps": 500}, {"max_len": 512})
    assert fn.max_len == 512
```

- [ ] **Step 2: Run just this file**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_reward_builders.py -q)`
Expected: `2 passed`.

- [ ] **Step 3: Run the whole suite**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
Expected: **all passed** (0 failed), assuming Tasks 1-2 are in.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test(rewards): update builder tests to cosine-only token_length"
```

### Task 7: Fix numeric grading bugs in `Domain` (comma asymmetry, zero truth)

**Files:** `pipeline/domains/base.py`, new tests in `pipeline/tests/test_domain_grading.py`

- [ ] **Step 1: Write the failing tests**

```python
# pipeline/tests/test_domain_grading.py
from domains.math.loader import MathDomain

d = MathDomain()

def test_score_numbers_strips_commas_from_truth():
    # comma-formatted ground truth must not crash the float() and score wrong
    assert d.score_numbers("1000", "1,000") == 3.5

def test_score_answer_zero_truth_correct_value():
    # "0.0" vs "0": strings differ, both are zero -> must be max reward, not -4.5
    assert d.score_answer("0.0", "0") == 5.0

def test_score_answer_zero_truth_wrong_value():
    assert d.score_answer("7", "0") == -2.5
```

- [ ] **Step 2: Run to confirm they fail**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_domain_grading.py -q)`
Expected: 3 failed (`-1.5` instead of `3.5`; `-4.5` instead of `5.0`/`-2.5`).

- [ ] **Step 3: Fix `score_numbers`** (base.py line 92): strip commas from the truth side too.

```python
            b = float(truth.strip().replace(",", ""))
```

- [ ] **Step 4: Fix `score_answer` zero-truth** (base.py 74-84): guard before the ratio.

```python
        try:
            ext_num = float(extracted.strip().replace(",", ""))
            truth_num = float(truth.strip().replace(",", ""))
            if truth_num == 0:
                return 5.0 if abs(ext_num) < 1e-9 else -2.5
            ratio = ext_num / truth_num
            if 0.9 <= ratio <= 1.1:
                return 2.0
            if 0.8 <= ratio <= 1.2:
                return 1.5
            return -2.5
        except (ValueError, AttributeError):
            return -4.5
```
(`ZeroDivisionError` is no longer reachable; dropping it from the `except` is fine.)

- [ ] **Step 5: Run tests + full suite**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "fix(rewards): strip commas from truth and handle zero ground truth in grading"
```

### Task 8: Anchor number extraction and require a digit

**Files:** `pipeline/domains/base.py`, `pipeline/domains/math/loader.py`, tests in `test_domain_grading.py`

Route `extract_number` through the already-anchored `extract_answer`, so it can only read inside the real SOLUTION block. Delete the fragile unanchored `_number_re`.

- [ ] **Step 1: Add failing tests**

```python
RS, RE = "<start_working_out>", "<end_working_out>"
SS, SE = "<SOLUTION>", "</SOLUTION>"

def test_extract_number_requires_a_digit():
    t = f"{RS}x{RE}{SS}.{SE}"          # bare punctuation, no digit
    assert d.extract_number(t) is None

def test_extract_number_reads_solution_block():
    t = f"{RS}think{RE}{SS}42{SE}"
    assert d.extract_number(t) == "42"

def test_extract_number_ignores_cot_solution_mention():
    # a <SOLUTION> mentioned inside the CoT must not be captured
    t = f"{RS}maybe {SS}99{SE} no{RE}{SS}7{SE}"
    assert d.extract_number(t) == "7"
```

- [ ] **Step 2: Confirm they fail**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_domain_grading.py -k extract_number -q)`
Expected: failures (`.` captured; CoT mention captured).

- [ ] **Step 3: Replace `_number_re` with a digit-requiring constant.** In `base.py`, delete the `self._number_re = re.compile(...)` block (lines 26-29). Add a module-level constant near the top:

```python
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
```

- [ ] **Step 4: Rewrite `extract_number`** in `math/loader.py` (lines 150-152) to go through `extract_answer`:

```python
    def extract_number(self, text: str) -> str | None:
        from domains.base import _NUMBER_RE
        sol = self.extract_answer(text)
        if sol is None:
            return None
        m = _NUMBER_RE.search(sol)
        return m.group(0) if m else None
```

- [ ] **Step 5: Run tests + full suite, commit**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
```bash
git add -A && git commit -m "fix(rewards): extract numbers only from the anchored SOLUTION block, require a digit"
```

### Task 9: Stop rewarding empty answers

**Files:** `pipeline/domains/base.py`, `pipeline/domains/math/loader.py`, tests

Two layers: drop unparseable-truth rows at load time (primary), and guard `score_answer` against an empty extraction (defense).

- [ ] **Step 1: Add failing test**

```python
def test_empty_extraction_not_rewarded():
    # emitting nothing must never earn the max-reward branch
    assert d.score_answer("", "") != 5.0
    assert d.score_answer("", "42") < 0
```

- [ ] **Step 2: Confirm failure**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_domain_grading.py -k empty -q)`
Expected: fail (`"" == ""` returns `5.0`).

- [ ] **Step 3: Guard `score_answer`** — replace the `if extracted is None: return -2.0` head (base.py 68-69) with:

```python
        if extracted is None or extracted.strip() == "":
            return -2.0
```

- [ ] **Step 4: Drop empty-answer rows at load.** In `math/loader.py`, after each `.map(...)` in `_load_gsm8k`, `_load_math`, `_load_dapo`, chain a filter so broken-label rows never train:

```python
        ).filter(lambda x: x["answer"].strip() != "")
```
(Apply to all three loaders; the `.map(...)` already returns the Dataset the filter attaches to.)

- [ ] **Step 5: Run tests + full suite, commit**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
```bash
git add -A && git commit -m "fix(domains): drop empty-answer rows and never reward empty extractions"
```

### Task 10: Reject non-dict reward values in `validate_config`

**Files:** `pipeline/training/config_schema.py`, new tests in `pipeline/tests/test_validate_config.py`

`rewards.numeric: false` currently re-enables the reward (train.py:79 `... or {}`) then crashes after the model loads. Make it a clean validation error.

- [ ] **Step 1: Write failing tests**

```python
# pipeline/tests/test_validate_config.py
import pytest
from training.config_schema import validate_config

def _base():
    return {"experiment_id": "x", "model": {"slug": "qwen-7b"},
            "training": {"dataset": "openai/gsm8k"}, "rewards": {}}

def test_rejects_bool_reward_value():
    cfg = _base(); cfg["rewards"]["numeric"] = False
    with pytest.raises(ValueError, match="numeric"):
        validate_config(cfg)

def test_accepts_dict_reward_value():
    cfg = _base(); cfg["rewards"]["numeric"] = {"enabled": False}
    validate_config(cfg)  # must not raise

def test_compose_method_string_still_ok():
    cfg = _base(); cfg["rewards"]["compose_method"] = "naive_sum"
    validate_config(cfg)  # compose_method is a string, not a reward dict
```

- [ ] **Step 2: Confirm the first test fails** (no error raised today)

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_validate_config.py -q)`
Expected: `test_rejects_bool_reward_value` fails (no `ValueError`).

- [ ] **Step 3: Add the check** in `validate_config`, right after the `unknown_rewards` block (config_schema.py ~227):

```python
    for reward_name in _KNOWN_REWARD_SUBKEYS:        # excludes compose_method
        val = rewards.get(reward_name)
        if val is not None and not isinstance(val, dict):
            errors.append(
                f"rewards.{reward_name} must be a mapping (e.g. {{enabled: false}}), "
                f"got {type(val).__name__}: {val!r}"
            )
```

- [ ] **Step 4: Run tests + full suite, commit**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
```bash
git add -A && git commit -m "fix(config): reject non-dict reward values instead of silently re-enabling"
```

---

## Phase 5 — Test the central mechanism (CPU-local; torch is in .venv-test)

### Task 11: Test AdvantageWeightedComposer's per-group z-scoring

**Files:** `pipeline/tests/test_compose_metrics.py` (extend)

The review's top untested gap. torch is lazy in the composer and present in `.venv-test`, so this runs locally now. Update the file's header comment that claims the advantage-weighted path "is validated on the GPU box."

- [ ] **Step 1: Add the test** (the path-load already exposes `_compose`; add the class):

```python
AdvantageWeightedComposer = _compose.AdvantageWeightedComposer

def test_advantage_weighted_zscores_per_group():
    # one component, two prompt-groups of size 2.
    # group A raw [0,2] -> mean 1, Bessel std sqrt(2) -> z [-0.707, +0.707]
    # group B raw [5,5] -> zero variance -> contributes 0
    class Comp:
        def __call__(self, prompts, completions, **kw): return [0.0, 2.0, 5.0, 5.0]
    comp = AdvantageWeightedComposer([(Comp(), 1.0)])
    out = comp(["p", "p", "q", "q"], ["a", "b", "c", "d"])
    inv = 2 ** 0.5
    assert abs(out[0] + 1 / inv) < 1e-5
    assert abs(out[1] - 1 / inv) < 1e-5
    assert out[2] == 0.0 and out[3] == 0.0

def test_advantage_weighted_weight_applies_after_zscoring():
    class Comp:
        def __call__(self, prompts, completions, **kw): return [0.0, 2.0]
    comp = AdvantageWeightedComposer([(Comp(), 2.0)])
    out = comp(["p", "p"], ["a", "b"])
    inv = 2 ** 0.5
    assert abs(out[0] + 2 / inv) < 1e-5 and abs(out[1] - 2 / inv) < 1e-5
```

- [ ] **Step 2: Run + verify**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_compose_metrics.py -q)`
Expected: all passed (including the two new ones).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test(compose): cover advantage-weighted per-group z-scoring locally"
```

---

## Phase 3 — Eval comparability (helpers CPU-local; generation [GPU-box])

### Task 12: Fix the MMLU letter fallback (no bare-article "A")

**Files:** `pipeline/eval/ood_probes.py`, tests in `pipeline/tests/test_mmlu_extraction.py`

- [ ] **Step 1: Read** `ood_probes.py` and find the MMLU answer-extraction helper (the function that prefers a SOLUTION block then falls back to "the first bare A-D in the raw text"). Note its exact name and signature.

- [ ] **Step 2: Write failing tests** against that helper (substitute its real name):

```python
from eval.ood_probes import <mmlu_extract_fn>

def test_prefers_solution_block_letter():
    assert <mmlu_extract_fn>("...<SOLUTION>C</SOLUTION>") == "C"

def test_does_not_match_article_a():
    # "A" as the English article must not be read as the answer
    assert <mmlu_extract_fn>("A cyclist rides north. The answer is B.") in (None, "B")
```

- [ ] **Step 3: Fix the fallback** so a bare single letter only counts when it stands as an answer token (e.g. require it be preceded by an answer cue / be an isolated 1-char token at a line/answer position, or a `(A)`/`A)`/`A.` form), not any "A" in prose. Keep the SOLUTION-block path as the primary.

- [ ] **Step 4: Run the helper tests + full suite**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_mmlu_extraction.py tests/ -q)`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "fix(eval): MMLU letter fallback no longer matches the article 'A'"
```

### Task 13: Tighten the capability-floor matcher

**Files:** `pipeline/eval/ood_probes.py`, tests in `pipeline/tests/test_capability_floor.py` (extend)

- [ ] **Step 1: Read** `_capability_match` (and how the floor scans text when SOLUTION tags are absent).

- [ ] **Step 2: Add failing tests** — a sentence-final period must not fail a correct answer; absent SOLUTION tags must not let a CoT mention auto-pass:

```python
def test_capability_match_ignores_trailing_period():
    assert _capability_match("cold", "The answer is cold.") is True

def test_capability_match_requires_real_answer_not_cot_echo():
    # "9.9" appearing in the restated question must not auto-pass the 9.9-vs-9.11 trap
    assert _capability_match("9.11", "Compare 9.9 and 9.11. <SOLUTION>9.9</SOLUTION>") is False
```

- [ ] **Step 3: Fix** — normalize trailing punctuation in both sides before whole-word comparison; when SOLUTION tags are present, match only against the block, not the whole CoT.

- [ ] **Step 4: Run + commit**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_capability_floor.py tests/ -q)`
```bash
git add -A && git commit -m "fix(eval): capability-floor match tolerates trailing period, scans only the SOLUTION block"
```

### Task 14: One generation budget across probes

**Files:** `pipeline/eval/ood_probes.py`

- [ ] **Step 1: Read** the probes and find where MMLU/capability floor the budget to 512/256, overriding `eval.max_new_tokens`.

- [ ] **Step 2: Remove the per-probe floors** so every probe uses the configured `eval.max_new_tokens` (with one shared default). Mean token count is then comparable across splits.

- [ ] **Step 3: [GPU-box] verify** under smoke:
`cd pipeline && python -m eval.runner --config configs/e0-baseline-math-qwen-7b.yaml --smoke`
Expected: report lists the same `max_new_tokens` for every split.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "fix(eval): use one generation budget across all probes"
```

### Task 15: Lenient baseline extractor for `vs_base_model`

**Files:** `pipeline/eval/ood_probes.py` or `pipeline/eval/runner.py` (whichever scores the `--baseline` pass), `pipeline/eval/report.py`, tests

The base model never learned the tags, so the tag-anchored grader floors its accuracy and the delta credits capability for format learning. Score the baseline with a format-agnostic extractor.

- [ ] **Step 1: Read** the baseline scoring path (`--baseline` in `runner.py`) and `report.py`'s `vs_base_model` block.

- [ ] **Step 2: Add a format-agnostic extractor** the baseline uses: prefer the SOLUTION block if present, else the last `\boxed{}` or the last number in the text. Unit-test it:

```python
def test_lenient_extractor_reads_bare_final_number():
    assert lenient_extract("The cyclist travels 42 km total") == "42"
def test_lenient_extractor_prefers_solution_block():
    assert lenient_extract("...<SOLUTION>7</SOLUTION>") == "7"
```

- [ ] **Step 3: Wire it into the baseline pass only**; trained eval keeps the tag-anchored grader. In `report.py`, annotate the `vs_base_model` block: "baseline scored with a lenient (format-agnostic) extractor."

- [ ] **Step 4: Run helper tests + [GPU-box] baseline smoke**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
[GPU-box]: `python -m eval.runner --config configs/e0-baseline-math-qwen-7b.yaml --baseline --smoke`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "fix(eval): score the base-model baseline with a format-agnostic extractor"
```

---

## Phase 4 — Batch robustness

### Task 16: checkpoint-final as the sole clobber trigger

**Files:** `pipeline/training/train.py` (lines 127-133)

A mid-training crash leaves `config.yaml` without `checkpoint-final/`, and every retry then trips the guard. Make the directory the only trigger.

- [ ] **Step 1: Edit the guard** — drop the `existing_config` term:

```python
    existing_final = os.path.join(run_dir, "checkpoint-final")
    if os.path.isdir(existing_final) and not args.overwrite:
        raise FileExistsError(
            f"Run directory {run_dir!r} already has checkpoint-final/. "
            "Pass --overwrite to replace, or change experiment_id."
        )
```
Remove the now-unused `existing_config` line.

- [ ] **Step 2: [GPU-box] verify** a re-run after a simulated crash (no `checkpoint-final/`) proceeds instead of raising. Logic note in commit body.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "fix(train): only checkpoint-final/ blocks a re-run, not the frozen config"
```

### Task 17: Content-aware skip predicates in batch

**Files:** `pipeline/training/batch.py`, tests in `pipeline/tests/test_batch_skips.py`

The three skip predicates test only file existence, so a stub or smoke artifact reads as finished work.

- [ ] **Step 1: Read** `batch.py`; find the three skip checks (train: `checkpoint-final/`; eval: `eval_report.json`; baseline: `baseline/eval_report.json`) and the stub-report writer (`status: "skipped"|"error"`).

- [ ] **Step 2: Add a testable helper** and unit-test it:

```python
def _is_real_report(path: str) -> bool:
    """True only for a finished, non-stub, non-smoke eval report."""
    import json, os
    if not os.path.isfile(path):
        return False
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("status") not in ("skipped", "error") and not data.get("smoke", False)
```

```python
# test_batch_skips.py
def test_stub_report_is_not_real(tmp_path):
    p = tmp_path / "r.json"; p.write_text('{"status": "error"}')
    assert _is_real_report(str(p)) is False
def test_smoke_report_is_not_real(tmp_path):
    p = tmp_path / "r.json"; p.write_text('{"smoke": true, "results": {}}')
    assert _is_real_report(str(p)) is False
def test_finished_report_is_real(tmp_path):
    p = tmp_path / "r.json"; p.write_text('{"results": {"id_split": {}}}')
    assert _is_real_report(str(p)) is True
```

- [ ] **Step 3: Route the eval and baseline skip checks through `_is_real_report`.** Ensure eval reports written under `--smoke` carry `"smoke": true` (add it where the report is generated if absent). For the train skip, keep `checkpoint-final/` existence but treat a run whose only report is a stub as resumable.

- [ ] **Step 4: Run + commit**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/test_batch_skips.py tests/ -q)`
```bash
git add -A && git commit -m "fix(batch): skip predicates check report status, not just file existence"
```

### Task 18: Single canonical baseline path (drop symlinks)

**Files:** `pipeline/training/batch.py`, `pipeline/eval/report.py` (baseline discovery)

- [ ] **Step 1: Read** the baseline dedup block in `batch.py` (the `setdefault` ownership + relative symlink creation) and where `report.py` looks for `runs/<exp>/baseline/eval_report.json`.

- [ ] **Step 2: Replace symlink-sharing with one canonical path** keyed by `model.slug`, e.g. `runs/_baselines/<slug>/eval_report.json`. The baseline phase runs once per slug and writes there. `report.py`'s `vs_base_model` discovery looks up the canonical path for the run's slug. Remove all `os.symlink` / dangling-link handling.

- [ ] **Step 3: [GPU-box] verify** a two-config same-slug batch runs the baseline once and both reports pick up `vs_base_model`.

- [ ] **Step 4: Run logic suite + commit**

Run: `(cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q)`
```bash
git add -A && git commit -m "refactor(batch): one canonical baseline dir per slug, no symlinks"
```

---

## Phase 6 — Lock docs to code

### Task 19: Reconcile CLAUDE.md, README, and docstrings

**Files:** `CLAUDE.md`, `pipeline/README.md`, docstrings in `eval/metrics.py`

- [ ] **Step 1: Fix the metric definitions** everywhere they appear: underthinking is the **P10 / reference threshold**, not "<=50 tokens"; the accuracy CI is **Wilson**, not "95% bootstrap"; the capability floor has **6** default prompts (and a graded GSM8K-tail mode), not 5; MMLU is **zero-shot**, not 5-shot; z-scoring cancels "any global **positive** scalar."

- [ ] **Step 2: Propagate the deletions** — remove every mention of `effort_proxy`, the coding domain stub, the vLLM twin configs, and the linear length shape / `_RewardStepCallback` schedule. Document the `--vllm` flag and the five-config matrix. Update the reward-signal table and model-registry notes.

- [ ] **Step 3: Verify no stale terms survive**

Run: `grep -rn 'effort_proxy\|5-shot\|bootstrap CI\|<= *50 tokens\|linear-ablation\|-vllm' CLAUDE.md pipeline/README.md ; echo done`
Expected: no live references (matches only inside historical/spec docs are fine).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: align CLAUDE.md and README with the slimmed pipeline and real metrics"
```

---

## Self-review

**Spec coverage:** Phase 1 (effort/coding/vLLM/linear/dead-knobs) → Tasks 1-5. Phase 2 signal (comma, zero-truth, number-extraction, empty-answer, reward-disable) → Tasks 7-10. Phase 3 eval (floor block, MMLU, capability match, budget, lenient baseline) → Tasks 5,12-15. Phase 4 batch (clobber, stub skips, canonical baseline) → Tasks 16-18. Phase 5 tests (failing test, z-scoring) → Tasks 6,11. Phase 6 docs → Task 19. All spec sections map to a task.

**Placeholders:** Phase 3/4 tasks that touch large generation/orchestration files (`ood_probes.py`, `batch.py`, `runner.py`) open with a read step and name the exact symbol to find, because the precise edit depends on code not yet read in full; the test contracts and intent are concrete. This is a deliberate read-then-edit, not a "TBD."

**Type/name consistency:** `_build_token_length` returns `CosineLengthReward` (Tasks 2, 6). `_is_real_report(path) -> bool` defined and tested in Task 17. `_NUMBER_RE` defined in `base.py` (Task 8), imported in `loader.py`. Local verify command identical across tasks.

**Known sequencing:** Tasks 2 and 6 must land together (Task 2 deletes the class `test_reward_builders.py` imports). All other tasks are independent commits.
