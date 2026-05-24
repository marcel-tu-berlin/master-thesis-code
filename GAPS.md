# Pipeline Gaps

Open gaps in the GRPO experiment pipeline. Each entry: what is missing, why it matters for the thesis goals (reward signals that reduce overthinking while holding or improving accuracy), and what would close it.

## 1. Coding domain is a stub

`pipeline/domains/coding/` loads HumanEval and MBPP, but `CodingDomain.is_correct` returns `False` and `score_answer` returns `0.0`. Training a coding experiment today rewards every completion at zero, so GRPO has no signal to learn from.

Why it matters: cross-domain claims ("reward X transfers from math to code") cannot be made until the coding reward works.

Fix: sandboxed execution (e.g. `exec` in subprocess with timeout and memory limit, or a container) that runs candidate code against unit tests and returns a pass/fail signal. Treat as untrusted code; isolate.

## 2. No multi-seed sweep

One config produces one training run with one random seed. RL training is noisy. A single-seed gap of 0.04 accuracy between two reward signals may be real or may be variance.

Why it matters: claims like "reward family A beats reward family B" need n ≥ 3 seeds, ideally 5, with mean and standard deviation reported. Reviewers will ask.

Fix: add a `seeds: [0, 1, 2]` field to configs, have `training.batch` fan out one subprocess per seed, then aggregate in `eval.compare` (mean ± std bars, paired statistical test).

## 3. Far-OOD is hardcoded to MMLU

`pipeline/eval/ood_probes.py` only recognises an MMLU-shaped probe; other values warn and skip. Fine for math reasoning, but limits the generalisation story.

Why it matters: a single far-OOD benchmark is one data point. Different far-OOD sets (BBH, AGIEval, GPQA) probe different capabilities. A reward signal that wins on MMLU but loses on BBH is worth knowing.

Fix: a small registry of far-OOD probes keyed by name, with loaders and a multiple-choice scoring helper shared across them.

## 4. Efficiency metric is tokens only

The pipeline measures mean token count, underthinking rate (correct under 50 tokens), and overthinking rate (correct above split-level P75). It does not measure wall-clock time, FLOPs, or per-step latency.

Why it matters: "efficient" can mean fewer tokens, less compute, or lower latency. Token count is a reasonable proxy for thinking budget but not for compute cost — a reward that produces shorter but harder-to-decode sequences could be slower in practice.

Fix: log generation wall-clock per sample during eval, and optionally estimate FLOPs from token count and model size. Add to `EvalMetrics` and the Pareto plot as an optional axis.

## 5. Underthinking threshold is fixed at 50 tokens

`compute_metrics` in `pipeline/eval/metrics.py` uses a constant `underthinking_threshold=50`. Overthinking uses P75 of the split, which adapts to data. Underthinking does not.

Why it matters: 50 tokens may be too low for harder MATH problems (where even a terse correct chain runs longer) or too high for short GSM8K answers. A fixed threshold can hide or invent the underthinking signal depending on dataset.

Fix: make the threshold configurable per probe in YAML, or derive it from a percentile of the split (e.g. P10) the way overthinking uses P75.

## 6. No paired statistical test in cross-experiment comparison

`eval.compare` plots deltas and shows bootstrap CIs per experiment, but does not test whether the difference between two experiments is significant. Two error-barred dots that visually overlap may still be reliably different on a paired test, and vice versa.

Why it matters: the thesis will argue "this reward signal is better." A formal paired test (bootstrap difference, permutation, or paired t on per-sample correctness) makes the claim defensible.

Fix: add a pairwise comparison routine to `eval.compare` — bootstrap the difference in accuracy on the matched ID-split sample set, report 95% CI on the delta and a p-value. Output a small markdown table alongside the existing plots.

## Out of scope (intentional)

- Per-step KL tracking and reward decomposition plots (already in `training_curves.png`).
- Distillation or self-consistency at eval time. Eval is single-sample by design to keep the efficiency signal honest.
