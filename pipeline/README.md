# GRPO Training Pipeline

A GRPO training and evaluation pipeline for reasoning models. Experiments are YAML configs, so swapping reward signals does not need a code change.

## Usage

### Prerequisites

- Python 3.12, CUDA 13.0, NVIDIA GPU (tested on RTX 4090)
- Setup environment and install dependencies:

```bash
[./setup.sh](../setup.sh)
```

### Train

```bash
cd pipeline
python -m training.train --config configs/e0-baseline-math-qwen-7b.yaml
```

Add `--eval` to run evaluation automatically after training finishes.

Add `--smoke` to override the config for a fast sanity check (3 steps, 2 rollouts, 512 seq length, 64-sample dataset limit):

```bash
python -m training.train --config configs/e0-baseline-math-qwen-7b.yaml --smoke
```

### Evaluate a trained checkpoint

```bash
python -m eval.runner --config configs/e0-baseline-math-qwen-7b.yaml
```

Optionally override the checkpoint path:

```bash
python -m eval.runner --config configs/e0-baseline-math-qwen-7b.yaml --checkpoint runs/e0-baseline-math-qwen-7b/checkpoint-final
```

Add `--smoke` to limit eval to 10 samples per split for quick sanity checks:

```bash
python -m eval.runner --config configs/e0-baseline-math-qwen-7b.yaml --smoke
```

### Assess the base model (before-finetune baseline)

Run the same probe suite against the un-adapted base model from `config["model"]["slug"]`. Use this to measure what the finetune actually changed:

```bash
python -m eval.runner --config configs/e0-baseline-math-qwen-7b.yaml --baseline
```

Artefacts land under `runs/<experiment_id>/baseline/` so they do not collide with the trained-checkpoint report. The baseline pass is shared across all experiments using the same base model and probe set — you only need to run it once per (model, config) combo. `--baseline` and `--smoke` compose normally.

### Compare experiments

After running eval on multiple experiments, generate cross-experiment comparison plots:

```bash
cd pipeline
python -m eval.compare \
  --runs runs/e0-baseline-math-qwen-7b runs/e1-token-entropy-qwen-7b runs/e2-multi-signal-qwen-7b
```

Outputs land in `runs/comparison/`:

```
runs/comparison/
  compare_accuracy.png     # grouped bar chart: accuracy per split per experiment
  compare_efficiency.png   # accuracy vs mean token count Pareto plot — colour by reward family, marker by compose method, bold edge on Pareto-optimal points, error bars from Wilson/bootstrap CIs
  compare_summary.md       # markdown table with Δ-accuracy, token counts, underthinking + overthinking rates
  compare_pairwise.md      # paired-bootstrap Δ-accuracy matrix on the ID split with 95% CIs and two-sided p-values
```

Override the output directory with `--out`:

```bash
python -m eval.compare --runs runs/e0-baseline-math-qwen-7b runs/e1-token-length-qwen-7b --out runs/length-ablation
```

Add `--facet-by model` to split the Pareto plot into one subplot per base model (useful once experiments cover multiple models from the registry — the shipped configs are all `qwen-7b`, so this is a no-op until you add configs on another registry slug):

```bash
python -m eval.compare --runs runs/e0-baseline-math-qwen-7b \
  runs/e1-token-length-qwen-7b runs/e1-token-entropy-qwen-7b --facet-by model
```

### Batch run (overnight queue)

`training.batch` queues many configs through training, eval, and/or baseline assessment as subprocesses. Each phase gets a fresh Python process so GPU memory is released cleanly between experiments — designed to be left running unattended on a single GPU.

```bash
python -m training.batch configs/e0-*.yaml configs/e1-*.yaml --train --eval --baseline
```

Phase flags are independent and combinable. Default (no flag given) is `--train --eval`.

| Flag | Behaviour |
|------|-----------|
| `--train` | Run `training.train` for each config |
| `--eval` | Run `eval.runner` for each config (after train if both given; standalone otherwise) |
| `--baseline` | Run `eval.runner --baseline` per unique `model.slug` into the canonical `runs/_baselines/<slug>/` dir (other configs sharing that slug reuse it) |
| `--smoke` | Pass `--smoke` through to every subprocess |
| `--force` | Re-run phases even if their output artifacts already exist; passes `--overwrite` to train |
| `--retries N` | Retry a failed phase N times before marking it failed (default `1`) |
| `--no-compare` | Skip the auto `eval.compare` run at the end of the batch |
| `--compare-out DIR` | Output directory for `eval.compare` artifacts (default `runs/comparison`) |
| `--summary-dir DIR` | Where to write `batch_summary_<timestamp>.md` (default `runs`) |

**Execution order** with all three phases enabled:

1. **Baselines first** (deduplicated by `model.slug`). The base-model assessment for a slug lives at one canonical path, `runs/_baselines/<slug>/`. The first config touching a given slug writes it there; later same-slug configs find the canonical report and skip. No symlinks, no per-experiment baseline dir. Baseline-first ordering means each trained eval report can pick up the `vs_base_model` delta block automatically.
2. **Train + eval per config**, looped. Eval runs immediately after each train so a `tail -f` from another shell sees full reports land one experiment at a time.
3. **Auto `eval.compare`** across every run that produced an `eval_report.json` this batch (skipped silently if fewer than two reports exist, or when `--no-compare` is passed).

**Resume behaviour:** by default each phase is skipped if its output artifacts exist — `checkpoint-final/` for train, `eval_report.json` for eval, the canonical `runs/_baselines/<slug>/eval_report.json` for baseline. Re-running the same batch command after a crash picks up where it left off. Pass `--force` to redo everything.

**End-of-batch outputs:**

```
runs/
  <exp_id>/
    batch_train.log         # captured stdout/stderr for the train phase
    batch_eval.log
    batch_baseline.log
    checkpoint-final/
    eval_report.json
    eval_report.md
  _baselines/
    <slug>/                 # one canonical base-model assessment per model slug
      eval_report.json
      eval_report.md
  comparison/
    compare_accuracy.png
    compare_efficiency.png
    compare_summary.md
    compare_pairwise.md
    batch_compare.log
  batch_summary_<timestamp>.md
```

The exit code is non-zero if any phase ultimately failed (after retries), so a wrapping shell script or cron job can detect it.

**Examples:**

```bash
# Train + eval the full e1 family across all model variants, retry twice on failure
python -m training.batch configs/e1-*.yaml --train --eval --retries 2

# Re-eval already-trained checkpoints (e.g. after changing eval.* in the config)
python -m training.batch configs/e2-*.yaml --eval --force

# Only baseline-assess every base model used by the e1 configs
python -m training.batch configs/e1-*.yaml --baseline

# Smoke the entire matrix end-to-end (3 train steps, 10 eval samples per split)
python -m training.batch configs/e*-*.yaml --train --eval --baseline --smoke
```

### Run a new experiment

1. Copy the template config:

```bash
cp configs/_template.yaml configs/e4-my-experiment.yaml
```

2. Edit `experiment_id`, toggle reward signals under `rewards:`, and adjust weights.
3. Run:

```bash
python -m training.train --config configs/e4-my-experiment.yaml --eval
```

4. Results land in `runs/<experiment_id>/`:

```
runs/e4-my-experiment/
  config.yaml              # frozen copy of the config used
  checkpoint-final/        # LoRA adapter + tokenizer
  eval_report.json         # structured metrics (trained model)
  eval_report.md           # human-readable summary
  training_curves.png      # reward, KL, completion length, loss over steps
  eval_accuracy.png        # accuracy with Wilson 95% CI across all eval splits
  token_distribution.png   # token count histogram: correct vs incorrect
  difficulty_scatter.png   # difficulty vs token count (MATH datasets only)
  baseline/                # populated by `eval.runner --baseline` (before-finetune assessment)
    eval_report.json       # same schema as the trained report
    eval_report.md
    eval_accuracy.png
    token_distribution.png
    difficulty_scatter.png
```

### Experiment matrix

Five configs ship, all targeting `qwen-7b`. Earlier `qwen-1.5b`/`qwen3-4b`/bare-name variants were retired to keep the matrix focused on the 7B target; the model registry still lists those slugs, so add configs for them if you want to sweep model size. vLLM is a runtime flag (`--vllm` on `training.train` / `training.batch`), not a separate config.

`e1-token-length` uses the correctness-coupled cosine reward (`CosineLengthReward`, Wu/Yeo 2025), which survives advantage-weighted z-scoring with real structure intact.

| Config | Reward signals | Compose method | Purpose |
|--------|---------------|----------------|---------|
| `e0-baseline-math-qwen-7b` | accuracy only | advantage_weighted | Baseline for delta comparisons |
| `e1-token-length-qwen-7b` | accuracy + token_length (cosine) | advantage_weighted | Correctness-coupled length reward |
| `e1-token-entropy-qwen-7b` | accuracy + token_entropy | advantage_weighted | Entropy reward, no fork masking |
| `e2-multi-signal-qwen-7b` | accuracy + token_length + token_entropy | advantage_weighted | Combined efficiency signals |
| `e3-ablation-naive-sum-qwen-7b` | same as e2 | naive_sum | Ablation: tests advantage weighting |

## Architecture

```
Config YAML
    |
    v
train.py:main()
    |
    +-- validate_config()            # schema check before anything runs
    +-- build_domain()               # MathDomain
    +-- GRPORunner(config)           # load model + tokenizer + LoRA
    +-- domain.build_chat_template() # inject reasoning tags into tokenizer
    +-- domain.load_dataset()        # HuggingFace dataset -> unified format
    +-- domain.filter_by_prompt_length()  # drop overly long prompts
    +-- build_reward_components()    # (reward_fn, weight) pairs from config
    +-- build_composer()             # AdvantageWeighted | NaiveSum
    +-- runner.train(dataset, reward_fn, callbacks)
    |       |
    |       v
    |   TRL GRPOTrainer (per step):
    |       1. Sample n_rollouts completions per prompt
    |       2. reward_fn(prompts, completions, **kwargs) -> list[float]
    |       3. Compute advantages + GRPO loss
    |       4. Backprop through LoRA
    |       5. Advance any reward schedulers via callback
    |
    +-- runner.save_lora()
    |
    v  (optional --eval)
eval/runner.py:run_eval()
    +-- run_ood_probes()             # ID + near-OOD + far-OOD + capability floor
    +-- generate_report()            # JSON + Markdown
    +-- plots.plot_all()             # PNGs: training_curves, accuracy_bars, token_dist, difficulty
```

### Design choices

#### Why YAML configs

Each experiment is a YAML file. `train.py` reads the config and wires up the domain, reward components, and training loop. There is no per-experiment branching in code; E0 vs E3 is a config diff.

#### The Domain class

`Domain` separates dataset loading, answer extraction, and scoring from the rest of the pipeline. To add theorem proving or a new math source, subclass `Domain` and implement the abstract methods. Reward functions do not change, because they call back into the domain for scoring.

#### Reward signals as callables

Every reward signal has signature `(prompts, completions, **kwargs) -> list[float]`. The composer combines them. Adding a new signal means one new class plus an entry in `REWARD_REGISTRY` and `_KNOWN_REWARD_KEYS`.

#### Advantage-weighted composition

Each reward component is z-scored per batch before the weighted sum (DIET §3.2). The motivation: binary accuracy has variance roughly C(1-C), which is much larger than the variance of a small length penalty, so a naive sum lets accuracy dominate no matter what weight you set. The `naive_sum` composer is kept around as the E3 ablation against E2 — that is how you measure whether the z-scoring actually mattered for your reward mix.

## Modules

### `configs/`

YAML experiment configurations. `_template.yaml` documents every available field with defaults. Copy it to create new experiments. Key sections:

- `model.slug` - references a model in `training/registry.py`
- `training` - dataset, hyperparameters, GRPO rollout count
- `rewards` - enable/disable and weight each reward signal, choose compose method
- `eval` - decoding strategy, OOD probe selection, baseline comparison

### `domains/`

Abstract base class and concrete implementations for problem domains.

**`base.py` - `Domain`**

Defines the interface every domain must implement:

- Tag constants: `reasoning_start`, `reasoning_end`, `solution_start`, `solution_end` - injected into the chat template and used by format rewards
- `_solution_re` / `_number_re` - regex patterns built from the tags for answer extraction
- `extract_answer(text)` - pulls the solution string from between `<SOLUTION>` tags
- `score_answer(extracted, truth)` - graded correctness: +5.0 exact match, +3.5 stripped match, +2.0 within 10%, +1.5 within 20%, negative penalties otherwise
- `score_numbers(extracted, truth)` - strict float equality: +3.5 if equal, -1.5 otherwise
- `is_correct(completion, truth)` - binary check for evaluation (exact or numeric)
- `build_chat_template(tokenizer)` - injects a Jinja2 template with reasoning tags into the tokenizer

**`math/loader.py` - `MathDomain`**

Supports GSM8K, Competition Math, and DAPO datasets. Extracts ground truth from `####` delimiters (GSM8K) or `\boxed{}` (MATH). Provides `filter_by_prompt_length()` to drop the top 10% longest prompts. Includes `extract_number()` for numeric reward.

### `training/`

**`train.py` - Main entry point**

Orchestrates the full training pipeline: load config, validate schema, set seeds, build domain, load model, build reward components, compose reward function, train, save.

**`grpo_runner.py` - `GRPORunner`**

Wraps Unsloth `FastLanguageModel` and TRL `GRPOTrainer`. Handles model loading (4-bit or 16-bit), LoRA application (rank, alpha, target modules), and GRPOConfig construction. Accepts optional `callbacks` parameter for trainer callbacks.

**`registry.py` - Model registry**

Maps slugs (e.g. `qwen-1.5b`) to model configs (HuggingFace name, quantization, max sequence length, max LoRA rank). `get_model_config(slug)` raises `KeyError` with available options if the slug is unknown.

**`config_schema.py` - Config validation**

Checks required keys (`experiment_id`, `model.slug`, `training.dataset`), numeric ranges (LoRA rank, learning rate, KL beta), model slug validity against the registry, and compose method validity. Runs before any model loading to fail fast on misconfiguration.

**`batch.py` - Multi-experiment runner**

Queues N configs through train, eval, and/or baseline phases as subprocesses (one fresh Python process per phase). Reads `experiment_id` and `model.slug` from each YAML cheaply up front so a bad config fails the batch immediately, before any GPU work starts. Tees each subprocess's combined stdout/stderr to both the parent terminal and a per-phase log file under `runs/<exp_id>/`. Baseline phase deduplicates by `model.slug` via one canonical `runs/_baselines/<slug>/` dir: the first config touching a slug writes it there, later same-slug configs find it and skip. Auto-invokes `eval.compare` at end of batch when at least two eval reports were produced.

### `training/rewards/`

Each reward is a callable class with signature `(prompts, completions, **kwargs) -> list[float]`. All use `extract_content(completion)` from `utils.py` for safe access to completion text.

`__init__.py` exposes `REWARD_REGISTRY: key → (default_enabled, default_weight, builder)`. `train.build_reward_components` iterates the registry instead of hard-coding branches; `config_schema._KNOWN_REWARD_KEYS` rejects unknown keys (catches typos like `enaabled: true`).

**`format.py`**

- `FormatExactReward` - +3.0 if the full `<end_working_out>...<SOLUTION>...</SOLUTION>` structure is present (regex match). 0.0 otherwise.
- `FormatApproxReward` - Per-tag partial credit: +0.5 if the tag occurs exactly once, -1.0 for zero or multiple occurrences. `reasoning_end` is counted on the full text; `solution_start` and `solution_end` are counted on the suffix after `reasoning_end` so the model can't earn credit by quoting tag names inside its CoT.

**`accuracy.py`**

- `AnswerReward` - Calls `domain.score_answer()` on the extracted answer. Graded from -4.5 to +5.0.
- `NumericReward` - Calls `domain.score_numbers()` on the extracted number. Strict float equality: +3.5 or -1.5. Returns -2.5 if the domain has no `extract_number` method (MathDomain always provides one).

**`cosine_length.py` - `CosineLengthReward`**

Correctness-coupled cosine length reward (Wu/Yeo 2025). For a completion of `n_tokens`, interpolates between a short-end and long-end value based on correctness: correct completions are rewarded more when shorter; wrong completions are penalized less when longer (wrong-and-short is the most-penalized cell). Non-linear in length and gated by correctness, so it survives advantage-weighted z-scoring with real structure intact.

**`token_entropy.py` - `TokenEntropyReward`**

Mean per-token Shannon entropy from model logits. Batched forward pass over all completions. Rewards high-uncertainty tokens (genuine deliberation). Optional `fork_mask_top_pct` focuses reward on the top-X% highest-entropy tokens only, targeting actual decision points.

**`compose.py`**

- `AdvantageWeightedComposer` - normalizes each component to zero-mean unit-variance per batch before weighted sum. Ensures weights faithfully control contribution regardless of raw reward scale.
- `NaiveSumComposer` - plain weighted sum. Ablation baseline for testing the advantage-weighting hypothesis.
- `build_composer(components, method)` - factory function.

**`utils.py`**

- `extract_content(completion)` - safely extracts `completion[0]["content"]` with fallback to empty string on malformed completions.

### `eval/`

**`runner.py` - `run_eval()`**

Loads a trained LoRA checkpoint, runs all OOD probes, and writes the evaluation report. Can also be invoked standalone via `python -m eval.runner`.

**`metrics.py`**

- `SampleResult` - per-sample dataclass: correct, n_tokens, difficulty
- `EvalMetrics` - aggregate metrics: accuracy with Wilson 95% interval, mean token count with bootstrap CI, underthinking rate, overthinking rate (+ absolute token threshold), Pearson r (difficulty vs length) with p-value
- `compute_metrics()` - computes all metrics from a list of SampleResults. Accuracy CI uses a Wilson score interval. Mean token count and thinking-rate CIs use bootstrap resampling (n=10,000). Underthinking is correct answers at or below the P10 token count of the split (or a fixed threshold pinned from a reference run); overthinking is above P75 similarly.

**`ood_probes.py`**

Runs four evaluation splits:
- **ID split** - held-out portion of the training dataset (200 samples default, configurable via `eval.id_split_limit`)
- **Near-OOD** - same domain, harder distribution (200 samples default, configurable via `eval.near_ood_limit`)
- **Far-OOD** - MMLU, zero-shot multiple-choice (100 samples default, configurable via `eval.far_ood_limit`)
- **Capability floor** - 6 default instruction-following prompts, or a graded GSM8K-tail slice (the mode the shipped configs use). Sanity check for catastrophic forgetting.

Use `--smoke` to cap all splits to 10 samples for quick sanity checks. `--smoke` on `training.train --eval` propagates the cap into eval automatically.

Decoding is configurable per config (`eval.temperature`, `eval.do_sample`). Defaults to greedy (`do_sample=false`; `temperature` is dropped from generation kwargs to silence HF warnings).

Generation is batched. Set `eval.batch_size` (default 8) per config. Tokenizer is set to left-padding during generation, otherwise the completion slice indices land in the wrong place for shorter prompts in a batch. MMLU and capability-floor probes go through the same chat template as ID/near-OOD so the trained model can still emit `<start_working_out>...<SOLUTION>...</SOLUTION>`; answer extraction prefers the SOLUTION block before falling back to raw text scanning.

**`report.py`**

Generates structured JSON and Markdown reports. Two independent baseline comparisons are populated when their data is available:

- `vs_reward_baseline` — trained-vs-trained comparison against an E0 reward-only baseline experiment (e.g., `e0-baseline-math-qwen-7b`). Discovery: explicit `baseline_id` from config first, then heuristic matching for entries starting with `e0-` or containing `-baseline-`. Useful for reward-stack ablation deltas.
- `vs_base_model` — before-vs-after comparison against the same model and probes evaluated *without* the LoRA adapter. Populated whenever `runs/<experiment_id>/baseline/eval_report.json` exists (produced by `eval.runner --baseline`). This is the actual "did finetuning help?" measurement and reports both Δ accuracy and Δ mean tokens.

Reports include accuracy (Wilson 95% CI), mean token count (bootstrap CI), underthinking rate, and overthinking rate; the Pearson r between difficulty and length (with p-value); plus both baseline-delta blocks above when applicable.

**`plots.py`**

Generates per-experiment PNG figures automatically at the end of each eval run:

- `plot_training_curves` — 2×2 panel: reward (±1 std band), KL divergence, completion length, policy loss. Reads `trainer_state.json` from the most recent checkpoint.
- `plot_accuracy_bars` — horizontal bar chart with Wilson 95% CI error bars, one bar per eval split.
- `plot_token_distribution` — overlapping histograms of token counts for correct vs incorrect completions (ID split).
- `plot_difficulty_scatter` — scatter of difficulty vs token count, coloured by correctness, annotated with Pearson r (MATH datasets only; skipped silently otherwise).
- `plot_all` — top-level entry point that calls all four; individual failures are caught and logged without aborting eval.

Requires `matplotlib`. Logs a warning and skips plotting if it is not installed.

**`compare.py`**

Standalone CLI for cross-experiment comparison. Reads only `eval_report.json` files — no model loading required.

```bash
python -m eval.compare --runs runs/e0-baseline runs/e1-token-entropy [--out runs/comparison]
```

Outputs:
- `compare_accuracy.png` — grouped bar chart per experiment and split, with baseline dashed line.
- `compare_efficiency.png` — Pareto plot: accuracy vs mean token count. Reward family encoded as colour, compose method as marker, Pareto-optimal points (non-dominated on accuracy↑ × tokens↓) drawn with a bold black edge. Error bars come from Wilson CIs on accuracy and bootstrap CIs on mean token count. Use `--facet-by model` to render one subplot per base model.
- `compare_summary.md` — markdown table: experiment, accuracy, Δ vs baseline, mean tokens, underthinking rate, overthinking rate.
- `compare_pairwise.md` — paired-bootstrap Δ-accuracy matrix on the ID split. For every ordered pair of experiments with matching `n_samples`, resamples per-sample correctness indices with replacement (n=2000, seeded) and reports Δ-accuracy, its 95% CI, and a two-sided p-value. Pairs without per-sample series (older reports) or with mismatched lengths are skipped explicitly rather than silently fabricated.

## Reward signal reference

| Signal | Class | Range | When enabled | Config key |
|--------|-------|-------|-------------|------------|
| Format exact | `FormatExactReward` | [0.0, 3.0] | Default | `rewards.format_exact` |
| Format approx | `FormatApproxReward` | [-3.0, 1.5] | Default | `rewards.format_approx` |
| Answer accuracy | `AnswerReward` | [-4.5, 5.0] | Default | `rewards.accuracy` |
| Numeric equality | `NumericReward` | [-2.5, 3.5] | Default | `rewards.numeric` |
| Token length (cosine) | `CosineLengthReward` | [-1.0, 1.0] | Opt-in | `rewards.token_length` |
| Token entropy | `TokenEntropyReward` | [0.0, +inf) | Opt-in | `rewards.token_entropy` |

## Model registry

| Slug | Model | Quantization | Max seq | Max LoRA rank |
|------|-------|-------------|---------|---------------|
| `qwen3-4b` | unsloth/Qwen3-4B-Base | 16-bit | 2048 | 32 |
| `qwen-1.5b` | Qwen/Qwen2.5-1.5B | 4-bit | 2048 | 32 |
| `qwen-7b` | Qwen/Qwen2-7B | 4-bit | 2048 | 64 |

To add a new model, add an entry to `MODEL_REGISTRY` in `training/registry.py`.

## Output format

The model is trained to emit completions in this structure:

```
<start_working_out>
[reasoning chain of thought]
<end_working_out>
<SOLUTION>
[answer]
</SOLUTION>
```

The `<start_working_out>` tag is prepended automatically by the chat template (via `add_generation_prompt=True`). The reward functions check for the remaining tags.
