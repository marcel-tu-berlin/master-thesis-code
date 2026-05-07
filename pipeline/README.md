# GRPO Training Pipeline

Reinforcement learning pipeline for training reasoning models with Group Relative Policy Optimization (GRPO). Built for systematic experimentation with different reward signals.

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
python -m training.train --config configs/e0-baseline-math-1.5b.yaml
```

Add `--eval` to run evaluation automatically after training finishes.

Add `--smoke` to override the config for a fast sanity check (3 steps, 2 rollouts, 512 seq length, 64-sample dataset limit):

```bash
python -m training.train --config configs/e0-baseline-math-1.5b.yaml --smoke
```

### Evaluate a trained checkpoint

```bash
python -m eval.runner --config configs/e0-baseline-math-1.5b.yaml
```

Optionally override the checkpoint path:

```bash
python -m eval.runner --config configs/e0-baseline-math-1.5b.yaml --checkpoint runs/e0-baseline-math-1.5b/checkpoint-final
```

Add `--smoke` to limit eval to 10 samples per split for quick sanity checks:

```bash
python -m eval.runner --config configs/e0-baseline-math-1.5b.yaml --smoke
```

### Compare experiments

After running eval on multiple experiments, generate cross-experiment comparison plots:

```bash
cd pipeline
python -m eval.compare \
  --runs runs/e0-baseline-math-1.5b runs/e1-token-entropy runs/e2-multi-signal
```

Outputs land in `runs/comparison/`:

```
runs/comparison/
  compare_accuracy.png     # grouped bar chart: accuracy per split per experiment
  compare_efficiency.png   # scatter: accuracy vs mean token count (efficiency frontier)
  compare_summary.md       # markdown table with Δ-accuracy and token counts
```

Override the output directory with `--out`:

```bash
python -m eval.compare --runs runs/e0-baseline runs/e1-token-length --out runs/length-ablation
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
  eval_report.json         # structured metrics
  eval_report.md           # human-readable summary
  training_curves.png      # reward, KL, completion length, loss over steps
  eval_accuracy.png        # accuracy with 95% CI across all eval splits
  token_distribution.png   # token count histogram: correct vs incorrect
  difficulty_scatter.png   # difficulty vs token count (MATH datasets only)
```

### Experiment matrix

| Config | Reward signals | Compose method | Purpose |
|--------|---------------|----------------|---------|
| `e0-baseline-math-1.5b` | accuracy only | advantage_weighted | Baseline for delta comparisons |
| `e1-token-length` | accuracy + token_length (cosine) | advantage_weighted | DIET length penalty |
| `e1-token-entropy` | accuracy + token_entropy | advantage_weighted | Entropy reward, no fork masking |
| `e1-token-entropy-forkmask` | accuracy + token_entropy (top-20%) | advantage_weighted | Entropy reward with fork masking |
| `e2-multi-signal` | accuracy + token_length + token_entropy | advantage_weighted | Combined efficiency signals |
| `e3-ablation-naive-sum` | same as e2 | naive_sum | Ablation: tests advantage weighting |

## Architecture

```
Config YAML
    |
    v
train.py:main()
    |
    +-- validate_config()            # schema check before anything runs
    +-- build_domain()               # MathDomain | CodingDomain
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
    |       5. _RewardStepCallback advances reward schedulers
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

**Config-driven, not code-driven.** Every experiment is a YAML file. The training script is a thin orchestrator that reads the config and wires together the domain, reward components, and training loop. This eliminates code divergence between experiment runs and makes the experiment matrix reproducible.

**Domain abstraction.** The `Domain` base class decouples dataset loading, answer extraction, and scoring from the reward functions and training loop. Adding a new domain (e.g. theorem proving, multilingual math) requires subclassing `Domain` and implementing three abstract methods. The reward functions stay unchanged.

**Composable reward signals.** Each reward signal is an independent callable `(prompts, completions, **kwargs) -> list[float]`. The composer combines them. This means you can add a new signal by writing a single class and adding an entry to the config, without touching any other code.

**Advantage-weighted composition.** By default, each reward component is normalized to zero-mean unit-variance per batch before being weighted and summed (following DIET section 3.2). This prevents high-variance signals like binary accuracy from dominating regardless of their assigned weight. The `naive_sum` composer is kept as an ablation baseline.

**Callback-driven schedulers.** Reward signals with time-dependent behavior (e.g. cosine-annealed length penalty) expose a `step()` method. A `TrainerCallback` calls all step-able rewards after each training step, so the training loop doesn't need to know about individual reward internals.

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

**`coding/loader.py` - `CodingDomain`**

Supports HumanEval and MBPP. Loads `test_code` and `entry_point` fields for future execution-based verification. `is_correct()` and `score_answer()` currently raise `NotImplementedError` since string/float comparison is not valid for code correctness.

### `training/`

**`train.py` - Main entry point**

Orchestrates the full training pipeline: load config, validate schema, set seeds, build domain, load model, build reward components, compose reward function, train, save. The `_RewardStepCallback` class advances any reward schedulers that expose a `step()` method after each training step.

**`grpo_runner.py` - `GRPORunner`**

Wraps Unsloth `FastLanguageModel` and TRL `GRPOTrainer`. Handles model loading (4-bit or 16-bit), LoRA application (rank, alpha, target modules), and GRPOConfig construction. Accepts optional `callbacks` parameter for trainer callbacks.

**`registry.py` - Model registry**

Maps slugs (e.g. `qwen-1.5b`) to model configs (HuggingFace name, quantization, max sequence length, max LoRA rank). `get_model_config(slug)` raises `KeyError` with available options if the slug is unknown.

**`config_schema.py` - Config validation**

Checks required keys (`experiment_id`, `model.slug`, `training.dataset`), numeric ranges (LoRA rank, learning rate, KL beta), model slug validity against the registry, and compose method validity. Runs before any model loading to fail fast on misconfiguration.

### `training/rewards/`

Each reward is a callable class with signature `(prompts, completions, **kwargs) -> list[float]`. All use `extract_content(completion)` from `utils.py` for safe access to completion text.

**`format.py`**

- `FormatExactReward` - +3.0 if the full `<end_working_out>...<SOLUTION>...</SOLUTION>` structure is present (regex match). 0.0 otherwise.
- `FormatApproxReward` - Per-tag partial credit: +0.5 for exactly one occurrence of each structural tag, -1.0 for zero or multiple. Only counts tags after `reasoning_end` to avoid false positives from the model discussing tags in its reasoning.

**`accuracy.py`**

- `AnswerReward` - Calls `domain.score_answer()` on the extracted answer. Graded from -4.5 to +5.0.
- `NumericReward` - Calls `domain.score_numbers()` on the extracted number. Strict float equality: +3.5 or -1.5. Falls back gracefully if the domain lacks `extract_number`.

**`token_length.py` - `TokenLengthReward`**

Negative length penalty: `-alpha * num_tokens`. Two modes:
- `constant` - fixed alpha throughout training
- `cosine` - alpha annealed from 0 to `alpha` over training (DIET schedule). The model learns to reason first, then compresses. Requires `step()` to be called each training step (handled by `_RewardStepCallback`).

**`token_entropy.py` - `TokenEntropyReward`**

Mean per-token Shannon entropy from model logits. Batched forward pass over all completions. Rewards high-uncertainty tokens (genuine deliberation). Optional `fork_mask_top_pct` focuses reward on the top-X% highest-entropy tokens only, targeting actual decision points.

**`effort_proxy.py` - `EffortProxyReward`**

Penalizes compute effort per rollout. Three metrics:
- `token_count` - raw token count (most reliable)
- `flops` - estimated FLOPs per token (2 x D^2 x L), normalized to GFLOPs (/ 1e9) so magnitude is comparable to token_count but reflects architecture differences
- `gpu_time` - falls back to token_count (wall-clock timing unavailable during reward computation)

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
- `EvalMetrics` - aggregate metrics: accuracy, 95% bootstrap CI, mean token count, underthinking rate, Pearson r (difficulty vs length) with p-value
- `compute_metrics()` - computes all metrics from a list of SampleResults. Uses `scipy.stats.pearsonr` for correlation with statistical significance, and bootstrap resampling (n=2000) for accuracy confidence intervals.

**`ood_probes.py`**

Runs four evaluation splits:
- **ID split** - held-out portion of the training dataset (200 samples default, configurable via `eval.id_split_limit`)
- **Near-OOD** - same domain, harder distribution (200 samples default, configurable via `eval.near_ood_limit`)
- **Far-OOD** - MMLU (100 samples default, configurable via `eval.far_ood_limit`)
- **Capability floor** - 5 fixed instruction-following questions (sanity check for catastrophic forgetting)

Use `--smoke` to cap all splits to 10 samples for quick sanity checks.

Decoding is configurable per config (`eval.temperature`, `eval.do_sample`). Defaults to greedy (temperature=0).

**`report.py`**

Generates structured JSON and Markdown reports. Compares against a baseline experiment if found. Baseline discovery uses explicit `baseline_id` from config first, then falls back to heuristic matching (entries starting with `e0-` or containing `-baseline-`). Reports include accuracy with 95% CI, Pearson r with p-value, and delta-accuracy vs baseline.

**`plots.py`**

Generates per-experiment PNG figures automatically at the end of each eval run:

- `plot_training_curves` — 2×2 panel: reward (±1 std band), KL divergence, completion length, policy loss. Reads `trainer_state.json` from the most recent checkpoint.
- `plot_accuracy_bars` — horizontal bar chart with 95% CI error bars, one bar per eval split.
- `plot_token_distribution` — overlapping histograms of token counts for correct vs incorrect completions (ID split).
- `plot_difficulty_scatter` — scatter of difficulty vs token count, coloured by correctness, annotated with Pearson r (MATH datasets only; skipped silently otherwise).
- `plot_all` — top-level entry point that calls all four; individual failures are caught and logged without aborting eval.

Requires `matplotlib`. Degrades gracefully (no crash) if not installed.

**`compare.py`**

Standalone CLI for cross-experiment comparison. Reads only `eval_report.json` files — no model loading required.

```bash
python -m eval.compare --runs runs/e0-baseline runs/e1-token-entropy [--out runs/comparison]
```

Outputs:
- `compare_accuracy.png` — grouped bar chart per experiment and split, with baseline dashed line.
- `compare_efficiency.png` — scatter of accuracy vs mean token count (efficiency frontier).
- `compare_summary.md` — markdown table: experiment, accuracy, Δ vs baseline, mean tokens, underthinking rate.

## Reward signal reference

| Signal | Class | Range | When enabled | Config key |
|--------|-------|-------|-------------|------------|
| Format exact | `FormatExactReward` | [0.0, 3.0] | Default | `rewards.format_exact` |
| Format approx | `FormatApproxReward` | [-3.0, 1.5] | Default | `rewards.format_approx` |
| Answer accuracy | `AnswerReward` | [-4.5, 5.0] | Default | `rewards.accuracy` |
| Numeric equality | `NumericReward` | [-2.5, 3.5] | Default | `rewards.numeric` |
| Token length | `TokenLengthReward` | (-inf, 0.0] | Opt-in | `rewards.token_length` |
| Token entropy | `TokenEntropyReward` | [0.0, +inf) | Opt-in | `rewards.token_entropy` |
| Effort proxy | `EffortProxyReward` | (-inf, 0.0] | Opt-in | `rewards.effort_proxy` |

## Model registry

| Slug | Model | Quantization | Max seq | Max LoRA rank |
|------|-------|-------------|---------|---------------|
| `qwen3-4b` | unsloth/Qwen3-4B-Base | 16-bit | 2048 | 32 |
| `qwen-1.5b` | Qwen/QwQ-1.5B | 4-bit | 2048 | 32 |
| `qwen-7b` | Qwen/QwQ-7B | 4-bit | 2048 | 64 |
| `llama-8b` | meta-llama/meta-Llama-3.1-8B-Instruct | 4-bit | 512 | 32 |
| `llama-1b` | meta-llama/Llama-3.2-1B-Instruct | 4-bit | 2048 | 16 |

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
