# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
./setup.sh
```

Creates `.venv` via `uv` with Python 3.12. Installs `unsloth`, a pinned `vllm` wheel (cu130), `ipykernel`, and `ipywidgets`. Run notebooks via Jupyter after activating `.venv`.

Hardware target: NVIDIA RTX 4090, CUDA 13.0, Linux. `--torch-backend=auto` selects the right torch variant.

## Running Notebooks

```bash
source .venv/bin/activate
jupyter notebook
```

Both notebooks are self-contained and must be run top-to-bottom. They are not unit-testable — correctness is validated by inspecting reward curves and model outputs inline.

## Pipeline (`pipeline/`)

A separate, config-driven training and evaluation pipeline lives in `pipeline/`. It is the primary surface for systematic experimentation. Full documentation in `pipeline/README.md`.

Key entry points (run from `pipeline/`):
```bash
python -m training.train --config configs/e0-baseline-math-1.5b.yaml --eval
python -m eval.runner --config configs/e0-baseline-math-1.5b.yaml
python -m eval.compare --runs runs/e0-baseline runs/e1-token-entropy
```

Add `--smoke` to any command for a fast sanity check (3 steps, 10 eval samples).

Outputs land in `runs/<experiment_id>/`: frozen config, LoRA checkpoint, eval JSON/Markdown, and PNG plots (training curves, accuracy bars, token distribution, difficulty scatter).

## Architecture (Notebooks)

### Notebooks

**`GRPO_Simple.ipynb`** — Minimal GRPO baseline.
- Base model: `meta-llama/meta-Llama-3.1-8B-Instruct` (4-bit, LoRA rank 32, `max_seq_length=512`)
- Dataset: `openai/gsm8k`
- Single reward function: exact string match on extracted `<answer>` tag
- No SFT pre-finetuning stage

**`Qwen3_(4B)_GRPO.ipynb`** — Full two-phase reasoning model pipeline.
- Base model: `unsloth/Qwen3-4B-Base` (16-bit, LoRA rank 32, `max_seq_length=2048`)
- **Phase 1 — SFT format priming**: ~59 examples from `unsloth/OpenMathReasoning-mini`, 2 epochs. Teaches the model to emit the custom tag format before RL begins.
- **Phase 2 — GRPO**: `open-r1/DAPO-Math-17k-Processed`, filtered to 90th-percentile prompt length.

### Reward Stack (Qwen3 notebook)

Four functions composed additively per completion:
1. `match_format_exactly` (+3.0) — regex confirms full tag structure present
2. `match_format_approximately` (±0.5 per tag) — partial credit for individual tags
3. `check_answer` (up to +5.0) — exact/stripped/ratio match against ground truth string
4. `check_numbers` (±3.5) — `float()` conversion and numeric equality

The pipeline (`pipeline/training/rewards/`) reimplements these as classes
(`FormatExactReward`, `FormatApproxReward`, `AnswerReward`, `NumericReward`)
plus opt-in efficiency signals (`TokenLengthReward`, `TokenEntropyReward`,
`EffortProxyReward`). Pipeline `FormatApproxReward` counts `reasoning_end`
on the full text and the solution tags on the suffix to avoid CoT false
positives — semantics differ from the notebook version.

### Custom Chat Template

Reasoning tags are injected into a Jinja2 template assigned to `tokenizer.chat_template`. `add_generation_prompt=True` prepends `<start_working_out>` to force the model into reasoning mode before decoding begins.

### LoRA Configuration

Target modules across all notebooks: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. `lora_alpha = lora_rank * 2`. Gradient checkpointing via `use_gradient_checkpointing="unsloth"`.

### Outputs (Notebooks)

- Training checkpoints: `unsloth_training_checkpoints/` (gitignored)
- LoRA adapters: saved with `model.save_lora("grpo_saved_lora")`
- Merged exports / GGUF: generated on demand via `model.save_pretrained_merged()` / `model.save_pretrained_gguf()`

### Outputs (Pipeline)

- `runs/<experiment_id>/config.yaml` — frozen experiment config
- `runs/<experiment_id>/checkpoint-final/` — LoRA adapter + tokenizer
- `runs/<experiment_id>/eval_report.json` / `eval_report.md` — structured metrics and human-readable summary
- `runs/<experiment_id>/training_curves.png`, `eval_accuracy.png`, `token_distribution.png`, `difficulty_scatter.png` — auto-generated eval figures
- `runs/comparison/` — cross-experiment comparison plots from `eval.compare`
