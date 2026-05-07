# AGENTS.md

## Setup

```bash
./setup.sh
```

Destroys and recreates `.venv` via `uv` (Python 3.12). Installs `unsloth`, a pinned `vllm` wheel (cu130), `ipykernel`, `ipywidgets`. `--torch-backend=auto` selects the right torch variant. **Hardware target: NVIDIA RTX 4090, CUDA 13.0, Linux.**

## Two code surfaces

| | Notebooks (root) | Pipeline (`pipeline/`) |
|---|---|---|
| Entry | `GRPO_Simple.ipynb`, `Qwen3_(4B)_GRPO.ipynb` | `python -m training.train --config configs/<exp>.yaml` |
| Run | Jupyter, top-to-bottom | CLI from `pipeline/` dir |
| Testable | No — validate by inspecting reward curves/outputs inline | No unit tests; validate by running an experiment and checking `runs/<exp_id>/eval_report.json` |
| Deps | Root `.venv` (setup.sh) | `pipeline/requirements.txt` (separate install) |

## Pipeline commands

All run from `pipeline/` directory:

```bash
python -m training.train --config configs/e0-baseline-math-1.5b.yaml
python -m training.train --config configs/e0-baseline-math-1.5b.yaml --eval
python -m eval.runner --config configs/e0-baseline-math-1.5b.yaml
python -m eval.runner --config configs/e0-baseline-math-1.5b.yaml --checkpoint runs/e0-baseline-math-1.5b/checkpoint-final
python -m eval.compare --runs runs/e0-baseline-math-1.5b runs/e1-token-entropy   # cross-experiment plots
```

New experiment: `cp configs/_template.yaml configs/e4-my-experiment.yaml`, edit, run.

## Architecture (pipeline)

- **Config-driven**: every experiment is a YAML file. Training script is a thin orchestrator.
- **Domain abstraction** (`domains/base.py`): `Domain` ABC — subclass to add new problem domains. Implements `load_dataset`, `extract_answer`, `difficulty`, `score_answer`, `score_numbers`, `is_correct`, `build_chat_template`.
- **Composable rewards** (`training/rewards/`): each reward is a callable `(prompts, completions, **kwargs) -> list[float]`. Register in `REWARD_REGISTRY` (`training/rewards/__init__.py`) with default enabled/weight + builder, and add the config key to `_KNOWN_REWARD_KEYS` in `training/config_schema.py` so typos fail validation.
- **Composer** (`training/rewards/compose.py`): `AdvantageWeightedComposer` (default, normalizes per-batch) or `NaiveSumComposer` (ablation).
- **Model registry** (`training/registry.py`): slug → model config. Add models there, not in configs.
- **Config validation** (`training/config_schema.py`): runs before model loading — fail-fast on bad config.
- **Callback-driven schedulers**: rewards with `step()` are advanced by `_RewardStepCallback` after each training step.

## Key conventions

- `model.slug` in config references `training/registry.py`, not a HuggingFace name directly.
- `lora_alpha` defaults to `lora_rank * 2`; override per-config via `model.lora_alpha` (read in `grpo_runner.py`).
- LoRA target modules are fixed in `registry.py:LORA_TARGET_MODULES` — not configurable per-experiment.
- Chat template is injected at runtime by `domain.build_chat_template(tokenizer)`, not stored in the model repo.
- `add_generation_prompt=True` prepends `<start_working_out>` to force reasoning mode.
- Output structure: `runs/<experiment_id>/` contains `config.yaml`, `checkpoint-final/`, `eval_report.json`, `eval_report.md`, and four PNGs — `training_curves.png`, `eval_accuracy.png`, `token_distribution.png`, `difficulty_scatter.png` (last one only for datasets with difficulty labels).
- `pipeline/runs/` and `unsloth_training_checkpoints/` are gitignored.

## Model output format

```
<start_working_out>
[reasoning chain of thought]
<end_working_out>
<SOLUTION>
[answer]
</SOLUTION>
```

`<start_working_out>` is prepended by the chat template. Reward functions check for the remaining tags.

## CodingDomain caveat

`CodingDomain.is_correct()` and `score_answer()` raise `NotImplementedError` — string/float comparison is invalid for code correctness. Don't enable accuracy/numeric rewards for coding experiments.

## Notebook-specific notes

- `GRPO_Simple.ipynb`: Llama-3.1-8B, 4-bit, `max_seq_length=512`, single reward (exact string match on `<answer>` tag). No SFT stage.
- `Qwen3_(4B)_GRPO.ipynb`: Qwen3-4B, 16-bit, `max_seq_length=2048`, two-phase (SFT format priming → GRPO). Four-function reward stack.
