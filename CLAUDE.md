# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

The project studies token-efficiency reward shaping in GRPO (cosine length
reward, token entropy) in an agentic, multi-environment setting. Training runs
against live OpenEnv environments through TRL's `environment_factory`; the policy
is a tool-calling model (Qwen3-1.7B) rewarded by the environment, not by grading
an answer string. The pipeline is agentic-only.

## Environment Setup

```bash
./setup.sh
```

Creates `.venv` via `uv` with Python 3.12. Installs the GPU stack (`trl`, `peft`,
`bitsandbytes`, `accelerate`, a pinned `vllm` cu130 wheel, `openenv-core`,
`reasoning-gym`) and clones `meta-pytorch/OpenEnv` to `/workspace/OpenEnv` (its
env servers are not on PyPI). Hardware target: NVIDIA L4 (24 GB) or RTX 4090,
CUDA 13.0, Linux. `--torch-backend=auto` selects the torch variant.

## Running the Pipeline (`pipeline/`)

The pipeline is the surface for systematic experimentation. Full docs in
`pipeline/README.md`. Run from `pipeline/`:

```bash
python -m training.train --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml --eval
python -m eval.runner --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml
python -m training.batch configs/e5-*.yaml --train --eval --seeds 42 43 44
```

Add `--smoke` to any command for a fast sanity check (3 steps, 10 eval episodes).

`training.train` refuses to clobber an existing `runs/<experiment_id>/` directory.
Pass `--overwrite` to replace, or change `experiment_id`. The frozen `config.yaml`
and `checkpoint-final/` are the trigger artifacts.

Outputs land in `runs/<experiment_id>/`: frozen config, LoRA checkpoint, and the
agentic eval report (`eval_report.json` / `eval_report.md`, keyed under the
`agentic` split).

### Batch runner

`training.batch` queues many configs through training and eval as subprocesses
(one fresh Python process per phase, so GPU memory is released cleanly between
runs). Built for unattended ablation and seed sweeps on a single GPU.

```bash
python -m training.batch configs/e5-*.yaml --train --eval
```

Phase flags `--train` and `--eval` are independent and combinable; default when
neither is given is `--train --eval`. Other flags: `--seeds A B C` (replicate
each config across seeds, each into its own `<exp>-s<seed>` run dir), `--smoke`,
`--vllm`, `--force` (re-run even if outputs exist; passes `--overwrite` to train),
`--retries N`. Skip predicates are content-aware and resume-friendly: a
non-smoke `checkpoint-final/` skips train and a real `eval_report.json` skips
eval, so a re-run after a crash picks up where it stopped. Per-phase logs land at
`runs/<exp>/batch_{train,eval}.log`; an end-of-batch summary is written to
`runs/batch_summary_<timestamp>.md`.

## Architecture

### Domains

A domain wraps one OpenEnv environment. `EnvDomain` (`domains/env_base.py`) is the
interface: `make_env_factory` (a zero-arg callable that builds one env adapter
against the server `base_url`), `build_seed_dataset` (rows of `{prompt, seed}`,
one distinct question per seed), `episode_messages` (the eval prompt for a
question), `episode_reward` / `is_correct` (read the env score), and
`server_module` (the `python -m ...` server entry point the runner launches).

`ReasoningGymDomain` (`domains/reasoning_gym/domain.py`) is the reference
environment (reasoning_gym task families, e.g. `chain_sum`).

### Agentic training loop

`training.mode: agentic` with `training.env: reasoning_gym` trains against a live
OpenEnv server. The model is driven through its native tool-calling template and
rewarded by the environment.

- **Server lifecycle (runner-owned).** `EnvServerProcess` (`training/env_server.py`)
  launches the OpenEnv env as a local HTTP server subprocess (no Docker:
  `python -m reasoning_gym_env.server.app`), waits for the port, and stops it after
  training. The server code lives in a clone of `meta-pytorch/OpenEnv` at
  `training.env_server.repo_path` (default `/workspace/OpenEnv/envs`). One server
  serves every rollout-slot client; `MAX_CONCURRENT_ENVS` is sized to
  `batch_size * n_rollouts`.
- **Env-factory adapter.** `ReasoningGymEnvAdapter` (`domains/reasoning_gym/adapter.py`)
  wraps the OpenEnv sync client. TRL's `GRPOTrainer(environment_factory=...)` builds
  one adapter per rollout slot, calls `reset(**row)` (its return is appended to the
  prompt), and exposes every other public method as a tool. The adapter's public
  surface is exactly `{reset, answer}`, so the model sees one tool, `answer(answer:
  str)`; the tool docstring needs a Google-style `Args:` block (transformers builds
  the tool JSON schema from it). The adapter stores the env score on `self.reward`.
- **Dataset.** `build_seed_dataset` returns `{prompt, seed}` rows; each seed is a
  distinct, deterministic reasoning_gym question. TRL repeats each row
  `num_generations` times to form a GRPO group.
- **Rewards.** `EnvReward` reads `[e.reward for e in kwargs["environments"]]`.
  The efficiency rewards are the other live signals; `CosineLengthReward` takes
  correctness from `environments` (env reward > 0) instead of an answer column.

Validated end to end on an L4 (24 GB) with `Qwen/Qwen3-1.7B` + vLLM colocate: the
model calls the tool, env reward flows into the composer, and the LoRA saves.

### Reward Registry

Rewards are wired via `REWARD_REGISTRY` in `pipeline/training/rewards/__init__.py`.
Each entry maps a config key (under `rewards:`) to `(default_enabled,
default_weight, builder)`: `env_reward` (task success), `token_length` (cosine
length), `token_entropy`. All default off; configs enable what they study. Adding
a reward requires both a builder + registry entry and the matching key in
`_KNOWN_REWARD_KEYS` (`pipeline/training/config_schema.py`), or validation rejects
it. `train.build_reward_components` iterates the registry, so there are no
per-reward branches in `train.py`.

`CosineLengthReward` (Wu/Yeo 2025) is the single token-length reward: correct
completions are rewarded more when shorter, wrong completions penalized less when
longer, making wrong-and-short the most-penalized cell. The reward is non-linear
in length and gated by correctness, so it survives per-group z-scoring with real
structure. `TokenEntropyReward.fork_mask_top_frac` averages entropy over the top
fraction of tokens by entropy.

### Reward Composition

Components are combined via a composer selected by `rewards.compose_method`:

- **`advantage_weighted`** (default) - `AdvantageWeightedComposer`
  (`pipeline/training/rewards/compose.py`). Per-prompt-group z-scoring of each
  component's raw rewards before the weighted sum (DIET 3.2): raw variance differs
  across components, so a naive sum lets a high-variance signal dominate regardless
  of weight. A component with zero within-group variance contributes 0, by design.
- **`naive_sum`** - `NaiveSumComposer`. Plain weighted sum, no normalisation. The
  ablation baseline that isolates the advantage-weighting effect.

**Scale-invariance.** Because `advantage_weighted` z-scores each component per
prompt-group, it is invariant to any global positive scalar in a component's raw
reward (a negative scalar flips the sign; zero silences it). Under the default
composer, `token_entropy.reward_scale` therefore does nothing. The live levers are
the component `weight`, the per-completion signal shape, and switching to
`naive_sum`. `build_reward_components` warns when a knob is inert as configured.

### Agentic Evaluation

`eval.runner` dispatches to `run_agentic_eval` (`pipeline/eval/agentic_eval.py`):
it loads the trained LoRA, launches the env server, runs N held-out episodes
(seeds disjoint from training via a +100000 offset), generates a greedy tool call
per episode, parses the `answer`, scores it via the env, and writes
`eval_report.json` / `.md` keyed under the `agentic` split. The generation budget
defaults to the training budget (`max_seq - max_prompt_length`); a smaller eval
cap silently truncates long completions before the tool call and tanks the
success rate. Metrics come from `eval/metrics.py`: accuracy with Wilson 95%
interval, mean token count with bootstrap CI, underthinking / overthinking rates,
and mean steps.

### LoRA Configuration

Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`. `lora_alpha = lora_r * 2`. The model loads in bf16 (or 4-bit nf4 when
the registry sets `load_in_4bit`), with vLLM colocate generation and micro-batch +
grad-accum to fit 24 GB.

### Outputs

- `runs/<experiment_id>/config.yaml` - frozen experiment config
- `runs/<experiment_id>/checkpoint-final/` - LoRA adapter + tokenizer
- `runs/<experiment_id>/eval_report.json` / `eval_report.md` - agentic episode
  metrics (success rate, token efficiency)
