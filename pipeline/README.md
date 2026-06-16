# GRPO Training Pipeline (agentic)

A GRPO training and evaluation pipeline for token-efficiency reward shaping in an
agentic setting. The policy is driven through its native tool-calling template
and rewarded by a live OpenEnv environment, not by grading an answer string.
Experiments are YAML configs, so swapping reward signals or an environment does
not need a code change.

## Setup

- Python 3.12, CUDA 13.0, an NVIDIA GPU (developed on L4 24 GB / RTX 4090).

```bash
./setup.sh
```

This creates `.venv` via `uv`, installs the GPU stack (`trl`, `peft`,
`bitsandbytes`, a pinned `vllm` cu130 wheel, `openenv-core`, `reasoning-gym`),
and clones `meta-pytorch/OpenEnv` to `/workspace/OpenEnv` (its env servers are
not on PyPI). Point `training.env_server.repo_path` at that clone's `envs/` dir.

## Train

```bash
cd pipeline
python -m training.train --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml --eval
```

`--eval` runs held-out episode evaluation after training. `--smoke` overrides the
config for a fast sanity check (3 steps, 2 rollouts, 512 seq, 10 eval episodes):

```bash
python -m training.train --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml --smoke --eval
```

### How agentic training works

- The runner launches the OpenEnv env server as a local subprocess (no Docker)
  and stops it when training ends. One server serves every rollout-slot client;
  `MAX_CONCURRENT_ENVS` is sized to `batch_size * n_rollouts`.
- Each training prompt is one environment question, selected by a seed. TRL's
  `environment_factory` builds one env adapter per rollout slot, calls
  `reset(**row)` (its return is appended to the prompt), and exposes the
  adapter's `answer` method as a tool. The model emits a tool call, the env
  scores the answer, and `EnvReward` reads that score off the env instance.
- Reward comes from the environment, so there is no answer column and no
  reasoning-tag format. The live signals are `env_reward` plus the
  token-efficiency rewards (`token_length`, `token_entropy`).
  `CosineLengthReward` takes correctness from the env (reward > 0).

## Evaluate a checkpoint

```bash
python -m eval.runner --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml
```

Runs N held-out episodes (default 100, set by `eval.agentic.n_episodes`) on seeds
disjoint from training (a fixed +100000 offset), parses each tool call, scores it
via the env, and writes `runs/<exp>/eval_report.json` + `.md` under the `agentic`
split. Override the checkpoint with `--checkpoint`, the generation budget with
`--max_new_tokens` (defaults to the training budget, `max_seq - max_prompt_length`),
or cap to 10 episodes with `--smoke`.

## Batch run

`training.batch` queues many configs through training and eval as subprocesses
(one fresh Python process per phase, so GPU memory is released between runs).
Built for unattended ablation and seed sweeps on a single GPU.

```bash
python -m training.batch configs/e5-*.yaml --train --eval
python -m training.batch configs/e5-*.yaml --train --eval --seeds 42 43 44
```

| Flag | Behaviour |
|------|-----------|
| `--train` | Run `training.train` for each config |
| `--eval` | Run `eval.runner` for each config (after train if both given) |
| `--seeds A B C` | Replicate each config across these seeds; each gets its own `<exp>-s<seed>` run dir |
| `--smoke` | Pass `--smoke` to every subprocess |
| `--force` | Re-run phases even if outputs exist (passes `--overwrite` to train) |
| `--retries N` | Retry a failed phase N times (default 1) |
| `--vllm` | Route training rollouts through vLLM |

Default with no phase flag is `--train --eval`. Skip predicates are
content-aware and resume-friendly: a non-smoke `checkpoint-final/` skips train, a
real `eval_report.json` skips eval. Per-phase logs land at
`runs/<exp>/batch_{train,eval}.log`; an end-of-batch summary is written to
`runs/batch_summary_<timestamp>.md`. Exit code is non-zero if any phase failed.

## Run a new experiment

1. Copy the template: `cp configs/_template.yaml configs/e6-my-experiment.yaml`
2. Edit `experiment_id`, toggle reward signals under `rewards:`, adjust weights.
3. Run: `python -m training.train --config configs/e6-my-experiment.yaml --eval`
4. Results land in `runs/<experiment_id>/`.

## Add a new environment

The pipeline is built to grow across OpenEnv environments. To add one:

1. Subclass `EnvDomain` (`domains/env_base.py`). Implement `make_env_factory`
   (returns a zero-arg callable that builds one env adapter against the server
   `base_url`), `build_seed_dataset` (rows of `{prompt, seed, ...}`, one distinct
   question per seed), `episode_messages` (the eval prompt for a question), and
   set `server_module` (the `python -m ...` server entry point).
2. Write an adapter whose public surface is exactly `{reset, answer}`. TRL turns
   every public method except `reset` into a tool, so keep it minimal. The
   `answer` tool's docstring needs a Google-style `Args:` block or transformers
   cannot build its JSON schema. Store the env score on `self.reward`.
3. Register the env id in `build_domain` (`training/train.py`) and in
   `eval/runner.py`.

`reasoning_gym` is the reference implementation.

## Architecture

```
Config YAML
    |
    v
train.py:main()
    +-- validate_config()        # agentic schema check, fail fast
    +-- build_domain()           # env id -> EnvDomain (reasoning_gym)
    +-- GRPORunner(config)        # load model + tokenizer + LoRA
    +-- build_reward_components() # (reward_fn, weight) from REWARD_REGISTRY
    +-- build_composer()          # AdvantageWeighted | NaiveSum
    +-- domain.build_seed_dataset()   # {prompt, seed} rows, one question each
    +-- build_env_server()        # EnvServerProcess (no Docker)
    +-- runner.train(server=, make_factory=)
    |       |  starts the env server, builds TRL environment_factory, then:
    |       v
    |   TRL GRPOTrainer (per step):
    |       1. one env adapter per rollout slot; reset(**row) -> prompt
    |       2. model emits a tool call; adapter.answer(...) scores via the env
    |       3. reward_fn reads [e.reward for e in environments] + length signal
    |       4. advantages + GRPO loss + LoRA backprop
    +-- runner.save_lora()
    |
    v  (optional --eval)
eval/agentic_eval.py:run_agentic_eval()  # N held-out episodes, env-scored report
```

## Modules

### `configs/`

YAML experiment configs. `_template.yaml` documents every field; copy it to make
a new experiment. `e5-agentic-reasoning-gym-qwen3-1_7b.yaml` is the reference run
(chain_sum, Qwen3-1.7B, env reward + cosine length).

### `domains/`

- `env_base.py` - `EnvDomain`, the interface every environment implements.
- `reasoning_gym/adapter.py` - `ReasoningGymEnvAdapter`, the minimal `{reset,
  answer}` wrapper over the OpenEnv sync client. `answer(answer: str)` is the one
  tool the model sees; it stores the env score on `self.reward`.
- `reasoning_gym/domain.py` - `ReasoningGymDomain`: env factory, seed-row
  dataset, and the `server_module` the runner launches.

### `training/`

- `train.py` - entry point: validate, build domain, load model, compose rewards,
  build the seed dataset and env server, train, save, optionally eval.
- `grpo_runner.py` - `GRPORunner` wraps TRL `GRPOTrainer` with PEFT LoRA (bf16 or
  4-bit nf4, vLLM colocate, micro-batch + grad-accum to fit 24 GB). In agentic
  training it owns the env-server lifecycle and passes TRL an
  `environment_factory`.
- `env_server.py` - `EnvServerProcess` launches/waits/stops the OpenEnv server
  subprocess; `build_env_server(config, domain)` sizes concurrency from the
  config.
- `registry.py` - model slugs -> loader configs. `get_model_config(slug)` raises
  with the available options on an unknown slug.
- `config_schema.py` - agentic config validation (requires `training.env`,
  rejects unknown reward keys), run before any model loading.
- `batch.py` - multi-experiment runner (train/eval phases, seed replication,
  resume-skip predicates, per-phase logs, summary).

### `training/rewards/`

Each reward is a callable `(prompts, completions, **kwargs) -> list[float]`.
`__init__.py` exposes `REWARD_REGISTRY: key -> (default_enabled, default_weight,
builder)`; `train.build_reward_components` iterates it. Adding a signal needs both
a registry entry and the key in `config_schema._KNOWN_REWARD_KEYS`.

- `env_reward.py` - `EnvReward` reads `[e.reward for e in kwargs["environments"]]`
  (the env's task-success score). The live correctness signal in agentic mode.
- `cosine_length.py` - `CosineLengthReward` (Wu/Yeo 2025): correct completions are
  rewarded more when shorter, wrong completions penalized less when longer, so
  wrong-and-short is the most-penalized cell. Correctness comes from the env. The
  reward is non-linear in length and correctness-gated, so it survives the
  advantage-weighted per-group z-scoring with real structure.
- `token_entropy.py` - `TokenEntropyReward`: mean per-token entropy from model
  logits over a batched forward pass. `fork_mask_top_frac` averages over the
  top fraction of tokens by entropy.
- `compose.py` - `AdvantageWeightedComposer` (z-scores each component per
  prompt-group before the weighted sum) and `NaiveSumComposer` (plain weighted
  sum, the ablation baseline). `build_composer(components, method)` is the factory.
- `utils.py` - `extract_content(completion)`, safe access to completion text.

### `eval/`

- `agentic_eval.py` - `run_agentic_eval`: loads the LoRA, launches the env
  server, runs N held-out episodes, parses the tool call, scores via the env, and
  writes the report under the `agentic` split.
- `runner.py` - thin `python -m eval.runner` entry that dispatches to
  `run_agentic_eval`.
- `metrics.py` - `SampleResult` and `compute_metrics`: accuracy with Wilson 95%
  interval, mean token count with bootstrap CI, underthinking / overthinking
  rates, mean steps.

## Reward composition and scale-invariance

`advantage_weighted` z-scores each component per prompt-group before the weighted
sum (DIET 3.2): raw variance differs across components, so a naive sum lets a
high-variance signal dominate regardless of weight. Because z-scoring cancels any
global positive scalar, `token_entropy.reward_scale` is inert under
`advantage_weighted` - the live levers are the component `weight`, the
per-completion signal shape, and switching to `naive_sum`. `build_reward_components`
warns when a configured knob is inert.

## Reward signal reference

| Signal | Class | When enabled | Config key |
|--------|-------|--------------|------------|
| Env reward (task success) | `EnvReward` | Opt-in | `rewards.env_reward` |
| Token length (cosine) | `CosineLengthReward` | Opt-in | `rewards.token_length` |
| Token entropy | `TokenEntropyReward` | Opt-in | `rewards.token_entropy` |

All default off; agentic configs enable `env_reward` plus the efficiency signals
they study.

## Model registry

| Slug | Model | Quantization | Max seq | Max LoRA rank |
|------|-------|-------------|---------|---------------|
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | bf16 | 2048 | 32 |
| `qwen3-4b` | Qwen/Qwen3-4B-Base | bf16 | 2048 | 32 |
| `qwen-1.5b` | Qwen/Qwen2.5-1.5B | 4-bit | 2048 | 32 |
| `qwen-7b` | Qwen/Qwen2-7B | 4-bit | 2048 | 64 |

`qwen3-1.7b` is the default target (bf16, plays cleanly with vLLM colocate). Add a
new model by adding an entry to `MODEL_REGISTRY` in `training/registry.py`.

## Outputs

```
runs/<experiment_id>/
  config.yaml          # frozen copy of the config used
  checkpoint-final/    # LoRA adapter + tokenizer
  eval_report.json     # structured metrics under results.agentic
  eval_report.md       # human-readable summary
```
