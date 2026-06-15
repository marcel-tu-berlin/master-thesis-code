# OpenEnv agentic domains: design

Date: 2026-06-15
Branch: `feat/openenv-agentic-domains`
Status: approved (design); implementation plan to follow

## Context

The pipeline trains reasoning models with GRPO on a single domain: math (GSM8K, Hendrycks MATH, DAPO). The training loop is dataset-centric. `Domain.load_dataset()` returns a static HuggingFace `Dataset` of prompts, and `GRPORunner.train(dataset, reward_fn)` runs single-turn GRPO where rewards are pure functions of the completion text.

The next research step is agentic, multi-turn training across multiple domains, using OpenEnv (the Meta PyTorch / Hugging Face environments hub) as the environment layer. OpenEnv is environment-centric: an environment runs as an HTTP/WebSocket server exposing `reset()` and `step()`, the trainer drives rollouts against it, and the environment computes the task reward. TRL's GRPOTrainer consumes an environment through a `rollout_func` or an `environment_factory`.

This is not a new `Domain` subclass. It changes the shape of the training loop, not just the data source.

## Decisions (settled in the design interview)

1. **Full migration to vanilla TRL.** Drop unsloth. One `GRPORunner` on vanilla TRL (>= 0.26) plus PEFT and bitsandbytes serves both single-turn and agentic modes. Rationale: unsloth's GRPOTrainer silently ignores `rollout_func` (unslothai/unsloth#3573), so the OpenEnv path needs vanilla TRL anyway. A single stack matches the project's preference for one principled default over two coexisting trainers. Consequence: no captured e0 baseline exists to reproduce, so the migrated math results become the project baseline, validated for sanity (a clear gain over the base model, no collapse) rather than bit-parity with the old stack.

2. **reasoning_gym is the first agentic domain.** Closest to the current math domain, so graders and efficiency rewards port most directly. It is single-step (one action per episode), which keeps the first OpenEnv integration simple. textarena and coding_env are the intended follow-ups and are genuinely multi-turn.

3. **Efficiency rewards are first-class in the agentic loop.** The thesis through-line is reasoning efficiency (cosine-length, token-entropy). These port to the agentic loop at episode level, counting model-generated tokens only, composed with the environment reward through the existing advantage-weighted composer.

4. **Integration path: `rollout_func`, not `environment_factory`.** The efficiency rewards need explicit access to `completion_ids` (model-token-only accounting). `rollout_func` returns those plus the environment reward as kwargs. `environment_factory` auto-masks the loss but does not reliably expose per-token masks to reward functions. For single-step reasoning_gym the manual rollout loop is short.

5. **Agentic default model: `qwen3-1.7b` (`Qwen/Qwen3-1.7B`).** A 7B model does not fit colocated vLLM on a 24 GB GPU. 1.7B leaves headroom; 4B is a stretch target. Existing 7B math configs keep running on the migrated stack.

## Goals

- Add an agentic, multi-turn training mode backed by OpenEnv, with reasoning_gym as the first domain.
- Keep one training stack (vanilla TRL + PEFT + bitsandbytes) for both modes.
- Preserve the efficiency-reward research line in the agentic setting (model-token-only).
- Reuse the reward composer, metrics, report, and plotting layers unchanged where possible.
- Re-validate the migrated single-turn math path against the pre-migration baseline before building anything agentic.

## Non-goals

- Multi-turn environments beyond reasoning_gym (textarena, coding_env). The abstraction is designed for them; they are not built here.
- Distributed or multi-GPU training. Single 24 GB GPU (the L4 node described under Hardware and deployment), colocate vLLM.
- Changing the reward composition math (advantage-weighted z-scoring stays as is).
- Re-tuning math hyperparameters. The migration aims to reproduce, not improve, the existing results.

## Architecture

### Current (dataset-centric)

```
config -> validate -> build_domain (MathDomain)
       -> GRPORunner (unsloth FastLanguageModel + TRL GRPOTrainer)
       -> domain.load_dataset() -> Dataset
       -> build_reward_components + compose
       -> runner.train(dataset, reward_fn)   # train_dataset=...
       -> save_lora -> eval
```

### Target (one stack, two modes)

```
config -> validate -> dispatch on training.mode
       -> GRPORunner (vanilla AutoModelForCausalLM + PEFT + TRL GRPOTrainer)
       |
       +-- mode: dataset (default)
       |     domain.load_dataset() -> Dataset
       |     runner.train(dataset=..., reward_fn=...)            # train_dataset
       |
       +-- mode: agentic
             env_domain.make_client() -> OpenEnv reasoning_gym
             runner.train(env_domain=..., reward_fn=...)         # rollout_func
       |
       -> save_lora -> eval (dataset probes OR episode probes)
```

The dispatch point is one place in `train.py`. Everything downstream (composer, metrics, report, plots) is shared.

## Detailed design

### Stack migration (unsloth to vanilla TRL)

`training/grpo_runner.py` and `eval/runner.py` both load the model through `unsloth.FastLanguageModel`. Replace with:

- Model load: `transformers.AutoModelForCausalLM.from_pretrained` with a `BitsAndBytesConfig` (nf4, double-quant, bf16 compute) when `load_in_4bit` is set.
- LoRA: `peft.LoraConfig` + `get_peft_model`, same target modules as today (`LORA_TARGET_MODULES`).
- Gradient checkpointing: standard `gradient_checkpointing=True` plus `model.enable_input_require_grads()`, replacing `use_gradient_checkpointing="unsloth"`.
- bf16 probe: `torch.cuda.is_bf16_supported()`, replacing unsloth's `is_bfloat16_supported`.
- vLLM: TRL `GRPOConfig(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=...)`, replacing unsloth's `fast_inference` kwarg path.
- Eval inference: `model.eval()` plus `PeftModel.from_pretrained` for the adapter (eval already uses `PeftModel`), replacing `FastLanguageModel.for_inference`.

`registry.py` model names lose the unsloth assumption. `unsloth/Qwen3-4B-Base` becomes a standard HuggingFace id. Add `qwen3-1.7b -> Qwen/Qwen3-1.7B`.

### Environment abstraction

`domains/base.py` keeps `Domain` (dataset). Add a sibling `EnvDomain` base. The reasoning-tag constants and `build_chat_template` move into a small shared mixin so both bases reuse them without duplication.

`EnvDomain` interface:

- `make_client()` returns an OpenEnv environment client (reasoning_gym), configured from `training.env_config`.
- `system_prompt` and the chat-template tags (shared mixin).
- `episode_reward(step_result) -> float` reads the environment reward from the OpenEnv `StepResult`.
- `is_correct(step_result) -> bool` for eval success scoring.
- `difficulty(task) -> float | None` when the environment exposes a difficulty label.

Concrete implementation: `domains/reasoning_gym/`.

### Training: the rollout function

`GRPORunner` gains an agentic branch. With vanilla TRL >= 0.26:

- `runner.train(env_domain=..., reward_fn=...)` builds a `rollout_func` and passes it to `GRPOTrainer` with `use_vllm=True, vllm_mode="colocate"`.
- The `rollout_func` generates completions through `trl.experimental.openenv.generate_rollout_completions(trainer, prompts)`, steps the environment client per completion, and returns a dict with `prompt_ids`, `completion_ids`, `logprobs`, and `env_reward` (one entry per rollout).
- Prompt replication preserves grouping: each task prompt is repeated `n_rollouts` times consecutively, so `AdvantageWeightedComposer._group_indices` partitions groups correctly (it compares consecutive prompts with `==`).

For single-step reasoning_gym the loop is: `reset(task)` once per task, generate, `step(completion)`, read reward. The completion is entirely model-generated, so model-token accounting is the whole completion. The multi-turn case (interleaved environment observations, a per-token model mask) is designed for but implemented when the first multi-turn environment lands.

### Rewards

- New `EnvReward` component, config key `rewards.env_reward`, registered in `REWARD_REGISTRY` and `_KNOWN_REWARD_KEYS`. It returns the environment reward passed via `kwargs["env_reward"]`. In agentic mode it is the task-success signal, the analog of `accuracy` in the dataset path. Default-enabled in agentic mode.
- `CosineLengthReward` and `TokenEntropyReward` consume `completion_ids` (model tokens). `TokenEntropyReward` already prefers `kwargs["completion_ids"]` and runs its own forward pass, so it ports with no shape change. `CosineLengthReward` currently re-encodes completion text; switch it to count `completion_ids` when provided. For reasoning_gym single-step these are the whole completion.
- The composer is unchanged. The environment reward is one more component fed into the per-group z-scoring. Origin (environment vs code) does not affect the math.

### Evaluation

Add an episode-driven probe path in `eval/ood_probes.py` (`_run_episodes`) that drives the environment client and scores each episode. The metrics, report, and plotting layers are reused: `EvalMetrics` keeps `correct` (episode success) and `n_tokens` (model tokens per episode). Add one optional field, `n_steps` (episodes per environment; 1 for reasoning_gym), surfaced in the report.

Probe mapping for the agentic mode:

- `id_split`: reasoning_gym at the trained configuration, held-out seed range.
- `near_ood`: reasoning_gym at a different task type or higher difficulty.
- `far_ood`: keep MMLU. It is model-level and environment-agnostic, so it ports unchanged.
- `capability_floor`: keep the instruction-following floor. Environment-agnostic, ports unchanged.

### Config and schema

- `training.mode: dataset | agentic`, default `dataset`.
- `training.env`: OpenEnv environment id (agentic mode).
- `training.env_config`: environment parameters (reasoning_gym task type, difficulty, seed range).
- `eval.agentic`: episode count and probe settings for the agentic eval path.
- Relax `_REQUIRED_KEYS`: `training.dataset` is required only in dataset mode; agentic mode requires `training.env` instead.
- Add the new keys to the schema whitelist and `rewards.env_reward` to the reward whitelist.

New reference config: `configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml`.

### Dependencies

`setup.sh` changes:

- Remove `unsloth`.
- Add `trl>=0.26`, `transformers`, `peft`, `bitsandbytes`, `accelerate`.
- Keep a `vllm` compatible with TRL 0.26 colocate (current pin is the 0.19.1 cu130 wheel; compatibility is a spike).
- Add `openenv[core]` and the reasoning_gym environment client (installed from its Hugging Face Space, or run via OpenEnv's local UV provider to isolate its dependencies from the training environment).

## Hardware and deployment

Validation runs on an NVIDIA L4 node (DevPod, reachable over `ssh gpu-l4`), not the RTX 4090 the configs were first written against. The relevant facts:

- L4, 23 GB VRAM, idle. Same 24 GB class as the 4090, so the model-size reasoning holds (7B tight, 1.7B agentic with colocate headroom). The L4 is a slower inference-class card, so wall-clock is longer: unit tests and `--smoke` runs finish quickly, a full 500-step e0 is an overnight-class job.
- CUDA 13.0 (driver 580.95), Python 3.12.3, uv 0.9.25. These match the pipeline's targets, so the cu130 wheels install without special handling.
- 48 vCPU, 188 GB RAM, 3.2 TB free. CPU-side work (data loading, environment servers) is unconstrained.

Deployment: rsync the working tree (including `.git`, so the box carries the tag and branch) to `/workspace/master-thesis-code`, excluding `.venv*`, `runs/`, and caches. Two environments live on the box: `.venv-test` (CPU torch, the fast unit-test loop) and `.venv` (the full GPU stack from `setup.sh`). The 86-test unit suite runs in `.venv-test` in a few seconds.

## Baseline establishment and sanity gate

No e0 result exists in the repo, and the project holds no e0 numbers to reproduce. So the migration establishes the baseline rather than matching one: run e0 once on the vanilla TRL stack, and that `eval_report.json` becomes the project's reference for every later delta.

A gate still sits between migration and agentic work, but it is a sanity check, not a parity check. The migrated e0 passes when:

- id_split accuracy is clearly above the base-model baseline (the `--baseline` pass over the same probes),
- the model has not collapsed: capability_floor stays near its base-model level, and mean token count sits in a sane range rather than at the format floor,
- training curves show reward rising and KL bounded.

Agentic implementation does not start until this passes. The `pre-agentic-math-only` tag remains the code snapshot of the pre-migration state; it is not used for numeric comparison.

## Risks and open questions

- **Dependency resolution (highest risk).** TRL 0.26 plus a compatible vLLM plus transformers, alongside the OpenEnv client. Resolved in milestone 0 by a smoke install and a 3-step run. The OpenEnv client may need isolation (UV provider) to avoid clashing with the pinned vLLM.
- **Migration reproduction.** The e0 baseline may not reproduce exactly. The gate uses a tolerance, not bit-exactness. If it fails, investigate quantization config (nf4 vs unsloth defaults) and seeding before proceeding.
- **`generate_rollout_completions` API churn.** It lives under `trl.experimental`. Pin the TRL version and isolate the call behind the runner.
- **Concurrency.** OpenEnv servers default to one session; the trainer opens one per generation. Set `SUPPORTS_CONCURRENT_SESSIONS` and `max_concurrent_envs >= generation_batch_size`, or run reasoning_gym single-session with a small batch.
- **OpenEnv Python version.** PyPI lists `openenv-core` as `>=3.10`; a community doc claims `>=3.13`. Verify against the 3.12 environment at install.

## Milestones

Front-loaded so the migration risk and the regression gate come before agentic work. Each milestone has a verification gate. TDD applies to the pure-logic surfaces: write the CPU test first (runnable in `.venv-test`), then the implementation. GPU-dependent paths are verified by smoke runs on the GPU box.

- **M0: dependency and stack spike.** Build the new environment with the rewritten `setup.sh` (vanilla TRL, PEFT, bitsandbytes, accelerate, a TRL-0.26-compatible vLLM, openenv-core). Confirm it resolves on the L4, then confirm a 3-step vanilla GRPO run on GSM8K completes (proves the migrated single-turn path trains). Verify: the environment resolves; the smoke train logs reward and loss. No unsloth build (establish-fresh makes it unnecessary).
- **M1: single-turn migration and sanity gate.** Rewrite `GRPORunner` and `eval/runner` on vanilla TRL + PEFT, keeping Domain, rewards, composer, and eval logic. Run full e0 plus its `--baseline` pass; the trained report becomes the project baseline. Verify: the sanity gate above passes (clear gain over base model, no collapse). This gate blocks agentic work.
- **M2: OpenEnv reasoning_gym path.** `EnvDomain` plus reasoning_gym plus the `rollout_func` wiring plus the `train.py` dispatch. Verify: smoke agentic train (3 steps) runs, the environment reward flows into the composer, a checkpoint saves.
- **M3: efficiency rewards in agentic.** Wire `CosineLengthReward` and `TokenEntropyReward` to model tokens in the agentic loop. Verify: efficiency components appear in the training log; an agentic config with `token_length` enabled trains.
- **M4: agentic eval.** `_run_episodes` probe plus the `n_steps` metric. Verify: an `eval_report.json` for an agentic run with success rate and model-token efficiency.
- **M5: configs and docs.** The `e5` reference config plus documentation. Verify: `validate_config` passes for the new config; docs read correctly.

## Documentation to adapt (part of M5)

- `CLAUDE.md`: environment setup (drop unsloth, add TRL and OpenEnv), the pipeline domains list (no longer math-only), the agentic mode, the environment-reward component, the agentic eval path. The notebook sections describe an unsloth stack that the migration removes; reconcile them with the new reality.
- `pipeline/README.md`: agentic mode, the `EnvDomain` abstraction, the new config keys, the reward table (add `env_reward`), the model registry (add `qwen3-1.7b`), the eval section (episode probes).
- `pipeline/configs/_template.yaml`: the `training.mode`, `training.env`, `training.env_config`, `eval.agentic` keys, and `rewards.env_reward`.
- `setup.sh`: the dependency changes.
- Docstrings across the migrated runner, env domain, and eval modules.

## Success criteria

- One training stack (vanilla TRL + PEFT + bitsandbytes); unsloth removed from the pipeline.
- e0 trains and evaluates sanely on the migrated vanilla-TRL stack (clear gain over the base model, no collapse) and is recorded as the project baseline.
- An agentic reasoning_gym GRPO run trains end to end on a single RTX 4090, with the environment reward and at least one efficiency reward composed through the advantage-weighted composer.
- An agentic eval report with episode success rate and model-token efficiency.
- CLAUDE.md, the pipeline README, the config template, and setup.sh describe the agentic pipeline accurately.
- CPU-logic tests cover the new schema rules, the mode dispatch, the env-reward wiring, and the eval-episode aggregation.
