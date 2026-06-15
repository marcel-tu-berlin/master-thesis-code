# B6 findings: OpenEnv reasoning_gym integration

Date: 2026-06-15. Captured during B6 prep (env inspected in a scratch checkout; training `.venv` untouched).

## TRL 1.6 rollout_func contract (pinned from source)

- `GRPOTrainer(rollout_func=...)`. Called as `output = rollout_func(prompts, trainer)`.
- `output` must be a dict with required keys `{"prompt_ids", "completion_ids", "logprobs"}` (lists, one entry per prompt).
- Any **extra keys flow to reward functions as kwargs** (trainer merges `extra_fields`). So returning `env_reward` is exactly how it reaches `EnvReward` (B3 is correct).
- Special extra key `env_mask`: treated as `tool_mask`, marks model tokens (1) vs environment tokens (0). Only needed for multi-turn envs; single-step reasoning_gym needs none (the whole completion is model tokens).
- `trl.experimental.openenv.generate_rollout_completions(trainer, prompts, *, generation_overrides=None, as_chat=None) -> list[dict]`, one dict per prompt with keys `prompt_ids`, `completion_ids`, `logprobs`, and decoded `text`. Use `out["text"]` for the env action; no manual decode. Requires `use_vllm=True` (colocate or server).

## reasoning_gym OpenEnv env API (`OpenEnv/envs/reasoning_gym_env`)

- Client `ReasoningGymEnv`; models `ReasoningGymAction(answer: str)`, `ReasoningGymObservation`.
- Observation fields: `question` (None after a step), `score` (0.0-1.0 from `dataset.score_answer()`), `correct_answer` (revealed after step), `dataset_metadata`.
- `reset(dataset_name, dataset_config, dataset_specs, seed, size)` builds a reasoning_gym dataset (e.g. `leg_counting`, or `composite` with specs) and returns the first question. `reset()` with no params advances to the next question, reusing the dataset.
- `step(ReasoningGymAction(answer=...))` scores the answer against the current entry, returns `score` + `correct_answer`, always `done=True`. Single-step.
- Runs as a server (the env ships a `Dockerfile` + `app.py`; `harness.py` exposes it as MCP tools). On this DevPod, Docker may be unavailable, so plan to run the env via OpenEnv's local/UV provider or in-process server rather than `from_docker_image`.

## The design fork (single-step env vs GRPO group structure)

GRPO needs `num_generations` completions for the *same* question (to compute within-group advantages), grouped by prompt. The env is a sequential iterator scoring one answer at a time. Reconciliation options:

1. **Harvest-to-dataset + env-as-grader (simplest, robust).** Drive the OpenEnv server once at setup to harvest N `(question, correct_answer/score fn)` into a HF dataset (prompt = question). Train via the existing dataset path; `EnvReward` scores each completion against the harvested answer (reasoning_gym `score_answer`, optionally via the live client). Standard GRPO grouping; efficiency rewards work (completion = model tokens). Uses OpenEnv for the env semantics, but not a live per-step loop. Honest fit for a single-step env; keeps the live rollout loop for the multi-turn envs (textarena/coding) where it is actually needed.

2. **Live env in rollout_func (true agentic loop).** `rollout_func` resets the env per question and steps each of the `num_generations` completions against it (step does not advance the entry; only `reset()` does). Needs the dataset and the env iterator kept in lockstep (deterministic seed, batch_size=1, no shuffle) so the prompt the model sees matches the env's current question. Fragile but exercises the live OpenEnv loop end to end.

3. **`environment_factory` instead of `rollout_func`.** TRL's env-driven path handles reset/step/grouping internally and auto-builds the token mask. Cleaner for single-step reasoning_gym; the reason we picked `rollout_func` (efficiency-reward mask control) does not bite here because the completion is entirely model tokens. Trade-off: less explicit control, and the efficiency rewards depend on `completion_ids` reaching reward funcs (needs a quick check).

Recommendation: start with (1) for a robust first agentic-domain result, and reserve the live rollout loop (2) for the first genuinely multi-turn env, where the mask and per-step interaction are essential. Revisit (3) if we want the canonical TRL env loop for reasoning_gym specifically. Decision pending.
