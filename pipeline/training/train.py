import argparse
import os
import random
import sys

import numpy as np
import yaml

# Allow running as: python -m training.train
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domains.math.loader import MathDomain
from training.grpo_runner import GRPORunner
from training.mode import select_mode
from training.env_server import build_env_server
from training.rewards import REWARD_REGISTRY
from training.rewards.compose import build_composer
from training.config_schema import validate_config, warn_inert_scalars
from transformers import TrainerCallback, set_seed


class _ComponentMetricsCallback(TrainerCallback):
    """Drain the composer's per-component reward metrics into the trainer log.

    TRL only sees one composed reward function, so it logs a single
    rewards/<composer>/{mean,std}; the individual accuracy/format/length/entropy
    contributions are invisible. on_log fires just before TRL records a log
    entry, so merging the popped metrics into `logs` lands them in
    trainer_state.json's log_history next to reward/kl/loss — visible in the
    training curves and inspectable post-hoc. Purely observational: it never
    touches the composed reward or the advantage math.
    """

    def __init__(self, composer) -> None:
        self.composer = composer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not hasattr(self.composer, "pop_step_metrics"):
            return
        logs.update(self.composer.pop_step_metrics())


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_domain(config: dict):
    if select_mode(config) == "agentic":
        env = config["training"]["env"]
        if env == "reasoning_gym":
            from domains.reasoning_gym import ReasoningGymDomain
            return ReasoningGymDomain()
        raise NotImplementedError(f"Env: {env}")
    name = config["training"].get("domain", "math")
    if name == "math":
        return MathDomain()
    raise NotImplementedError(f"Domain: {name}")


def build_reward_components(config: dict, domain, runner: GRPORunner) -> list:
    """Build (reward_fn, weight) pairs from config using REWARD_REGISTRY."""
    rewards_cfg = config.get("rewards", {}) or {}
    training_cfg = config.get("training", {}) or {}

    method = rewards_cfg.get("compose_method", "advantage_weighted")
    for w in warn_inert_scalars(rewards_cfg, method):
        print(f"⚠  {w}")

    components = []

    for key, (_reg_enabled, default_weight, builder) in REWARD_REGISTRY.items():
        cfg = rewards_cfg.get(key) or {}
        if not cfg.get("enabled", _reg_enabled):
            continue
        weight = float(cfg.get("weight", default_weight))
        components.append((builder(domain, runner, training_cfg, cfg), weight))

    return components


def apply_smoke_overrides(config: dict) -> dict:
    """Patch config for fast smoke testing: 3 steps, 2 rollouts, short seq.

    Sets `_smoke=True` so downstream eval also caps to 10 samples per split.
    """
    config.setdefault("model", {})
    config.setdefault("training", {})
    config["model"]["max_seq_length"] = 512
    config["training"]["max_steps"] = 3
    config["training"]["save_steps"] = 3
    config["training"]["n_rollouts"] = 2
    config["training"]["dataset_size_limit"] = 64
    config["_smoke"] = True
    print("⚠  Smoke mode: max_steps=3, n_rollouts=2, max_seq_length=512, dataset_size_limit=64, eval=10/split")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval", action="store_true", help="Run eval after training")
    parser.add_argument("--smoke", action="store_true", help="Override config for fast smoke test (3 steps, 2 rollouts, 512 seq)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run directory")
    parser.add_argument("--vllm", action="store_true", help="Route GRPO rollouts through vLLM fast inference")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        apply_smoke_overrides(config)
    if args.vllm:
        config.setdefault("model", {})
        config["model"]["use_vllm"] = True
        config["model"].setdefault("gpu_memory_utilization", 0.6)
        config["model"]["enforce_eager"] = True
        print("⚠  vLLM fast inference ON (--vllm): gpu_memory_utilization=0.6, enforce_eager=True")
    validate_config(config)
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)  # covers torch + cuda RNGs

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)
    # Refuse to clobber an existing run unless --overwrite is given. Without
    # this guard, a re-invocation with the same experiment_id silently
    # overwrites the prior frozen config and (later) trampling checkpoints.
    existing_final = os.path.join(run_dir, "checkpoint-final")
    if os.path.isdir(existing_final) and not args.overwrite:
        raise FileExistsError(
            f"Run directory {run_dir!r} already has checkpoint-final/. "
            "Pass --overwrite to replace, or change experiment_id."
        )
    os.makedirs(run_dir, exist_ok=True)

    # Persist config alongside run artifacts. Strip the runtime-only `_smoke`
    # marker so re-running eval against the frozen config does not silently
    # cap each split to 10 samples.
    frozen = {k: v for k, v in config.items() if k != "_smoke"}
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(frozen, f)

    domain = build_domain(config)
    runner = GRPORunner(config)
    mode = select_mode(config)

    # Build composed reward function (shared by both modes; the enabled set
    # differs per config — agentic configs enable env_reward + token_length).
    components = build_reward_components(config, domain, runner)
    if not components:
        raise ValueError("No reward components enabled. Check config rewards section.")

    method = config.get("rewards", {}).get("compose_method", "advantage_weighted")
    reward_fn = build_composer(components, method)

    callbacks = []
    # The callback holds the same composer instance passed as the reward fn, so
    # it drains the very buffer the trainer's reward calls populate (T2.1).
    if hasattr(reward_fn, "pop_step_metrics"):
        callbacks.append(_ComponentMetricsCallback(reward_fn))
    callbacks = callbacks or None

    print(f"Experiment: {exp_id}  (mode={mode})")
    print(f"Reward components: {[type(fn).__name__ for fn, _ in components]}")
    print(f"Compose method: {method}")

    checkpoint_dir = os.path.join(run_dir, "checkpoint-final")

    if mode == "agentic":
        # Native tool-calling template (NOT the reasoning-tag one). Each seed-row
        # is a distinct reasoning_gym question; the runner owns the env-server
        # subprocess and builds the TRL environment_factory against its base_url.
        env_config = config["training"].get("env_config", {}) or {}
        n_prompts = int(env_config.get("size", 500))
        dataset = domain.build_seed_dataset(env_config, n=n_prompts, seed_base=seed)
        server = build_env_server(config, domain, python=sys.executable)
        make_factory = lambda base_url: domain.make_env_factory(base_url, env_config)  # noqa: E731
        print(f"Agentic env: {config['training']['env']}  seed-rows: {len(dataset)}  "
              f"server: {server.base_url} (max_concurrent={server.max_concurrent})")
        runner.train(dataset, reward_fn, output_dir=run_dir, callbacks=callbacks,
                     server=server, make_factory=make_factory)
    else:
        # Dataset mode: reasoning-tag template + a HF dataset. Truncate FIRST so
        # the (slow) prompt-length filter runs only over the kept slice.
        domain.build_chat_template(runner.tokenizer)
        ds_cfg = config["training"].get("dataset", "openai/gsm8k")
        dataset = domain.load_dataset(ds_cfg, split=config["training"].get("split", "train"))

        size_limit = config["training"].get("dataset_size_limit")
        if size_limit is not None and len(dataset) > size_limit:
            size_limit = int(size_limit)
            # Shuffle before truncation so the subset is representative across the
            # source distribution. DAPO and Hendrycks MATH are clustered by
            # difficulty/category — range(0, N) would exclude later categories.
            dataset = dataset.shuffle(seed=seed).select(range(size_limit))
            print(f"Dataset shuffled and truncated to {size_limit} samples (dataset_size_limit, seed={seed})")

        if hasattr(domain, "filter_by_prompt_length"):
            dataset = domain.filter_by_prompt_length(
                dataset, runner.tokenizer, quantile=config["training"].get("prompt_length_quantile", 0.9)
            )
        print(f"Dataset size: {len(dataset)}")
        runner.train(dataset, reward_fn, output_dir=run_dir, callbacks=callbacks)

    runner.save_lora(checkpoint_dir)
    if config.get("_smoke"):
        open(os.path.join(checkpoint_dir, ".smoke"), "w").close()

    if args.eval:
        if mode == "agentic":
            from eval.agentic_eval import run_agentic_eval
            run_agentic_eval(config, checkpoint_dir, domain, run_dir)
        else:
            from eval.runner import run_eval
            run_eval(config, checkpoint_dir, domain, run_dir)


if __name__ == "__main__":
    main()
