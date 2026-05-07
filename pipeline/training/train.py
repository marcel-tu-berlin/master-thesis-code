import argparse
import os
import random
import sys

import numpy as np
import yaml

# Allow running as: python -m training.train
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domains.math.loader import MathDomain
from domains.coding.loader import CodingDomain
from training.grpo_runner import GRPORunner
from training.rewards import REWARD_REGISTRY
from training.rewards.compose import build_composer
from training.config_schema import validate_config
from transformers import TrainerCallback, set_seed


class _RewardStepCallback(TrainerCallback):
    """TRL TrainerCallback that advances reward schedulers after each step."""

    def __init__(self, step_fns: list) -> None:
        self.step_fns = step_fns

    def on_step_end(self, args, state, control, **kwargs):
        for fn in self.step_fns:
            fn()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_domain(config: dict):
    name = config["training"].get("domain", "math")
    if name == "math":
        return MathDomain()
    if name == "coding":
        return CodingDomain()
    raise NotImplementedError(f"Domain: {name}")


def build_reward_components(config: dict, domain, runner: GRPORunner) -> list:
    """Build (reward_fn, weight) pairs from config using REWARD_REGISTRY."""
    rewards_cfg = config.get("rewards", {}) or {}
    training_cfg = config.get("training", {}) or {}
    components = []

    for key, (default_enabled, default_weight, builder) in REWARD_REGISTRY.items():
        cfg = rewards_cfg.get(key) or {}
        if not cfg.get("enabled", default_enabled):
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
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        apply_smoke_overrides(config)
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
    existing_config = os.path.join(run_dir, "config.yaml")
    existing_final = os.path.join(run_dir, "checkpoint-final")
    if (os.path.exists(existing_config) or os.path.isdir(existing_final)) and not args.overwrite:
        raise FileExistsError(
            f"Run directory {run_dir!r} already contains artifacts. "
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

    # Apply domain chat template to tokenizer
    domain.build_chat_template(runner.tokenizer)

    # Load + preprocess dataset.
    # Truncate FIRST so the (slow) prompt-length filter runs only over the kept slice —
    # matters for smoke runs against datasets like DAPO-17k.
    ds_cfg = config["training"].get("dataset", "openai/gsm8k")
    dataset = domain.load_dataset(ds_cfg, split=config["training"].get("split", "train"))

    size_limit = config["training"].get("dataset_size_limit")
    if size_limit is not None and len(dataset) > size_limit:
        size_limit = int(size_limit)
        # Shuffle before truncation so subset is representative across the
        # source distribution. DAPO and Hendrycks MATH are clustered by
        # difficulty/category — taking range(0, N) would systematically
        # exclude later categories. Seeded for reproducibility.
        dataset = dataset.shuffle(seed=seed).select(range(size_limit))
        print(f"Dataset shuffled and truncated to {size_limit} samples (dataset_size_limit, seed={seed})")

    if hasattr(domain, "filter_by_prompt_length"):
        dataset = domain.filter_by_prompt_length(
            dataset, runner.tokenizer, quantile=config["training"].get("prompt_length_quantile", 0.9)
        )

    # Build composed reward function
    components = build_reward_components(config, domain, runner)
    if not components:
        raise ValueError("No reward components enabled. Check config rewards section.")

    method = config.get("rewards", {}).get("compose_method", "advantage_weighted")
    reward_fn = build_composer(components, method)

    step_fns = [fn.step for fn, _ in components if hasattr(fn, "step") and callable(fn.step)]
    callbacks = [_RewardStepCallback(step_fns)] if step_fns else None

    print(f"Experiment: {exp_id}")
    print(f"Reward components: {[type(fn).__name__ for fn, _ in components]}")
    print(f"Compose method: {method}")
    print(f"Dataset size: {len(dataset)}")

    checkpoint_dir = os.path.join(run_dir, "checkpoint-final")
    runner.train(dataset, reward_fn, output_dir=run_dir, callbacks=callbacks)
    runner.save_lora(checkpoint_dir)

    if args.eval:
        from eval.runner import run_eval
        run_eval(config, checkpoint_dir, domain, run_dir)


if __name__ == "__main__":
    main()
