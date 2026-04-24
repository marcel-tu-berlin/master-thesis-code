from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

# Allow running as: python -m training.train
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domains.math.loader import MathDomain
from domains.coding.loader import CodingDomain
from training.grpo_runner import GRPORunner
from training.rewards.accuracy import AnswerReward, NumericReward
from training.rewards.compose import build_composer
from training.rewards.effort_proxy import EffortProxyReward
from training.rewards.format import FormatApproxReward, FormatExactReward
from training.rewards.token_entropy import TokenEntropyReward
from training.rewards.token_length import TokenLengthReward


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
    """Build (reward_fn, weight) pairs from config."""
    rewards_cfg = config.get("rewards", {})
    components = []

    def enabled(key: str) -> bool:
        return rewards_cfg.get(key, {}).get("enabled", True)

    def weight(key: str, default: float = 1.0) -> float:
        return rewards_cfg.get(key, {}).get("weight", default)

    if enabled("format_exact"):
        components.append((FormatExactReward(domain), weight("format_exact", 1.0)))

    if enabled("format_approx"):
        components.append((FormatApproxReward(domain), weight("format_approx", 0.5)))

    if enabled("accuracy"):
        components.append((AnswerReward(domain), weight("accuracy", 1.0)))

    if enabled("numeric"):
        components.append((NumericReward(domain), weight("numeric", 1.0)))

    if enabled("token_length"):
        cfg = rewards_cfg.get("token_length", {})
        components.append((
            TokenLengthReward(
                runner.tokenizer,
                alpha=cfg.get("alpha", 0.001),
                mode=cfg.get("schedule", "constant"),
                total_steps=config["training"].get("max_steps", 500),
            ),
            weight("token_length", 1.0),
        ))

    if enabled("token_entropy"):
        cfg = rewards_cfg.get("token_entropy", {})
        components.append((
            TokenEntropyReward(
                runner.model,
                runner.tokenizer,
                reward_scale=cfg.get("reward_scale", 0.1),
                fork_mask_top_pct=cfg.get("fork_mask_top_pct", 0.0),
            ),
            weight("token_entropy", 1.0),
        ))

    if enabled("effort_proxy"):
        cfg = rewards_cfg.get("effort_proxy", {})
        components.append((
            EffortProxyReward(
                runner.tokenizer,
                metric=cfg.get("metric", "token_count"),
                alpha=cfg.get("alpha", 0.001),
            ),
            weight("effort_proxy", 1.0),
        ))

    return components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval", action="store_true", help="Run eval after training")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)
    os.makedirs(run_dir, exist_ok=True)

    # Persist config alongside run artifacts
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    domain = build_domain(config)
    runner = GRPORunner(config)

    # Apply domain chat template to tokenizer
    domain.build_chat_template(runner.tokenizer)

    # Load + preprocess dataset
    ds_cfg = config["training"].get("dataset", "openai/gsm8k")
    dataset = domain.load_dataset(ds_cfg, split=config["training"].get("split", "train"))

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

    print(f"Experiment: {exp_id}")
    print(f"Reward components: {[type(fn).__name__ for fn, _ in components]}")
    print(f"Compose method: {method}")
    print(f"Dataset size: {len(dataset)}")

    checkpoint_dir = os.path.join(run_dir, "checkpoint-final")
    runner.train(dataset, reward_fn, output_dir=run_dir)
    runner.save_lora(checkpoint_dir)

    if args.eval:
        from eval.runner import run_eval
        run_eval(config, checkpoint_dir, domain, run_dir)


if __name__ == "__main__":
    main()
