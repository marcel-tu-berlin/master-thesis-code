"""Agentic eval entry point (`python -m eval.runner --config ...`).

Thin wrapper: loads the config, resolves the checkpoint, and dispatches to
eval.agentic_eval.run_agentic_eval (held-out OpenEnv episodes). Dataset-mode
eval was removed with the agentic-only migration.
"""
import os

import yaml


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override eval.max_new_tokens (default: the training completion budget)")
    parser.add_argument("--smoke", action="store_true", help="Limit eval to 10 episodes")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        config["_smoke"] = True
        print("Smoke mode: eval limited to 10 episodes")

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)
    checkpoint = args.checkpoint or os.path.join(run_dir, "checkpoint-final")

    if args.max_new_tokens is not None:
        config.setdefault("eval", {})["max_new_tokens"] = args.max_new_tokens

    env = config["training"].get("env")
    if env != "reasoning_gym":
        raise NotImplementedError(f"Env: {env!r} (only 'reasoning_gym' is implemented)")

    from domains.reasoning_gym import ReasoningGymDomain
    from eval.agentic_eval import run_agentic_eval
    run_agentic_eval(config, checkpoint, ReasoningGymDomain(), run_dir)


if __name__ == "__main__":
    main()
