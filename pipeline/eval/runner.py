"""Agentic eval entry point (`python -m eval.runner --config ...`).

Thin wrapper: loads the config, resolves the checkpoint, and dispatches to
eval.agentic_eval.run_agentic_eval (held-out OpenEnv episodes). Dataset-mode
eval was removed with the agentic-only migration.
"""
import os

import yaml


def smoke_conflict(run_dir: str) -> str | None:
    """Error message if a smoke eval would overwrite real results, else None.

    A --smoke eval writes 4 episodes per split into the same run dir as a real
    one, so pointing it at a harvested arm destroys `eval_report.json`,
    `eval_report.md` and every `episodes_*.jsonl` - the trajectory records the
    off-target panel is answered from, which no snapshot of the report alone
    brings back. Same hazard as `--base-model` against a trained run, same
    answer: refuse, and let a throwaway experiment_id take the write.
    """
    from training.batch import _is_real_report
    if not _is_real_report(os.path.join(run_dir, "eval_report.json")):
        return None
    return (
        f"--smoke would overwrite the real eval results in {run_dir!r} "
        "(report and episodes_*.jsonl). Copy the config with a throwaway "
        "experiment_id to smoke-test against this checkpoint."
    )


def base_model_conflict(run_dir: str) -> str | None:
    """Error message if E0 results would overwrite a trained run, else None.

    run_dir keys on experiment_id alone, so `--base-model` against a trained
    arm's config used to overwrite its eval_report.json and episodes_*.jsonl
    with untrained-model numbers - silently, with no backup. E0 gets its own
    config and its own experiment_id (the e0 / e0b pattern).
    """
    if not os.path.isdir(os.path.join(run_dir, "checkpoint-final")):
        return None
    return (
        f"--base-model would write E0 results into {run_dir!r}, which holds a "
        "trained checkpoint. Use a dedicated E0 config with its own "
        "experiment_id (see configs/e0-*.yaml)."
    )


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--base-model", action="store_true",
                        help="Evaluate the base model with no LoRA adapter (E0)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override eval.max_new_tokens (default: the training completion budget)")
    parser.add_argument("--smoke", action="store_true", help="Limit eval to 4 episodes per split")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Same guard training gets. An eval-only invocation used to skip validation
    # entirely, so a typo'd key here was caught only if the config had also been
    # through `training.train` at some point.
    from training.config_schema import validate_config
    validate_config(config)

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)

    if args.smoke:
        conflict = smoke_conflict(run_dir)
        if conflict:
            parser.error(conflict)
        config["_smoke"] = True
        print("Smoke mode: eval limited to 4 episodes per split")

    if args.base_model:
        if args.checkpoint:
            parser.error("--base-model and --checkpoint are mutually exclusive")
        conflict = base_model_conflict(run_dir)
        if conflict:
            parser.error(conflict)
        checkpoint = None
    else:
        checkpoint = args.checkpoint or os.path.join(run_dir, "checkpoint-final")

    if args.max_new_tokens is not None:
        config.setdefault("eval", {})["max_new_tokens"] = args.max_new_tokens

    from domains import build_domain
    domain = build_domain(config)

    from eval.agentic_eval import run_agentic_eval
    run_agentic_eval(config, checkpoint, domain, run_dir)


if __name__ == "__main__":
    main()
