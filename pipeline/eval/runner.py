import json
import os

import yaml

from eval.ood_probes import run_ood_probes
from eval.report import generate_report


def run_eval(
    config: dict,
    checkpoint_dir: str,
    domain,
    run_dir: str,
    max_new_tokens: int = 512,
) -> dict:
    """max_new_tokens: default budget; eval.max_new_tokens in config overrides."""
    """
    Load trained LoRA checkpoint, run all eval splits, write eval_report.json.
    Returns the report dict.
    """
    from unsloth import FastLanguageModel
    from training.registry import get_model_config

    print(f"Loading checkpoint: {checkpoint_dir}")
    model_cfg = get_model_config(config["model"]["slug"])
    max_seq = config["model"].get("max_seq_length", model_cfg["max_seq_length"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=max_seq,
        load_in_4bit=config["model"].get("load_in_4bit", model_cfg["load_in_4bit"]),
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    domain.build_chat_template(tokenizer)

    FastLanguageModel.for_inference(model)

    eval_cfg = config.get("eval", {})
    smoke = config.get("_smoke", False)
    # Config wins over the run_eval default; CLI override (--max_new_tokens
    # not equal to default) is handled by the caller in main().
    cfg_budget = eval_cfg.get("max_new_tokens")
    if cfg_budget is not None:
        max_new_tokens = int(cfg_budget)
    ood_results = run_ood_probes(model, tokenizer, domain, config, eval_cfg, max_new_tokens, smoke=smoke)

    report = generate_report(config, ood_results, run_dir)

    try:
        from eval.plots import plot_all
        plot_all(run_dir, ood_results, report, run_dir)
    except Exception as exc:
        print(f"Warning: plot generation failed: {exc}")

    report_path = os.path.join(run_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Eval report written to {report_path}")

    return report


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override eval.max_new_tokens from the config (default: config value or 512)")
    parser.add_argument("--smoke", action="store_true", help="Limit eval to 10 samples per split for quick sanity checks")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        config["_smoke"] = True
        print("⚠ Smoke mode: eval limited to 10 samples per split")

    from domains.math.loader import MathDomain
    from domains.coding.loader import CodingDomain

    domain_name = config["training"].get("domain", "math")
    domain = MathDomain() if domain_name == "math" else CodingDomain()

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)
    checkpoint = args.checkpoint or os.path.join(run_dir, "checkpoint-final")

    # CLI > config > 512 default. CLI explicit overrides config.
    if args.max_new_tokens is not None:
        config.setdefault("eval", {})["max_new_tokens"] = args.max_new_tokens
    run_eval(config, checkpoint, domain, run_dir)


if __name__ == "__main__":
    main()
