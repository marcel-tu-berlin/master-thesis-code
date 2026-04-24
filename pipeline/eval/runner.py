from __future__ import annotations

import json
import os

import yaml

from eval.metrics import EvalMetrics, SampleResult, compute_metrics
from eval.ood_probes import OODResults, run_ood_probes
from eval.report import generate_report


def run_eval(
    config: dict,
    checkpoint_dir: str,
    domain,
    run_dir: str,
    max_new_tokens: int = 512,
) -> dict:
    """
    Load trained LoRA checkpoint, run all eval splits, write eval_report.json.
    Returns the report dict.
    """
    from unsloth import FastLanguageModel
    from training.registry import get_model_config, LORA_TARGET_MODULES

    print(f"Loading checkpoint: {checkpoint_dir}")
    model_cfg = get_model_config(config["model"]["slug"])
    lora_rank = config["model"].get("lora_r", model_cfg["max_lora_rank"])
    max_seq = config["model"].get("max_seq_length", model_cfg["max_seq_length"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=max_seq,
        load_in_4bit=config["model"].get("load_in_4bit", model_cfg["load_in_4bit"]),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing=False,
    )
    model.load_lora(checkpoint_dir)
    domain.build_chat_template(tokenizer)

    FastLanguageModel.for_inference(model)

    eval_cfg = config.get("eval", {})
    ood_results = run_ood_probes(model, tokenizer, domain, config, eval_cfg, max_new_tokens)

    report = generate_report(config, ood_results, run_dir)

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
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from domains.math.loader import MathDomain
    from domains.coding.loader import CodingDomain

    domain_name = config["training"].get("domain", "math")
    domain = MathDomain() if domain_name == "math" else CodingDomain()

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)
    checkpoint = args.checkpoint or os.path.join(run_dir, "checkpoint-final")

    run_eval(config, checkpoint, domain, run_dir, args.max_new_tokens)


if __name__ == "__main__":
    main()
