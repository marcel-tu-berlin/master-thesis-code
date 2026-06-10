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
    baseline: bool = False,
) -> dict:
    """max_new_tokens: default budget; eval.max_new_tokens in config overrides.

    When baseline=True: skip LoRA adapter loading and run probes against the
    raw base model. Artefacts are written to runs/<exp>/baseline/ instead of
    the run root so a trained assessment can coexist with its before-finetune
    counterpart.

    A crashed or partial eval always leaves a stub eval_report.json behind
    (status:'error') so the run still appears in auto-compare instead of
    silently vanishing; the exception is then re-raised so the subprocess
    still exits non-zero and the batch marks the phase failed.
    """
    # Computed up front so the except below can drop a stub in the right place
    # (run root, or the baseline/ subdir) regardless of where the failure hit.
    output_dir = os.path.join(run_dir, "baseline") if baseline else run_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        from unsloth import FastLanguageModel
        from training.registry import get_model_config

        model_cfg = get_model_config(config["model"]["slug"])
        max_seq = config["model"].get("max_seq_length", model_cfg["max_seq_length"])

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["model_name"],
            max_seq_length=max_seq,
            load_in_4bit=config["model"].get("load_in_4bit", model_cfg["load_in_4bit"]),
        )
        if baseline:
            print(f"Baseline mode: assessing base model {model_cfg['model_name']} (no LoRA)")
        else:
            print(f"Loading checkpoint: {checkpoint_dir}")
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
        ood_results = run_ood_probes(model, tokenizer, domain, config, eval_cfg, max_new_tokens, smoke=smoke, lenient=baseline)

        # Baseline reports are themselves "before" measurements — comparing them
        # against another e0-* run would be nonsense, so suppress sibling search.
        report = generate_report(
            config,
            ood_results,
            run_dir,
            output_dir=output_dir,
            skip_baseline_compare=baseline,
        )

        try:
            from eval.plots import plot_all
            # plot_training_curves still reads from run_dir (no trainer state in
            # baseline mode — that plot bails silently).
            plot_all(run_dir, ood_results, report, output_dir)
        except Exception as exc:
            # Per-plot failures inside plot_all are already named by plots.py;
            # this outer catch only fires on top-level failures (import, mkdir,
            # etc.), so include the exception type to make those easier to triage.
            print(f"Warning: plot_all failed before per-plot dispatch: {type(exc).__name__}: {exc}")

        report_path = os.path.join(output_dir, "eval_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {report_path}")

        return report
    except Exception as exc:
        _write_stub_report(config, output_dir, status="error", error=f"{type(exc).__name__}: {exc}")
        raise


def _write_stub_report(config: dict, output_dir: str, status: str, error: str | None = None) -> None:
    """Write a minimal eval_report.json so a failed/partial/skipped eval still
    leaves a discoverable artifact. `status` is 'error' (crashed) or 'skipped'
    (never ran). Mirrors the key shape of a real report so consumers can read
    experiment_id / model_slug / seed without special-casing.
    """
    stub = {
        "experiment_id": config.get("experiment_id"),
        "model_slug": (config.get("model") or {}).get("slug"),
        "seed": config.get("seed", 42),
        "compose_method": (config.get("rewards") or {}).get("compose_method", "advantage_weighted"),
        "status": status,
        "results": {},
    }
    if error:
        stub["error"] = error
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_report.json")
    with open(path, "w") as f:
        json.dump(stub, f, indent=2)
    print(f"⚠ Wrote stub eval_report.json (status={status}) to {path}")


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
    parser.add_argument("--baseline", action="store_true",
                        help="Skip the LoRA adapter and assess the base model. Writes to runs/<exp>/baseline/.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        config["_smoke"] = True
        print("⚠ Smoke mode: eval limited to 10 samples per split")

    from domains.math.loader import MathDomain

    domain_name = config["training"].get("domain", "math")
    if domain_name != "math":
        raise NotImplementedError(f"Domain: {domain_name}")
    domain = MathDomain()

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)
    checkpoint = args.checkpoint or os.path.join(run_dir, "checkpoint-final")

    # CLI > config > 512 default. CLI explicit overrides config.
    if args.max_new_tokens is not None:
        config.setdefault("eval", {})["max_new_tokens"] = args.max_new_tokens
    run_eval(config, checkpoint, domain, run_dir, baseline=args.baseline)


if __name__ == "__main__":
    main()
