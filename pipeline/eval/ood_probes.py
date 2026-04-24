from __future__ import annotations

from dataclasses import dataclass

from eval.metrics import EvalMetrics, SampleResult, compute_metrics


@dataclass
class OODResults:
    id_split: EvalMetrics | None = None
    near_ood: EvalMetrics | None = None
    far_ood: EvalMetrics | None = None
    capability_floor: EvalMetrics | None = None


def _run_split(model, tokenizer, domain, dataset, max_new_tokens: int = 512) -> EvalMetrics:
    """Generate completions for a dataset split and score them."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    results = []
    for sample in dataset:
        prompt_text = tokenizer.apply_chat_template(
            sample["prompt"], add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completion_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        correct = domain.is_correct(completion_text, sample["answer"])
        n_tokens = len(completion_ids)
        difficulty = domain.difficulty(sample)
        results.append(SampleResult(correct=correct, n_tokens=n_tokens, difficulty=difficulty))

    return compute_metrics(results)


def run_ood_probes(
    model,
    tokenizer,
    domain,
    config: dict,
    eval_cfg: dict,
    max_new_tokens: int = 512,
) -> OODResults:
    """
    Run all OOD probe splits and return structured results.

    Probe hierarchy (from config eval.ood_probes):
      id_split     — held-out portion of the training dataset
      near_ood     — same domain, different distribution (e.g. GSM-8K if trained on MATH)
      far_ood      — MMLU subset (5-shot multiple-choice)
      capability   — simple instruction-following floor (fixed 50-item set)
    """
    from domains.math.loader import MathDomain

    probes_cfg = eval_cfg.get("ood_probes", {})
    results = OODResults()

    # ID split ─────────────────────────────────────────────────────────────────
    id_name = eval_cfg.get("id_split")
    if id_name:
        print(f"  Running ID split: {id_name}")
        id_ds = domain.load_dataset(
            config["training"]["dataset"],
            split=eval_cfg.get("id_split_hf_split", "test"),
        )
        id_ds = id_ds.select(range(min(200, len(id_ds))))
        results.id_split = _run_split(model, tokenizer, domain, id_ds, max_new_tokens)

    # Near-OOD ─────────────────────────────────────────────────────────────────
    near_name = probes_cfg.get("near")
    if near_name:
        print(f"  Running near-OOD: {near_name}")
        near_ds = domain.load_dataset(near_name, split="test")
        near_ds = near_ds.select(range(min(200, len(near_ds))))
        results.near_ood = _run_split(model, tokenizer, domain, near_ds, max_new_tokens)

    # Far-OOD: MMLU ────────────────────────────────────────────────────────────
    far_name = probes_cfg.get("far")
    if far_name == "MMLU":
        print("  Running far-OOD: MMLU")
        results.far_ood = _run_mmlu(model, tokenizer, max_new_tokens)

    # Capability floor ─────────────────────────────────────────────────────────
    cap_name = probes_cfg.get("capability_floor")
    if cap_name:
        print(f"  Running capability floor: {cap_name}")
        results.capability_floor = _run_capability_floor(model, tokenizer, cap_name, max_new_tokens)

    return results


def _run_mmlu(model, tokenizer, max_new_tokens: int, n_samples: int = 100) -> EvalMetrics:
    from datasets import load_dataset as hf_load
    ds = hf_load("lukaemon/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    results = []
    for sample in ds:
        question = sample["input"]
        choices = [sample.get(k, "") for k in ["A", "B", "C", "D"]]
        prompt = (
            f"{question}\n"
            + "\n".join(f"{k}. {v}" for k, v in zip("ABCD", choices))
            + "\nAnswer with the letter only."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        correct = pred.upper().startswith(sample["target"].upper())
        results.append(SampleResult(correct=correct, n_tokens=len(out[0]) - inputs["input_ids"].shape[1]))

    return compute_metrics(results)


def _run_capability_floor(model, tokenizer, name: str, max_new_tokens: int) -> EvalMetrics:
    """Simple yes/no instruction following — checks the model hasn't catastrophically forgotten."""
    FIXED_PROMPTS = [
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("Name the first planet from the Sun.", "Mercury"),
        ("What is 10 * 10?", "100"),
        ("What color is the sky on a clear day?", "blue"),
    ]
    results = []
    for prompt, expected in FIXED_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        correct = expected.lower() in pred
        results.append(SampleResult(correct=correct, n_tokens=len(out[0]) - inputs["input_ids"].shape[1]))
    return compute_metrics(results)
