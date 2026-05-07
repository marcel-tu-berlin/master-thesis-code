from dataclasses import dataclass

from eval.metrics import EvalMetrics, SampleResult, compute_metrics


@dataclass
class OODResults:
    id_split: EvalMetrics | None = None
    near_ood: EvalMetrics | None = None
    far_ood: EvalMetrics | None = None
    capability_floor: EvalMetrics | None = None


def _gen_kwargs(eval_cfg: dict) -> dict:
    """Build generation kwargs. Drop `temperature` entirely when not sampling
    so HF doesn't warn about an unused argument.
    """
    do_sample = eval_cfg.get("do_sample", False)
    if do_sample:
        return {"do_sample": True, "temperature": float(eval_cfg.get("temperature", 1.0))}
    return {"do_sample": False}


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def _generate_batch(model, tokenizer, prompt_texts: list[str], max_new_tokens: int, gen_kwargs: dict):
    """Tokenize + generate a batch of prompts. Returns (completion_token_ids_per_row, completion_text_per_row).
    Uses left padding so we can slice out continuations cleanly.
    """
    _ensure_pad_token(tokenizer)
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=False).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )
    finally:
        tokenizer.padding_side = prev_padding_side

    prompt_len = inputs["input_ids"].shape[1]
    completions_ids = out[:, prompt_len:]
    texts = tokenizer.batch_decode(completions_ids, skip_special_tokens=True)
    # n_tokens excludes pad on the right (special tokens already stripped via skip_special_tokens for text;
    # for n_tokens we want generated length excluding pad).
    n_tokens_per_row = []
    pad_id = tokenizer.pad_token_id
    for row in completions_ids:
        nz = (row != pad_id).sum().item()
        n_tokens_per_row.append(int(nz))
    return n_tokens_per_row, texts


def _run_split(
    model,
    tokenizer,
    domain,
    dataset,
    max_new_tokens: int = 512,
    gen_kwargs: dict | None = None,
    batch_size: int = 8,
) -> EvalMetrics:
    """Generate completions for a dataset split (batched) and score them."""
    gen_kwargs = gen_kwargs or {"do_sample": False}
    results = []

    rows = list(dataset)
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(s["prompt"], add_generation_prompt=True, tokenize=False)
            for s in chunk
        ]
        n_tokens_list, texts = _generate_batch(model, tokenizer, prompt_texts, max_new_tokens, gen_kwargs)
        for sample, n_tokens, completion_text in zip(chunk, n_tokens_list, texts):
            correct = domain.is_correct(completion_text, sample["answer"])
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
    smoke: bool = False,
) -> OODResults:
    """
    Run all OOD probe splits and return structured results.

    Probe hierarchy (from config eval.ood_probes):
    id_split — held-out portion of the training dataset
    near_ood — same domain, different distribution (e.g. GSM-8K if trained on MATH)
    far_ood — MMLU subset (5-shot multiple-choice). NOTE: hardcoded to MMLU; add
              a probe registry if more far-OOD datasets are needed.
    capability — simple instruction-following floor (fixed 5-item set)
    """
    probes_cfg = eval_cfg.get("ood_probes", {})
    gk = _gen_kwargs(eval_cfg)
    batch_size = int(eval_cfg.get("batch_size", 8))
    results = OODResults()

    id_limit = 10 if smoke else eval_cfg.get("id_split_limit", 200)
    near_limit = 10 if smoke else eval_cfg.get("near_ood_limit", 200)
    mmlu_limit = 10 if smoke else eval_cfg.get("far_ood_limit", 100)

    # ID split
    id_name = eval_cfg.get("id_split")
    if id_name:
        print(f"  Running ID split: {id_name}")
        id_ds = domain.load_dataset(
            config["training"]["dataset"],
            split=eval_cfg.get("id_split_hf_split", "test"),
        )
        id_ds = id_ds.select(range(min(id_limit, len(id_ds))))
        results.id_split = _run_split(model, tokenizer, domain, id_ds, max_new_tokens, gen_kwargs=gk, batch_size=batch_size)

    # Near-OOD
    near_name = probes_cfg.get("near")
    if near_name:
        print(f"  Running near-OOD: {near_name}")
        near_ds = domain.load_dataset(near_name, split="test")
        near_ds = near_ds.select(range(min(near_limit, len(near_ds))))
        results.near_ood = _run_split(model, tokenizer, domain, near_ds, max_new_tokens, gen_kwargs=gk, batch_size=batch_size)

    # Far-OOD: MMLU
    far_name = probes_cfg.get("far")
    if far_name == "MMLU":
        print("  Running far-OOD: MMLU")
        results.far_ood = _run_mmlu(
            model, tokenizer, domain, max_new_tokens, n_samples=mmlu_limit, gen_kwargs=gk, batch_size=batch_size
        )

    # Capability floor
    cap_name = probes_cfg.get("capability_floor")
    if cap_name:
        print(f"  Running capability floor: {cap_name}")
        results.capability_floor = _run_capability_floor(
            model, tokenizer, domain, max_new_tokens, gen_kwargs=gk, batch_size=batch_size
        )

    return results


def _format_letter_answer(question: str, choices: list[str]) -> str:
    return (
        f"{question}\n"
        + "\n".join(f"{k}. {v}" for k, v in zip("ABCD", choices))
        + "\nAnswer with the letter only."
    )


def _extract_letter(domain, completion_text: str) -> str | None:
    """Pull a single A/B/C/D answer from a completion. Prefers the SOLUTION
    block emitted by the trained model; falls back to first uppercase letter
    in the text."""
    extracted = domain.extract_answer(completion_text)
    candidate = (extracted or completion_text).strip()
    for ch in candidate:
        if ch.upper() in "ABCD":
            return ch.upper()
    return None


def _run_mmlu(
    model,
    tokenizer,
    domain,
    max_new_tokens: int,
    n_samples: int = 100,
    gen_kwargs: dict | None = None,
    batch_size: int = 8,
) -> EvalMetrics:
    from datasets import load_dataset as hf_load
    ds = hf_load("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    gen_kwargs = gen_kwargs or {"do_sample": False}

    rows = list(ds)
    prompts = [
        [{"role": "user", "content": _format_letter_answer(s["question"], s["choices"])}]
        for s in rows
    ]
    expected = ["ABCD"[s["answer"]] for s in rows]

    results = []
    # MMLU answers may live behind the SOLUTION tag, so allow enough budget.
    budget = max(max_new_tokens, 256)
    for i in range(0, len(rows), batch_size):
        chunk_prompts = prompts[i : i + batch_size]
        chunk_expected = expected[i : i + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in chunk_prompts
        ]
        n_tokens_list, texts = _generate_batch(model, tokenizer, prompt_texts, budget, gen_kwargs)
        for n_tokens, text, ans in zip(n_tokens_list, texts, chunk_expected):
            pred = _extract_letter(domain, text)
            correct = pred == ans
            results.append(SampleResult(correct=correct, n_tokens=n_tokens))

    return compute_metrics(results)


def _run_capability_floor(
    model,
    tokenizer,
    domain,
    max_new_tokens: int,
    gen_kwargs: dict | None = None,
    batch_size: int = 8,
) -> EvalMetrics:
    """Simple instruction following — checks the model hasn't catastrophically forgotten."""
    FIXED_PROMPTS = [
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("Name the first planet from the Sun.", "Mercury"),
        ("What is 10 * 10?", "100"),
        ("What color is the sky on a clear day?", "blue"),
    ]
    gen_kwargs = gen_kwargs or {"do_sample": False}

    prompts = [[{"role": "user", "content": q}] for q, _ in FIXED_PROMPTS]
    expected = [a for _, a in FIXED_PROMPTS]

    results = []
    budget = max(max_new_tokens, 256)
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        chunk_expected = expected[i : i + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in chunk
        ]
        n_tokens_list, texts = _generate_batch(model, tokenizer, prompt_texts, budget, gen_kwargs)
        for n_tokens, text, exp in zip(n_tokens_list, texts, chunk_expected):
            extracted = domain.extract_answer(text) or text
            correct = exp.lower() in extracted.lower()
            results.append(SampleResult(correct=correct, n_tokens=n_tokens))
    return compute_metrics(results)
