import re
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
    # When pad_token == eos_token (the common HF fallback we set above), a
    # naive non-pad count strips the natural-termination EOS too, biasing
    # n_tokens by -1 on every completed row. Detect natural termination by
    # checking whether the first pad position holds an EOS token, and add
    # it back to the count.
    n_tokens_per_row = []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    pad_eq_eos = pad_id == eos_id and eos_id is not None
    for row in completions_ids:
        nz = int((row != pad_id).sum().item())
        if pad_eq_eos and nz < row.shape[0]:
            # First pad slot holds the natural EOS; count it as one generated token.
            if int(row[nz].item()) == eos_id:
                nz += 1
        n_tokens_per_row.append(nz)
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

    # Far-OOD: MMLU. Match case-insensitively so configs may use 'mmlu',
    # 'MMLU', or 'cais/mmlu' interchangeably; warn rather than silently skip
    # so a typo doesn't quietly drop the probe from the report.
    far_name = probes_cfg.get("far")
    if far_name:
        if "mmlu" in str(far_name).lower():
            print(f"  Running far-OOD: MMLU (config: {far_name})")
            results.far_ood = _run_mmlu(
                model, tokenizer, domain, max_new_tokens, n_samples=mmlu_limit, gen_kwargs=gk, batch_size=batch_size
            )
        else:
            print(f"  Warning: unrecognized far-OOD probe {far_name!r}; only MMLU is implemented. Skipping.")

    # Capability floor
    cap_name = probes_cfg.get("capability_floor")
    if cap_name:
        print(f"  Running capability floor: {cap_name}")
        results.capability_floor = _run_capability_floor(
            model, tokenizer, domain, max_new_tokens,
            gen_kwargs=gk, batch_size=batch_size,
            prompts=eval_cfg.get("capability_floor_prompts"),
        )

    return results


def _format_letter_answer(question: str, choices: list[str]) -> str:
    return (
        f"{question}\n"
        + "\n".join(f"{k}. {v}" for k, v in zip("ABCD", choices))
        + "\nAnswer with the letter only."
    )


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _extract_letter(domain, completion_text: str) -> str | None:
    """Pull a single A/B/C/D answer from a completion.

    Strategy:
      1. Prefer the SOLUTION block. Match the first whole-word A/B/C/D inside it.
      2. Fall back to the first whole-word A/B/C/D in the raw completion.
    Whole-word match avoids picking up letters embedded in tokens like 'Asia'
    or model prose like 'Option A is wrong, the answer is B' (still picks A
    from raw completion fallback, but SOLUTION-block hits take priority).
    """
    extracted = domain.extract_answer(completion_text)
    if extracted:
        m = _LETTER_RE.search(extracted)
        if m:
            return m.group(1)
    m = _LETTER_RE.search(completion_text)
    return m.group(1) if m else None


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
    # MMLU answers live behind the SOLUTION tag, after a CoT chain. 256 tokens
    # truncates many runs, producing false negatives. Use the configured
    # budget as a floor; bump default to 512.
    budget = max(max_new_tokens, 512)
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


_DEFAULT_CAPABILITY_PROMPTS = [
    ("What is 2+2?", "4"),
    ("What is the capital of France?", "Paris"),
    ("Name the first planet from the Sun.", "Mercury"),
    ("What is 10 * 10?", "100"),
    ("What color is the sky on a clear day?", "blue"),
]


def _capability_match(expected: str, extracted: str) -> bool:
    """Whole-word, case-insensitive match. Avoids substring false positives
    such as expected='4' matching '14', or expected='blue' matching 'bluefin'.
    Falls back to numeric equality so '4' matches '4.0'.
    """
    exp = expected.strip()
    text = extracted.strip()
    pattern = r"(?<![\w\d.])" + re.escape(exp) + r"(?![\w\d.])"
    if re.search(pattern, text, flags=re.IGNORECASE):
        return True
    try:
        return float(exp) == float(text.split()[0].rstrip(".,"))
    except (ValueError, IndexError):
        return False


def _run_capability_floor(
    model,
    tokenizer,
    domain,
    max_new_tokens: int,
    gen_kwargs: dict | None = None,
    batch_size: int = 8,
    prompts: list | None = None,
) -> EvalMetrics:
    """Simple instruction following — checks the model hasn't catastrophically forgotten.

    `prompts` may be a list of `[question, expected]` pairs from config. Falls
    back to a fixed 5-question default set.
    """
    if prompts:
        pair_iter = [(p[0], p[1]) for p in prompts]
    else:
        pair_iter = _DEFAULT_CAPABILITY_PROMPTS
    gen_kwargs = gen_kwargs or {"do_sample": False}

    prompt_msgs = [[{"role": "user", "content": q}] for q, _ in pair_iter]
    expected = [a for _, a in pair_iter]

    results = []
    budget = max(max_new_tokens, 256)
    for i in range(0, len(prompt_msgs), batch_size):
        chunk = prompt_msgs[i : i + batch_size]
        chunk_expected = expected[i : i + batch_size]
        prompt_texts = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in chunk
        ]
        n_tokens_list, texts = _generate_batch(model, tokenizer, prompt_texts, budget, gen_kwargs)
        for n_tokens, text, exp in zip(n_tokens_list, texts, chunk_expected):
            extracted = domain.extract_answer(text) or text
            correct = _capability_match(exp, extracted)
            results.append(SampleResult(correct=correct, n_tokens=n_tokens))
    return compute_metrics(results)
