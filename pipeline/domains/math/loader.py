import re

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from domains.base import Domain

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    "Place it between <start_working_out> and <end_working_out>.\n"
    "Then, provide your solution between <SOLUTION></SOLUTION>"
)

_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    match = _BOXED_RE.search(text)
    if match is None:
        return None
    start = match.end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def _extract_hash_answer(text: str) -> str | None:
    if text is None or "####" not in text:
        return None
    return text.split("####")[1].strip()


class MathDomain(Domain):
    system_prompt = SYSTEM_PROMPT
    _SUPPORTED = {"openai/gsm8k", "EleutherAI/hendrycks_math", "open-r1/DAPO-Math-17k-Processed"}

    def load_dataset(self, name: str, split: str = "train") -> Dataset:
        if name == "openai/gsm8k":
            return self._load_gsm8k(split)
        if name == "EleutherAI/hendrycks_math":
            return self._load_math(split)
        if name == "open-r1/DAPO-Math-17k-Processed":
            return self._load_dapo(split)
        raise NotImplementedError(f"Dataset not supported: {name}. Supported: {self._SUPPORTED}")

    def _load_gsm8k(self, split: str) -> Dataset:
        data = load_dataset("openai/gsm8k", "main")[split]
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": _extract_hash_answer(x.get("answer", "")) or "",
                "difficulty": None,
                "dataset": "gsm8k",
            },
            remove_columns=data.column_names,
        )

    _MATH_CONFIGS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    def _load_math(self, split: str) -> Dataset:
        subsets = [load_dataset("EleutherAI/hendrycks_math", c)[split] for c in self._MATH_CONFIGS]
        data = concatenate_datasets(subsets)
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["problem"]},
                ],
                "answer": _extract_boxed(x.get("solution", "")) or "",
                "difficulty": float(x["level"].split()[-1]) if x.get("level") else None,
                "dataset": "math",
            },
            remove_columns=data.column_names,
        )

    def _load_dapo(self, split: str = "train") -> Dataset:
        data = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")[split]
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["prompt"]},
                ],
                "answer": x.get("solution", ""),
                "difficulty": None,
                "dataset": "dapo",
            },
            remove_columns=data.column_names,
        )

    def filter_by_prompt_length(self, dataset: Dataset, tokenizer, quantile: float = 0.9) -> Dataset:
        """Drop top (1-quantile) prompts by token length — matches notebook preprocessing."""
        lengths = [
            len(tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True))
            for x in dataset
        ]
        threshold = int(np.quantile(lengths, quantile))
        return dataset.select([i for i, l in enumerate(lengths) if l <= threshold])

    def extract_answer(self, text: str) -> str | None:
        m = self._solution_re.search(text)
        return m.group(1).strip() if m else None

    def extract_number(self, text: str) -> str | None:
        m = self._number_re.search(text)
        return m.group(1) if m else None

    def difficulty(self, sample: dict) -> float | None:
        return sample.get("difficulty")
