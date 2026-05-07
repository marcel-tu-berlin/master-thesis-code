import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from domains.base import Domain

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    "Place it between <start_working_out> and <end_working_out>.\n"
    "Then, provide your solution between <SOLUTION></SOLUTION>"
)

_BOXED_TOKEN = r"\boxed{"


def _extract_boxed(text: str) -> str | None:
    """Extract content from the *last* \\boxed{...}, handling nested braces.

    Hendrycks MATH solutions sometimes contain multiple \\boxed expressions
    (intermediate steps); the final answer is the last one.
    """
    idx = text.rfind(_BOXED_TOKEN)
    if idx < 0:
        return None
    start = idx + len(_BOXED_TOKEN)
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

    @staticmethod
    def _dapo_answer(raw: str) -> str:
        """DAPO's `solution` field is sometimes a bare string, sometimes a
        LaTeX expression wrapping a final \\boxed{...}. Prefer the boxed
        contents when present so reward functions compare clean values."""
        if not raw:
            return ""
        boxed = _extract_boxed(raw)
        return boxed if boxed is not None else raw.strip()

    # DAPO ships only a `train` split on HuggingFace. When eval requests
    # `test` (or anything non-train), carve a deterministic held-out tail
    # from train so ID-split eval doesn't crash and never overlaps with the
    # training prefix used under `dataset_size_limit`.
    _DAPO_HELDOUT_TAIL = 1000

    def _load_dapo(self, split: str = "train") -> Dataset:
        all_splits = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")
        if split in all_splits:
            data = all_splits[split]
        elif split == "train":
            raise KeyError(f"DAPO dataset missing required split: train. Available: {list(all_splits)}")
        else:
            train = all_splits["train"]
            tail = min(self._DAPO_HELDOUT_TAIL, max(len(train) // 10, 1))
            data = train.select(range(len(train) - tail, len(train)))
            print(f"  DAPO: synthesising '{split}' from last {tail} train rows (no native split)")
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["prompt"]},
                ],
                "answer": self._dapo_answer(x.get("solution", "")),
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
