from __future__ import annotations

from datasets import Dataset, load_dataset

from domains.base import Domain

SYSTEM_PROMPT = (
    "You are given a programming problem.\n"
    "Think through the solution step by step.\n"
    "Place your reasoning between <start_working_out> and <end_working_out>.\n"
    "Then provide your final Python code between <SOLUTION></SOLUTION>"
)


class CodingDomain(Domain):
    """Scaffold — HumanEval / MBPP loader. Full execution-based verifier is in coding/verifier.py."""

    system_prompt = SYSTEM_PROMPT
    _SUPPORTED = {"openai/openai_humaneval", "google-research-datasets/mbpp"}

    def load_dataset(self, name: str, split: str = "test") -> Dataset:
        if "humaneval" in name:
            return self._load_humaneval(split)
        if "mbpp" in name:
            return self._load_mbpp(split)
        raise NotImplementedError(f"Dataset not supported: {name}. Supported: {self._SUPPORTED}")

    def _load_humaneval(self, split: str) -> Dataset:
        data = load_dataset("openai/openai_humaneval")[split]
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["prompt"]},
                ],
                "answer": x["canonical_solution"],
                "test_code": x["test"],
                "entry_point": x["entry_point"],
                "difficulty": None,
                "dataset": "humaneval",
            },
            remove_columns=data.column_names,
        )

    def _load_mbpp(self, split: str) -> Dataset:
        data = load_dataset("google-research-datasets/mbpp")[split]
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["text"]},
                ],
                "answer": x["code"],
                "test_code": "\n".join(x["test_list"]),
                "entry_point": None,
                "difficulty": None,
                "dataset": "mbpp",
            },
            remove_columns=data.column_names,
        )

    def extract_answer(self, text: str) -> str | None:
        m = self._solution_re.search(text)
        return m.group(1).strip() if m else None

    def difficulty(self, sample: dict) -> float | None:
        return None
