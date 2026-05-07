import re
from abc import ABC, abstractmethod

from datasets import Dataset


class Domain(ABC):
    system_prompt: str
    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"

    def __init__(self) -> None:
        # Match <reasoning_end> ... <SOLUTION>(group)<\/SOLUTION>. Trailing
        # tokens after </SOLUTION> are tolerated so the model is not penalized
        # for emitting EOS markers, whitespace, or stray noise after the close.
        self._solution_re = re.compile(
            re.escape(self.reasoning_end)
            + r".*?"
            + re.escape(self.solution_start)
            + r"(.+?)"
            + re.escape(self.solution_end),
            re.DOTALL,
        )
        self._number_re = re.compile(
            re.escape(self.solution_start) + r".*?[\s]{0,}([-]?[\d\.,]{1,})",
            re.MULTILINE | re.DOTALL,
        )

    @property
    def tags(self) -> dict[str, str]:
        return {
            "reasoning_start": self.reasoning_start,
            "reasoning_end": self.reasoning_end,
            "solution_start": self.solution_start,
            "solution_end": self.solution_end,
        }

    @abstractmethod
    def load_dataset(self, name: str, split: str) -> Dataset: ...

    @abstractmethod
    def extract_answer(self, text: str) -> str | None:
        """Extract the answer string from a model completion."""
        ...

    @abstractmethod
    def difficulty(self, sample: dict) -> float | None:
        """Numeric difficulty score for a sample, or None if unavailable."""
        ...

    def check_exact(self, extracted: str, ground_truth: str) -> bool:
        return extracted.strip() == ground_truth.strip()

    def check_numeric(self, extracted: str, ground_truth: str, tol: float = 1e-6) -> bool:
        try:
            a = float(extracted.strip().replace(",", ""))
            b = float(ground_truth.strip().replace(",", ""))
            if b == 0:
                return abs(a) < tol
            return abs(a - b) / abs(b) < tol or abs(a - b) < tol
        except (ValueError, AttributeError):
            return False

    def score_answer(self, extracted: str | None, truth: str) -> float:
        """Graded answer reward — magnitudes match thesis notebooks."""
        if extracted is None:
            return -2.0
        if extracted == truth:
            return 5.0
        if extracted.strip() == truth.strip():
            return 3.5
        try:
            ext_num = float(extracted.strip().replace(",", ""))
            truth_num = float(truth.strip().replace(",", ""))
            ratio = ext_num / truth_num
            if 0.9 <= ratio <= 1.1:
                return 2.0
            if 0.8 <= ratio <= 1.2:
                return 1.5
            return -2.5
        except (ValueError, ZeroDivisionError, AttributeError):
            return -4.5

    def score_numbers(self, extracted: str | None, truth: str) -> float:
        """Strict float equality reward for numeric answers."""
        if extracted is None:
            return -2.5
        try:
            a = float(extracted.strip().replace(",", ""))
            b = float(truth.strip())
            return 3.5 if a == b else -1.5
        except (ValueError, AttributeError):
            return -1.5

    def is_correct(self, completion: str, ground_truth: str) -> bool:
        """Binary correctness for evaluation (not training)."""
        extracted = self.extract_answer(completion)
        if extracted is None:
            return False
        return self.check_exact(extracted, ground_truth) or self.check_numeric(extracted, ground_truth)

    def build_chat_template(self, tokenizer) -> None:
        """Inject domain-specific Jinja2 chat template into tokenizer."""
        sp = self.system_prompt
        rs = self.reasoning_start
        chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            f"{{{{ '{sp}' + eos_token }}}}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            f"{{% if add_generation_prompt %}}{{{{ '{rs}' }}}}{{% endif %}}"
        )
        tokenizer.chat_template = chat_template
