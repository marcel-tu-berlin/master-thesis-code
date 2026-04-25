from __future__ import annotations


def extract_content(completion) -> str:
    try:
        return completion[0]["content"]
    except (IndexError, KeyError, TypeError):
        return ""
