"""Eval probe extraction/matching correctness (Tasks 12-13)."""
from eval.ood_probes import _extract_letter, _capability_match, _final_answer
from domains.math.loader import MathDomain

d = MathDomain()
RS, RE = "<start_working_out>", "<end_working_out>"
SS, SE = "<SOLUTION>", "</SOLUTION>"


# Task 12 — MMLU letter extraction
def test_letter_prefers_solution_block():
    assert _extract_letter(d, f"{RS}x{RE}{SS}C{SE}") == "C"

def test_letter_ignores_article_a():
    assert _extract_letter(d, "A cyclist rides north. The answer is B.") == "B"

def test_letter_bare_compliant_answer():
    assert _extract_letter(d, "B") == "B"


# Task 13 — capability-floor matching
def test_capability_match_trailing_period():
    assert _capability_match("cold", "The answer is cold.") is True

def test_capability_match_decimal_equiv():
    assert _capability_match("4", "4.0") is True

def test_capability_match_substring_guard():
    assert _capability_match("4", "14") is False

def test_capability_match_integer_does_not_match_longer_decimal():
    # expected integer must not spuriously match a wrong decimal that starts
    # with it: '4' is not '4.5'. (Tolerating the trailing period must not also
    # tolerate a decimal continuation.)
    assert _capability_match("4", "4.5") is False
    assert _capability_match("7", "7.2") is False
    assert _capability_match("4", "The answer is 4.5 dollars") is False
    # but a value-equal decimal still matches via the numeric fallback, and an
    # integer with a trailing sentence period still matches
    assert _capability_match("4", "4.0") is True
    assert _capability_match("4", "The answer is 4.") is True

def test_final_answer_uses_last_line_when_no_solution():
    assert _final_answer(d, "Comparing 9.9 and 9.11...\n9.11") == "9.11"

def test_final_answer_prefers_solution_block():
    assert _final_answer(d, f"{RS}x{RE}{SS}9.9{SE}") == "9.9"
