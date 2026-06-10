"""Lenient (format-agnostic) extraction for base-model assessment (Task 15)."""
from domains.math.loader import MathDomain

d = MathDomain()
RS, RE = "<start_working_out>", "<end_working_out>"
SS, SE = "<SOLUTION>", "</SOLUTION>"


def test_lenient_prefers_solution_block():
    assert d.extract_answer_lenient(f"{RS}x{RE}{SS}7{SE}") == "7"

def test_lenient_extracts_bare_final_number():
    assert d.extract_answer_lenient("The cyclist travels 42 km total") == "42"

def test_lenient_extracts_boxed():
    assert d.extract_answer_lenient(r"final answer \boxed{15}") == "15"

def test_lenient_returns_none_when_no_number():
    assert d.extract_answer_lenient("no numerals here") is None

def test_is_correct_strict_vs_lenient():
    # untagged base-model output: strict fails, lenient passes
    assert d.is_correct("the answer is 42", "42") is False
    assert d.is_correct("the answer is 42", "42", lenient=True) is True

def test_is_correct_default_is_strict():
    # default (training) behavior unchanged: no lenient kwarg = strict
    assert d.is_correct("the answer is 42", "42") is False
