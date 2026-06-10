"""Correctness of MathDomain answer grading (training-signal bugs)."""
from domains.math.loader import MathDomain

d = MathDomain()

RS, RE = "<start_working_out>", "<end_working_out>"
SS, SE = "<SOLUTION>", "</SOLUTION>"


# Task 7: numeric grading
def test_score_numbers_strips_commas_from_truth():
    assert d.score_numbers("1000", "1,000") == 3.5

def test_score_answer_zero_truth_correct_value():
    assert d.score_answer("0.0", "0") == 5.0

def test_score_answer_zero_truth_wrong_value():
    assert d.score_answer("7", "0") == -2.5


# Task 8: number extraction is anchored to the SOLUTION block and needs a digit
def test_extract_number_requires_a_digit():
    assert d.extract_number(f"{RS}x{RE}{SS}.{SE}") is None

def test_extract_number_reads_solution_block():
    assert d.extract_number(f"{RS}think{RE}{SS}42{SE}") == "42"

def test_extract_number_ignores_cot_solution_mention():
    t = f"{RS}maybe {SS}99{SE} no{RE}{SS}7{SE}"
    assert d.extract_number(t) == "7"


# Task 9: empty answers are never rewarded
def test_empty_extraction_not_rewarded():
    assert d.score_answer("", "") != 5.0
    assert d.score_answer("", "42") < 0
