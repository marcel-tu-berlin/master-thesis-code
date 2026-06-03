"""The capability floor must be able to fail.

The old 5-item set ("2+2", "capital of France") was so trivial every run scored
1.0, so the floor caught no regression. The probe set now includes items a
collapsed or over-compressed model fails, and the trap items mark a wrong answer
as wrong so the floor discriminates.
"""
from eval.ood_probes import _DEFAULT_CAPABILITY_PROMPTS, _capability_match


def test_probe_set_includes_discriminative_items():
    answers = {a for _, a in _DEFAULT_CAPABILITY_PROMPTS}
    assert "9.9" in answers          # decimal-magnitude trap
    assert "cold" in answers         # one-word instruction compliance
    assert len(_DEFAULT_CAPABILITY_PROMPTS) >= 6


def test_magnitude_trap_marks_wrong_answer_wrong():
    assert _capability_match("9.9", "9.9") is True
    assert _capability_match("9.9", "9.11") is False  # fell for the trap


def test_two_step_arithmetic_marks_partial_wrong():
    # 12*12 - 100 = 44; a model that forgets the "-100" answers 144.
    assert _capability_match("44", "44") is True
    assert _capability_match("44", "144") is False


def test_one_word_instruction():
    assert _capability_match("cold", "cold") is True
    assert _capability_match("cold", "hot") is False
