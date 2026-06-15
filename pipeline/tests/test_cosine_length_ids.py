from training.rewards.cosine_length import CosineLengthReward


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return text.split()  # one pseudo-token per word


class _Dom:
    def is_correct(self, text, truth):
        return True


def test_prefers_completion_ids_over_text():
    r = CosineLengthReward(_Tok(), _Dom(), max_len=10)
    # completion_ids says 2 tokens; the decoded text would re-encode to 5.
    out_ids = r(["p"], ["a b c d e"], answer=["x"], completion_ids=[[1, 2]])
    out_txt = r(["p"], ["a b c d e"], answer=["x"])
    assert out_ids != out_txt  # id path (2 tokens) differs from text path (5)


def test_falls_back_to_text_without_ids():
    r = CosineLengthReward(_Tok(), _Dom(), max_len=10)
    out = r(["p"], ["a b c"], answer=["x"])
    assert len(out) == 1
