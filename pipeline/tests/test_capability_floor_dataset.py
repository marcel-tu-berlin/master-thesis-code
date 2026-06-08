"""Dataset-backed capability floor (T2.6).

The 6-prompt floor saturates at acc=1.0, so it can't catch a regression. A
graded slice of a real benchmark (n>=50) can. The slice must be disjoint from
the id_split head[:N] it would otherwise duplicate, so the floor takes a tail
slice; selection is a pure function, tested here. Generation + grading reuse
_run_split, validated end-to-end on the GPU box.
"""
from eval.ood_probes import _floor_slice_indices


def test_floor_tail_slice_is_disjoint_from_id_head():
    tail = list(_floor_slice_indices(1319, 50, "tail"))   # GSM8K test ~1319 rows
    assert tail == list(range(1269, 1319))
    head = list(_floor_slice_indices(1319, 200, "head"))  # id_split head[:200]
    assert set(tail).isdisjoint(head)


def test_floor_head_slice():
    assert list(_floor_slice_indices(1319, 50, "head")) == list(range(0, 50))


def test_floor_slice_clamps_to_total():
    assert list(_floor_slice_indices(30, 50, "tail")) == list(range(0, 30))
    assert list(_floor_slice_indices(30, 50, "head")) == list(range(0, 30))
