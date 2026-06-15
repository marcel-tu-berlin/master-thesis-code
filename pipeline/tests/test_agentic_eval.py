from eval.metrics import SampleResult, compute_metrics


def test_mean_steps_reported():
    rs = [
        SampleResult(correct=True, n_tokens=10, n_steps=1),
        SampleResult(correct=False, n_tokens=20, n_steps=3),
    ]
    m = compute_metrics(rs)
    assert abs(m.mean_steps - 2.0) < 1e-9


def test_mean_steps_none_for_dataset_eval():
    rs = [
        SampleResult(correct=True, n_tokens=10),
        SampleResult(correct=False, n_tokens=20),
    ]
    assert compute_metrics(rs).mean_steps is None
