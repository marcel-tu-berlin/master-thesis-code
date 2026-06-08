"""Per-component reward logging (T2.1).

TRL only sees the single composed reward function, so the individual
accuracy/format/length/entropy contributions are invisible in the training
logs. The composer stashes per-component raw mean/std and weighted contribution
each step; pop_step_metrics() drains them (averaged across calls) for a
TrainerCallback to log. The composed reward itself must be unchanged.

NaiveSumComposer is torch-free, so its metrics are exercised here; the
advantage-weighted path needs torch and is validated on the GPU box.
"""
import importlib.util
import os

# Direct-load compose.py by path so we skip the training.rewards package
# __init__ (which imports every reward builder -> torch + datasets). torch is
# lazy inside AdvantageWeightedComposer, so NaiveSumComposer loads clean.
_COMPOSE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "rewards", "compose.py",
)
_spec = importlib.util.spec_from_file_location("_compose_under_test", _COMPOSE_PATH)
_compose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compose)
NaiveSumComposer = _compose.NaiveSumComposer


def _const(name, value):
    """A reward callable of a distinctly-named class returning a fixed value.
    Real reward components are distinct classes, so the composer keys metrics on
    type(fn).__name__; the stubs mirror that with dynamic class names.
    """
    cls = type(name, (), {
        "__call__": lambda self, prompts, completions, **kwargs: [value] * len(completions)
    })
    return cls()


def test_naive_sum_unchanged_reward():
    # acc weight 1.0 (value 1.0) + length weight 0.5 (value 2.0) = 1.0 + 1.0 = 2.0
    comp = NaiveSumComposer([(_const("AccStub", 1.0), 1.0), (_const("LenStub", 2.0), 0.5)])
    out = comp(["p", "p"], ["c1", "c2"])
    assert out == [2.0, 2.0]


def test_pop_step_metrics_reports_per_component_and_clears():
    a, b = _const("AccStub", 1.0), _const("LenStub", 2.0)
    comp = NaiveSumComposer([(a, 1.0), (b, 0.5)])
    comp(["p", "p", "p"], ["c", "c", "c"])
    m = comp.pop_step_metrics()
    # Component names come from the callable; here both are the inner 'fn'.
    # Keyed by reward/<name>/<stat> — assert the raw means and L1 contributions.
    raw_means = {k: v for k, v in m.items() if k.endswith("/raw_mean")}
    contribs = {k: v for k, v in m.items() if k.endswith("/contrib_l1")}
    assert abs(sum(raw_means.values()) - 3.0) < 1e-9      # 1.0 + 2.0
    # contrib_l1 = |weight * raw|: 1.0*1.0 + 0.5*2.0 = 2.0
    assert abs(sum(contribs.values()) - 2.0) < 1e-9
    # Buffer cleared after pop.
    assert comp.pop_step_metrics() == {}


def test_pop_averages_across_calls_in_a_step():
    comp = NaiveSumComposer([(_const("AccStub", 1.0), 1.0)])
    comp(["p"], ["c"])          # raw_mean 1.0
    comp(["p", "p"], ["c", "c"])  # raw_mean 1.0
    m = comp.pop_step_metrics()
    rm = [v for k, v in m.items() if k.endswith("/raw_mean")]
    assert len(rm) == 1 and abs(rm[0] - 1.0) < 1e-9
