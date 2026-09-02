import json
import os

import matplotlib
matplotlib.use("Agg")

from eval import plots
from eval.metrics import SampleResult, compute_metrics
from eval.agentic_eval import _metrics_to_dict


def _report_dict(exp_id, toks_correct, toks_wrong):
    """A faithful eval_report.json built through the production metrics path."""
    results = (
        [SampleResult(correct=True, n_tokens=t, n_steps=1) for t in toks_correct]
        + [SampleResult(correct=False, n_tokens=t, n_steps=1) for t in toks_wrong]
    )
    m = compute_metrics(results)
    return {
        "experiment_id": exp_id,
        "model_slug": "qwen3-1.7b",
        "seed": 42,
        "compose_method": "advantage_weighted",
        "mode": "agentic",
        "results": {"agentic": _metrics_to_dict(m)},
    }


def _write_run(tmp_path, exp_id, toks_correct, toks_wrong, train_log=None):
    d = tmp_path / exp_id
    d.mkdir()
    (d / "eval_report.json").write_text(json.dumps(_report_dict(exp_id, toks_correct, toks_wrong)))
    if train_log is not None:
        (d / "train_log.json").write_text(json.dumps(train_log))
    return str(d)


_SYNTH_LOG = [
    {"step": s, "reward": 0.1 * s, "kl": 0.01 * s, "loss": 1.0 - 0.05 * s,
     "completions/mean_length": 400 - 5 * s, "completions/clipped_ratio": 0.5 - 0.05 * s,
     "reward/EnvReward/raw_mean": 0.2 * s, "reward/CosineLengthReward/raw_mean": -0.1 * s}
    for s in range(1, 8)
]


def test_load_report_from_dir(tmp_path):
    d = _write_run(tmp_path, "e5-foo", [200, 300], [1024])
    r = plots.load_report(d)
    assert r["experiment_id"] == "e5-foo"
    assert r["agentic"]["accuracy"] == 2 / 3
    assert len(r["samples"]) == 3


def test_load_report_from_json_path(tmp_path):
    d = _write_run(tmp_path, "e6-bar", [100], [])
    r = plots.load_report(os.path.join(d, "eval_report.json"))
    assert r["experiment_id"] == "e6-bar"


def test_load_report_missing_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        plots.load_report(str(tmp_path / "nope"))


def test_load_report_takes_the_only_split_when_it_has_a_protocol_name(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"experiment_id": "z", "results": {"held_out": {}}}))
    assert plots.load_report(str(p))["split_name"] == "held_out"


def test_load_report_refuses_to_guess_between_splits(tmp_path):
    # Silently picking one would compare the held-out split of one arm against
    # the shifted split of another.
    import pytest
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"experiment_id": "z",
                             "results": {"held_out": {}, "shifted": {}}}))
    with pytest.raises(ValueError):
        plots.load_report(str(p))
    assert plots.load_report(str(p), split_name="shifted")["split_name"] == "shifted"


def test_load_report_empty_results_raises(tmp_path):
    import pytest
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"experiment_id": "z", "results": {}}))
    with pytest.raises(ValueError):
        plots.load_report(str(p))


def test_correct_wrong_split():
    samples = [{"correct": True, "n_tokens": 10}, {"correct": False, "n_tokens": 1024},
               {"correct": True, "n_tokens": 20}]
    c, w = plots._correct_wrong_tokens(samples)
    assert sorted(c.tolist()) == [10.0, 20.0] and w.tolist() == [1024.0]


def test_mean_ci_on_correct():
    samples = [{"correct": True, "n_tokens": 100}, {"correct": True, "n_tokens": 300},
               {"correct": False, "n_tokens": 1024}]
    mean, lo, hi = plots._mean_ci_on_correct(samples)
    assert mean == 200.0 and lo <= mean <= hi


def test_mean_ci_on_correct_no_correct():
    mean, lo, hi = plots._mean_ci_on_correct([{"correct": False, "n_tokens": 1024}])
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


def test_plot_comparison_two_axes_bars(tmp_path):
    reports = [plots.load_report(_write_run(tmp_path, e, [200, 300], [1024]))
               for e in ("e5", "e6", "e7")]
    fig = plots.plot_comparison(reports)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 3 and len(fig.axes[1].patches) == 3


def test_plot_distributions_one_axis_per_report(tmp_path):
    reports = [plots.load_report(_write_run(tmp_path, e, [200, 300], [1024]))
               for e in ("e5", "e6")]
    fig = plots.plot_distributions(reports)
    assert len(fig.axes) == 2


def test_plot_efficiency_one_axis_labeled_points(tmp_path):
    reports = [plots.load_report(_write_run(tmp_path, e, [200, 300], [1024]))
               for e in ("e5", "e6", "e7")]
    fig = plots.plot_efficiency(reports)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].texts) == 3  # one annotation per experiment


def test_plot_training_curves_panels():
    fig = plots.plot_training_curves(_SYNTH_LOG)
    assert fig is not None and len(fig.axes) >= 5  # reward, length, kl, loss, components


def test_plot_training_curves_empty_none():
    assert plots.plot_training_curves([]) is None


def test_make_figures_writes_core_pngs(tmp_path):
    runs = [_write_run(tmp_path, e, [200, 300, 250], [1024, 900]) for e in ("e5", "e6")]
    out = tmp_path / "plots"
    written = plots.make_figures(runs, str(out))
    names = {os.path.basename(w) for w in written}
    assert {"comparison.png", "distributions.png", "efficiency.png"} <= names
    for w in written:
        assert os.path.exists(w) and os.path.getsize(w) > 0


def test_make_figures_training_curves_when_log_present(tmp_path):
    runs = [
        _write_run(tmp_path, "e7", [200, 300], [1024], train_log=_SYNTH_LOG),
        _write_run(tmp_path, "e8", [200, 300], [1024]),
    ]
    out = tmp_path / "plots"
    written = plots.make_figures(runs, str(out))
    curve_files = [w for w in written if os.path.basename(w).startswith("training_curves_")]
    assert len(curve_files) == 1 and os.path.exists(curve_files[0])


# --- multi-split (protocol) reports: load_report refuses to guess, but
# --- make_figures must enumerate the splits rather than crash. Before this,
# --- e27/e28/e29 could produce no figures at all.

def _protocol_report(tmp_path, exp_id, split_names):
    run = tmp_path / exp_id
    run.mkdir(parents=True)
    results = {}
    for i, name in enumerate(split_names):
        m = compute_metrics(
            [SampleResult(correct=True, n_tokens=100 + i, n_steps=1),
             SampleResult(correct=True, n_tokens=120 + i, n_steps=1),
             SampleResult(correct=False, n_tokens=900 + i, n_steps=1),
             SampleResult(correct=False, n_tokens=920 + i, n_steps=1)]
        )
        results[name] = _metrics_to_dict(m)
    (run / "eval_report.json").write_text(
        json.dumps({"experiment_id": exp_id, "results": results}))
    return str(run)


def test_make_figures_renders_one_set_per_split(tmp_path):
    run = _protocol_report(tmp_path, "e27-x", ["held_out", "shifted"])
    written = {os.path.basename(p) for p in
               plots.make_figures([run], str(tmp_path / "out"))}
    for split in ("held_out", "shifted"):
        for kind in ("comparison", "distributions", "efficiency"):
            assert f"{kind}_{split}.png" in written


def test_make_figures_single_split_keeps_unsuffixed_names(tmp_path):
    run = _protocol_report(tmp_path, "e5-x", ["agentic"])
    written = {os.path.basename(p) for p in
               plots.make_figures([run], str(tmp_path / "out"))}
    assert {"comparison.png", "distributions.png", "efficiency.png"} <= written


def test_make_figures_split_override_renders_only_that_split(tmp_path):
    run = _protocol_report(tmp_path, "e27-x", ["held_out", "shifted"])
    written = {os.path.basename(p) for p in
               plots.make_figures([run], str(tmp_path / "out"), split="held_out")}
    assert "comparison.png" in written              # single split -> no suffix
    assert not any("shifted" in w for w in written)


# --- cross-arm training overlay, dose-response, per-episode deltas ---

def test_plot_training_overlay_one_panel_per_present_key():
    logs = [("e30", _SYNTH_LOG), ("e31", _SYNTH_LOG)]
    fig = plots.plot_training_overlay(logs, keys=[("kl", "KL"), ("nope", "absent")])
    # The absent key gets no panel; the contrib_l1 panel is absent too (this log
    # carries raw_mean only), so exactly one panel is drawn.
    assert fig is not None and len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 4               # raw + smoothed, per arm


def test_plot_training_overlay_adds_a_shaped_component_panel():
    log = [dict(e, **{"reward/CosineLengthReward/contrib_l1": 0.5,
                      "reward/EnvReward/contrib_l1": 0.9}) for e in _SYNTH_LOG]
    fig = plots.plot_training_overlay([("e31", log)], keys=[("kl", "KL")])
    assert len(fig.axes) == 2
    # EnvReward has its own panel already; only the shaped component is overlaid.
    labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert labels == ["e31: CosineLengthReward"]


def test_plot_training_overlay_no_data_returns_none():
    assert plots.plot_training_overlay([("e30", [])], keys=[("kl", "KL")]) is None


def test_plot_dose_response_draws_a_series_per_condition_and_a_reference():
    rows = [
        {"condition": "E2", "lam": 0.0, "losses": 0, "median_dtok": 0.0},
        {"condition": "E2", "lam": 1.0, "losses": 14, "median_dtok": (-1130.0, -1161.0, -986.0)},
        {"condition": "E3", "lam": 0.0, "losses": 0, "median_dtok": 0.0},
        {"condition": "E3", "lam": 1.0, "losses": 54, "median_dtok": (-70.0, -151.0, 61.0)},
    ]
    fig = plots.plot_dose_response(
        rows, [("losses", "losses"), ("median_dtok", "median dtok")],
        refs={"losses": ("E0 base", 40)})
    assert len(fig.axes) == 2
    # Two conditions -> two connected series per panel (plus the dashed E0 line).
    assert len([l for l in fig.axes[0].lines if l.get_linestyle() == "-"]) == 2
    assert any(l.get_linestyle() == "--" for l in fig.axes[0].lines)


def test_plot_dose_response_skips_a_missing_metric():
    rows = [{"condition": "E2", "lam": 0.0, "losses": 0, "median_dtok": None},
            {"condition": "E2", "lam": 1.0, "losses": 3, "median_dtok": None}]
    fig = plots.plot_dose_response(rows, [("median_dtok", "median dtok")])
    assert len(fig.axes[0].lines) == 0


def test_plot_paired_deltas_one_panel_per_arm_sorted():
    fig = plots.plot_paired_deltas([("e31", [-100, 20, -300]), ("e32", [5, -5])])
    assert len(fig.axes) == 2
    heights = [p.get_height() for p in fig.axes[0].patches]
    assert heights == sorted(heights)


def test_plot_paired_deltas_no_diffs_returns_none():
    assert plots.plot_paired_deltas([("e31", [])]) is None


def test_make_figures_draws_the_overlay_only_from_two_logs_up(tmp_path):
    one = _write_run(tmp_path, "e7", [200], [1024], train_log=_SYNTH_LOG)
    names = {os.path.basename(p) for p in plots.make_figures([one], str(tmp_path / "o1"))}
    assert "training_overlay.png" not in names
    two = _write_run(tmp_path, "e8", [200], [1024], train_log=_SYNTH_LOG)
    names = {os.path.basename(p) for p in plots.make_figures([one, two], str(tmp_path / "o2"))}
    assert "training_overlay.png" in names


def test_make_figures_skips_a_run_missing_the_split(tmp_path):
    both = _protocol_report(tmp_path, "e27-x", ["held_out", "shifted"])
    only = _protocol_report(tmp_path, "e5-x", ["held_out"])
    # Must not raise: the run without `shifted` is skipped, the other still draws.
    written = {os.path.basename(p) for p in
               plots.make_figures([both, only], str(tmp_path / "out"))}
    assert "comparison_held_out.png" in written
    assert "comparison_shifted.png" in written
