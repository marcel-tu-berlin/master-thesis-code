import json
import os

import matplotlib

matplotlib.use("Agg")

from eval import plots
from eval.agentic_eval import _metrics_to_dict
from eval.metrics import SampleResult, compute_metrics


def _report_dict(exp_id, toks_correct, toks_wrong):
    """A faithful eval_report.json built through the production metrics path."""
    results = [
        SampleResult(correct=True, n_tokens=t, n_steps=1) for t in toks_correct
    ] + [SampleResult(correct=False, n_tokens=t, n_steps=1) for t in toks_wrong]
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
    (d / "eval_report.json").write_text(
        json.dumps(_report_dict(exp_id, toks_correct, toks_wrong))
    )
    if train_log is not None:
        (d / "train_log.json").write_text(json.dumps(train_log))
    return str(d)


_SYNTH_LOG = [
    {
        "step": s,
        "reward": 0.1 * s,
        "kl": 0.01 * s,
        "loss": 1.0 - 0.05 * s,
        "completions/mean_length": 400 - 5 * s,
        "completions/clipped_ratio": 0.5 - 0.05 * s,
        "reward/EnvReward/raw_mean": 0.2 * s,
        "reward/CosineLengthReward/raw_mean": -0.1 * s,
    }
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
    p.write_text(
        json.dumps({"experiment_id": "z", "results": {"held_out": {}, "shifted": {}}})
    )
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
    samples = [
        {"correct": True, "n_tokens": 10},
        {"correct": False, "n_tokens": 1024},
        {"correct": True, "n_tokens": 20},
    ]
    c, w = plots._correct_wrong_tokens(samples)
    assert sorted(c.tolist()) == [10.0, 20.0] and w.tolist() == [1024.0]


def test_mean_ci_on_correct():
    samples = [
        {"correct": True, "n_tokens": 100},
        {"correct": True, "n_tokens": 300},
        {"correct": False, "n_tokens": 1024},
    ]
    mean, lo, hi = plots._mean_ci_on_correct(samples)
    assert mean == 200.0 and lo <= mean <= hi


def test_mean_ci_on_correct_no_correct():
    mean, lo, hi = plots._mean_ci_on_correct([{"correct": False, "n_tokens": 1024}])
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


def test_plot_comparison_two_axes_bars(tmp_path):
    reports = [
        plots.load_report(_write_run(tmp_path, e, [200, 300], [1024]))
        for e in ("e5", "e6", "e7")
    ]
    fig = plots.plot_comparison(reports)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 3 and len(fig.axes[1].patches) == 3


def test_plot_distributions_one_axis_per_report(tmp_path):
    reports = [
        plots.load_report(_write_run(tmp_path, e, [200, 300], [1024]))
        for e in ("e5", "e6")
    ]
    fig = plots.plot_distributions(reports)
    assert len(fig.axes) == 2


def test_plot_efficiency_one_axis_labeled_points(tmp_path):
    reports = [
        plots.load_report(_write_run(tmp_path, e, [200, 300], [1024]))
        for e in ("e5", "e6", "e7")
    ]
    fig = plots.plot_efficiency(reports)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].texts) == 3  # one annotation per experiment


def test_plot_training_curves_panels():
    fig = plots.plot_training_curves(_SYNTH_LOG)
    assert (
        fig is not None and len(fig.axes) >= 5
    )  # reward, length, kl, loss, components


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
    curve_files = [
        w for w in written if os.path.basename(w).startswith("training_curves_")
    ]
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
            [
                SampleResult(correct=True, n_tokens=100 + i, n_steps=1),
                SampleResult(correct=True, n_tokens=120 + i, n_steps=1),
                SampleResult(correct=False, n_tokens=900 + i, n_steps=1),
                SampleResult(correct=False, n_tokens=920 + i, n_steps=1),
            ]
        )
        results[name] = _metrics_to_dict(m)
    (run / "eval_report.json").write_text(
        json.dumps({"experiment_id": exp_id, "results": results})
    )
    return str(run)


def test_make_figures_renders_one_set_per_split(tmp_path):
    run = _protocol_report(tmp_path, "e27-x", ["held_out", "shifted"])
    written = {
        os.path.basename(p) for p in plots.make_figures([run], str(tmp_path / "out"))
    }
    for split in ("held_out", "shifted"):
        for kind in ("comparison", "distributions", "efficiency"):
            assert f"{kind}_{split}.png" in written


def test_make_figures_single_split_keeps_unsuffixed_names(tmp_path):
    run = _protocol_report(tmp_path, "e5-x", ["agentic"])
    written = {
        os.path.basename(p) for p in plots.make_figures([run], str(tmp_path / "out"))
    }
    assert {"comparison.png", "distributions.png", "efficiency.png"} <= written


def test_make_figures_split_override_renders_only_that_split(tmp_path):
    run = _protocol_report(tmp_path, "e27-x", ["held_out", "shifted"])
    written = {
        os.path.basename(p)
        for p in plots.make_figures([run], str(tmp_path / "out"), split="held_out")
    }
    assert "comparison.png" in written  # single split -> no suffix
    assert not any("shifted" in w for w in written)


# --- cross-arm training overlay, dose-response, per-episode deltas ---


def test_plot_training_overlay_one_panel_per_present_key():
    logs = [("e30", _SYNTH_LOG), ("e31", _SYNTH_LOG)]
    fig = plots.plot_training_overlay(logs, keys=[("kl", "KL"), ("nope", "absent")])
    # The absent key gets no panel; the contrib_l1 panel is absent too (this log
    # carries raw_mean only), so exactly one panel is drawn.
    assert fig is not None and len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 4  # raw + smoothed, per arm


def test_plot_training_overlay_adds_a_shaped_component_panel():
    log = [
        dict(
            e,
            **{
                "reward/CosineLengthReward/contrib_l1": 0.5,
                "reward/EnvReward/contrib_l1": 0.9,
            },
        )
        for e in _SYNTH_LOG
    ]
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
        {
            "condition": "E2",
            "lam": 1.0,
            "losses": 14,
            "median_dtok": (-1130.0, -1161.0, -986.0),
        },
        {"condition": "E3", "lam": 0.0, "losses": 0, "median_dtok": 0.0},
        {
            "condition": "E3",
            "lam": 1.0,
            "losses": 54,
            "median_dtok": (-70.0, -151.0, 61.0),
        },
    ]
    fig = plots.plot_dose_response(
        rows,
        [("losses", "losses"), ("median_dtok", "median dtok")],
        refs={"losses": ("E0 base", 40)},
    )
    assert len(fig.axes) == 2
    # Two conditions -> two connected series per panel (plus the dashed E0 line).
    assert len([n for n in fig.axes[0].lines if n.get_linestyle() == "-"]) == 2
    assert any(n.get_linestyle() == "--" for n in fig.axes[0].lines)


def test_plot_dose_response_skips_a_missing_metric():
    rows = [
        {"condition": "E2", "lam": 0.0, "losses": 0, "median_dtok": None},
        {"condition": "E2", "lam": 1.0, "losses": 3, "median_dtok": None},
    ]
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
    names = {
        os.path.basename(p) for p in plots.make_figures([one], str(tmp_path / "o1"))
    }
    assert "training_overlay.png" not in names
    two = _write_run(tmp_path, "e8", [200], [1024], train_log=_SYNTH_LOG)
    names = {
        os.path.basename(p)
        for p in plots.make_figures([one, two], str(tmp_path / "o2"))
    }
    assert "training_overlay.png" in names


def test_make_figures_skips_a_run_missing_the_split(tmp_path):
    both = _protocol_report(tmp_path, "e27-x", ["held_out", "shifted"])
    only = _protocol_report(tmp_path, "e5-x", ["held_out"])
    # Must not raise: the run without `shifted` is skipped, the other still draws.
    written = {
        os.path.basename(p)
        for p in plots.make_figures([both, only], str(tmp_path / "out"))
    }
    assert "comparison_held_out.png" in written
    assert "comparison_shifted.png" in written


# --- off-target panel (RQ2), stop reasons, per-turn profile, seed replicates ---


def _offtarget_results(n_done, n_cap, invalid=0, repeated=0):
    """Episodes carrying the termination and action-level fields the RQ2 panel
    reads, built through the production SampleResult shape."""
    out = [
        SampleResult(
            correct=True,
            n_tokens=300,
            n_steps=2,
            terminated=True,
            stop_reason="env_done",
            tool_calls=["click", "click"],
            n_actions=2,
            n_invalid_actions=invalid,
            n_repeated_actions=repeated,
        )
        for _ in range(n_done)
    ]
    out += [
        SampleResult(
            correct=False,
            n_tokens=4096,
            n_steps=1,
            terminated=False,
            stop_reason="hit_generation_cap",
            tool_calls=["click"],
            n_actions=1,
            n_invalid_actions=0,
            n_repeated_actions=0,
        )
        for _ in range(n_cap)
    ]
    return out


def _offtarget_run(tmp_path, exp_id, n_done, n_cap, invalid=0, repeated=0, turns=None):
    d = tmp_path / exp_id
    d.mkdir()
    m = compute_metrics(_offtarget_results(n_done, n_cap, invalid, repeated))
    (d / "eval_report.json").write_text(
        json.dumps(
            {"experiment_id": exp_id, "results": {"agentic": _metrics_to_dict(m)}}
        )
    )
    if turns is not None:
        (d / "episodes_agentic.jsonl").write_text(
            "".join(
                json.dumps({"seed": i, "correct": True, "n_tokens": 300, "turns": t})
                + "\n"
                for i, t in enumerate(turns)
            )
        )
    return str(d)


def _turn(n_tokens, reasoning=None, content=None):
    rec = {"n_tokens": n_tokens, "reasoning": "r", "content": "c", "tool_calls": []}
    if reasoning is not None:
        rec["n_reasoning_tokens"] = reasoning
        rec["n_content_tokens"] = content
    return rec


def test_short_keeps_a_seed_suffix():
    # Without it every seed of one arm draws an identically-labelled line, in
    # the figure whose whole job is separating an effect from seed noise.
    assert plots._short("e30-browsergym-e1-menu-qwen3-1_7b") == "e30"
    assert plots._short("e30-browsergym-e1-menu-qwen3-1_7b-s43") == "e30-s43"


def test_plot_offtarget_draws_a_group_per_present_rate(tmp_path):
    reports = [
        plots.load_report(_offtarget_run(tmp_path, "e30", 8, 2, invalid=1)),
        plots.load_report(_offtarget_run(tmp_path, "e31", 6, 4)),
    ]
    fig = plots.plot_offtarget(reports)
    assert fig is not None
    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "non-termination" in ticks and "wrong termination" in ticks
    assert "first-action termination" in ticks
    assert "invalid action" in ticks and "repeated action" in ticks
    count_ticks = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert "mean preceding actions" in count_ticks


def test_plot_offtarget_none_when_no_report_carries_a_rate(tmp_path):
    # A dataset-mode report has no termination records at all.
    r = plots.load_report(_write_run(tmp_path, "e5", [200], [1024]))
    assert plots.plot_offtarget([r]) is None


def test_plot_stop_reasons_stacks_to_one_per_arm(tmp_path):
    reports = [
        plots.load_report(_offtarget_run(tmp_path, "e30", 8, 2)),
        plots.load_report(_offtarget_run(tmp_path, "e31", 5, 5)),
    ]
    fig = plots.plot_stop_reasons(reports)
    ax = fig.axes[0]
    totals = {}
    for p in ax.patches:
        totals[round(p.get_x(), 3)] = (
            totals.get(round(p.get_x(), 3), 0) + p.get_height()
        )
    assert all(abs(v - 1.0) < 1e-9 for v in totals.values())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "hit_generation_cap" in labels and "env_done" in labels


def test_plot_stop_reasons_none_without_records(tmp_path):
    r = plots.load_report(_write_run(tmp_path, "e5", [200], [1024]))
    assert plots.plot_stop_reasons([r]) is None


def test_plot_turn_profile_averages_only_episodes_that_reached_the_turn():
    # Arm e31: one episode of two turns (100, 200), one of a single turn (400).
    # Turn 1 averages both (250); turn 2 sees only the first episode (200).
    fig = plots.plot_turn_profile([("e31", [[_turn(100), _turn(200)], [_turn(400)]])])
    line = fig.axes[0].lines[0]
    assert list(line.get_xdata()) == [1, 2]
    assert list(line.get_ydata()) == [250.0, 200.0]


def test_plot_turn_profile_hides_the_split_panel_without_a_tokenizer_count():
    fig = plots.plot_turn_profile([("e31", [[_turn(100)]])])
    assert fig.axes[1].get_visible() is False
    fig = plots.plot_turn_profile([("e31", [[_turn(100, reasoning=80, content=5)]])])
    assert fig.axes[1].get_visible() is True


def test_plot_turn_profile_none_without_records():
    assert plots.plot_turn_profile([("e31", [])]) is None


def test_plot_dose_response_averages_seed_replicates():
    rows = [
        {"condition": "E2", "lam": 0.0, "losses": 0},
        {"condition": "E2", "lam": 1.0, "losses": 10},
        {"condition": "E2", "lam": 1.0, "losses": 20},
    ]
    fig = plots.plot_dose_response(rows, [("losses", "losses")])
    ax = fig.axes[0]
    mean_line = next(n for n in ax.lines if n.get_linestyle() == "-")
    assert list(mean_line.get_ydata()) == [0.0, 15.0]
    # Both replicates are still drawn, so the seed spread stays visible.
    scatter = [n for n in ax.lines if n.get_linestyle() == "None"]
    assert scatter and sorted(scatter[0].get_ydata()) == [10.0, 20.0]


def test_make_figures_writes_the_offtarget_and_stop_reason_views(tmp_path):
    runs = [
        _offtarget_run(tmp_path, "e30", 8, 2),
        _offtarget_run(tmp_path, "e31", 5, 5),
    ]
    names = {
        os.path.basename(p) for p in plots.make_figures(runs, str(tmp_path / "out"))
    }
    assert {"offtarget.png", "stop_reasons.png"} <= names


def test_make_figures_turn_profile_only_when_episodes_carry_turns(tmp_path):
    plain = _offtarget_run(tmp_path, "e30", 4, 1)
    names = {
        os.path.basename(p) for p in plots.make_figures([plain], str(tmp_path / "o1"))
    }
    assert "turn_profile.png" not in names
    withturns = _offtarget_run(tmp_path, "e31", 4, 1, turns=[[_turn(100), _turn(50)]])
    names = {
        os.path.basename(p)
        for p in plots.make_figures([plain, withturns], str(tmp_path / "o2"))
    }
    assert "turn_profile.png" in names
