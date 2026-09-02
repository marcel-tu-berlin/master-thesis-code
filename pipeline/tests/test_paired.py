import json

import pytest

from eval import paired


def _ep(seed, correct, n_tokens, terminated=True, stop_reason="env_done"):
    return {"index": seed, "seed": seed, "correct": correct, "reward": 1.0 if correct else 0.0,
            "n_tokens": n_tokens, "n_steps": 2, "terminated": terminated,
            "stop_reason": stop_reason, "tool_calls": ["click"]}


def _runs(episodes):
    return {e["seed"]: e for e in episodes}


# --- the two tests, computed by hand ---

def test_exact_binomial_p_matches_the_closed_form():
    # 0 of 14 discordant pairs in the arm's favour: 2 * 0.5**14.
    assert paired.exact_binomial_p(0, 14) == pytest.approx(2 * 0.5 ** 14)
    assert paired.exact_binomial_p(7, 14) == pytest.approx(1.0)


def test_exact_binomial_p_no_discordant_pairs_is_not_evidence():
    assert paired.exact_binomial_p(0, 0) == 1.0


def test_sign_test_drops_ties():
    # Three negatives, one positive, two ties -> the ties carry no direction.
    assert paired.sign_test_p([-5, -5, -5, 3, 0, 0]) == pytest.approx(
        paired.exact_binomial_p(3, 4))


def test_sign_test_all_ties_is_one():
    assert paired.sign_test_p([0, 0, 0]) == 1.0


def test_bootstrap_median_ci_brackets_the_median_and_is_deterministic():
    diffs = [-100, -90, -80, -70, -60, 10]
    med, lo, hi = paired.bootstrap_median_ci(diffs, n_bootstrap=2000)
    assert med == pytest.approx(-75.0)
    assert lo <= med <= hi
    assert paired.bootstrap_median_ci(diffs, n_bootstrap=2000) == (med, lo, hi)


def test_bootstrap_median_ci_empty_is_none():
    assert paired.bootstrap_median_ci([]) == (None, None, None)


# --- pairing ---

def test_compare_counts_flips_and_token_diffs_on_jointly_correct_only():
    base = _runs([_ep(1, True, 1000), _ep(2, True, 900), _ep(3, False, 4096)])
    arm = _runs([_ep(1, True, 400), _ep(2, False, 4096), _ep(3, True, 500)])
    c = paired.compare(base, arm)
    assert (c.wins, c.losses) == (1, 1)
    assert c.n_joint_correct == 1                  # only seed 1 is correct in both
    assert c.median_token_diff == pytest.approx(-600.0)
    assert c.mcnemar_p == pytest.approx(1.0)       # 1 vs 1 discordant


def test_compare_ignores_seeds_only_one_arm_ran():
    # A seed the arm never evaluated must not count as a failure - that is how an
    # interrupted split turns into a fabricated regression.
    base = _runs([_ep(1, True, 100), _ep(2, True, 100)])
    arm = _runs([_ep(1, False, 100)])
    c = paired.compare(base, arm)
    assert c.n_paired == 1 and c.losses == 1


def test_compare_reads_non_termination_but_leaves_it_absent_when_unrecorded():
    base = _runs([_ep(1, True, 100, terminated=False, stop_reason="max_turns"),
                  _ep(2, True, 100)])
    arm = _runs([_ep(1, True, 100), _ep(2, True, 100)])
    c = paired.compare(base, arm)
    assert c.base_nonterm == pytest.approx(0.5) and c.arm_nonterm == pytest.approx(0.0)

    old = {1: {"seed": 1, "correct": True, "n_tokens": 100}}
    assert paired.compare(old, old).base_nonterm is None


def test_stop_reason_transitions_follow_individual_episodes():
    base = _runs([_ep(1, True, 100), _ep(2, True, 100)])
    arm = _runs([_ep(1, False, 4096, terminated=False, stop_reason="hit_generation_cap"),
                 _ep(2, True, 100)])
    t = paired.stop_reason_transitions(base, arm)
    assert t[("env_done", "hit_generation_cap")] == 1 and t[("env_done", "env_done")] == 1


# --- training noise floor ---

def _log(n=40, slope=0.0, base=1.0, key="reward"):
    return [{"step": i, key: base + slope * i} for i in range(n)]


def test_training_tail_reports_slope_over_the_window_only():
    tail = paired.training_tail(_log(n=40, slope=0.01), keys=["reward"], window=30)
    assert tail["reward"]["slope"] == pytest.approx(0.01)
    assert tail["reward"]["n"] == 30
    assert tail["reward"]["sd"] > 0


def test_training_tail_flat_series_has_zero_noise():
    tail = paired.training_tail(_log(n=40), keys=["reward"], window=10)
    assert tail["reward"]["sd"] == pytest.approx(0.0)
    assert tail["reward"]["mean"] == pytest.approx(1.0)


def test_training_tail_skips_keys_without_two_points():
    assert paired.training_tail([{"step": 0, "reward": 1.0}], keys=["reward"]) == {}


def test_gradient_liveness_counts_steps_with_any_variance():
    log = [{"frac_reward_zero_std": v} for v in (1.0, 1.0, 0.5, 0.0)]
    live = paired.gradient_liveness(log)
    assert live["frac_steps_live"] == pytest.approx(0.5)
    assert live["mean_frac_zero_std"] == pytest.approx(0.625)


def test_gradient_liveness_absent_series_is_none():
    assert paired.gradient_liveness([{"reward": 1.0}]) is None


# --- config-derived condition and family labels ---

def test_arm_condition_reads_the_shaped_weight_as_lambda():
    cfg = {"rewards": {"env_reward": {"enabled": True, "weight": 1.0},
                       "token_length": {"enabled": True, "weight": 0.25}}}
    assert paired.arm_condition(cfg) == ("E2 cosine", 0.25)


def test_arm_condition_control_is_lambda_zero():
    cfg = {"rewards": {"env_reward": {"enabled": True, "weight": 1.0}}}
    assert paired.arm_condition(cfg) == ("control (task reward only)", 0.0)


def test_arm_condition_two_shaped_rewards_have_no_single_dose():
    cfg = {"rewards": {"token_length": {"enabled": True, "weight": 1.0},
                       "non_termination": {"enabled": True, "weight": 0.5}}}
    assert paired.arm_condition(cfg) == ("E4 combined", None)


def test_episode_families_uses_the_adapters_own_seed_mapping():
    from domains.browsergym.adapter import task_for_seed

    cfg = {"training": {"env": "browsergym", "env_config": {"tasks": ["a"]}},
           "eval": {"agentic": {"splits": [
               {"name": "shifted", "env_config": {"tasks": ["x", "y", "z"]}}]}}}
    fams = paired.episode_families(cfg, "shifted", [100, 101, 102])
    assert fams == {s: task_for_seed(["x", "y", "z"], s) for s in (100, 101, 102)}


def test_episode_families_empty_for_a_single_family_split():
    cfg = {"training": {"env": "browsergym", "env_config": {"tasks": ["click-menu-2"]}}}
    assert paired.episode_families(cfg, "held_out", [1, 2]) == {}


# --- disk boundary ---

def _write_run(tmp_path, name, split, episodes, config=None, train_log=None):
    d = tmp_path / name
    d.mkdir()
    (d / f"episodes_{split}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes))
    if config is not None:
        import yaml
        (d / "config.yaml").write_text(yaml.safe_dump(config))
    if train_log is not None:
        (d / "train_log.json").write_text(json.dumps(train_log))
    return str(d)


def test_load_episodes_keys_by_seed_and_keeps_the_last_duplicate(tmp_path):
    d = _write_run(tmp_path, "e1-x", "held_out",
                   [_ep(7, False, 10), _ep(8, True, 20), _ep(7, True, 30)])
    eps = paired.load_episodes(d, "held_out")
    assert sorted(eps) == [7, 8] and eps[7]["n_tokens"] == 30


def test_load_episodes_missing_split_raises(tmp_path):
    d = _write_run(tmp_path, "e1-x", "held_out", [_ep(1, True, 10)])
    with pytest.raises(FileNotFoundError):
        paired.load_episodes(d, "shifted")


def test_dose_rows_emit_the_control_once_per_condition(tmp_path):
    base = _write_run(tmp_path, "e30-c", "held_out", [_ep(i, True, 1000) for i in range(4)],
                      config={"rewards": {"env_reward": {"enabled": True, "weight": 1.0}}})
    arm = _write_run(tmp_path, "e33-cos", "held_out", [_ep(i, True, 400) for i in range(4)],
                     config={"rewards": {"token_length": {"enabled": True, "weight": 0.25}}})
    rows = paired.dose_rows(base, [arm])
    assert [r["lam"] for r in rows] == [0.0, 0.25]
    assert {r["condition"] for r in rows} == {"E2 cosine"}
    assert rows[0]["median_dtok"] == 0.0             # the control against itself
    assert rows[1]["median_dtok"][0] == pytest.approx(-600.0)


def test_dose_panel_takes_the_family_mapping_from_a_supplied_config(tmp_path):
    # A base-model eval never trained, so no config was frozen into its run dir
    # and its episodes cannot be grouped by family from its own files alone.
    cfg = {"training": {"env": "browsergym", "env_config": {"tasks": ["m"]}},
           "eval": {"agentic": {"splits": [
               {"name": "shifted", "env_config": {"tasks": ["x", "y"]}}]}}}
    d = _write_run(tmp_path, "e0m-base", "shifted",
                   [_ep(0, True, 10), _ep(1, False, 20), _ep(2, True, 30), _ep(3, True, 40)])
    assert paired.dose_panel(d, "shifted", "x")["family_acc"] is None
    # seeds 0 and 2 map to "x"; both correct.
    assert paired.dose_panel(d, "shifted", "x", cfg)["family_acc"] == pytest.approx(1.0)


def test_render_markdown_contains_a_row_per_comparison(tmp_path):
    base = _write_run(tmp_path, "e30-c", "held_out", [_ep(i, True, 1000) for i in range(4)],
                      train_log=[{"step": i, "reward": 1.0, "frac_reward_zero_std": 0.5}
                                 for i in range(5)])
    arm = _write_run(tmp_path, "e31-cos", "held_out", [_ep(i, i % 2 == 0, 400) for i in range(4)])
    c = paired.compare(paired.load_episodes(base, "held_out"),
                       paired.load_episodes(arm, "held_out"),
                       base_id="e30", arm_id="e31", split="held_out")
    md = paired.render_markdown([c], [base, arm], base, "held_out")
    assert "| e31 | held_out | all | 4 |" in md
    assert "frac steps live" in md and "`reward`" in md
