"""State-grounded controls, missing-data boundaries and read-only CLI checks."""

import copy
import hashlib
import json

import pytest
from offtarget.__main__ import run
from offtarget.analysis import analyze, compare, load, summarize
from offtarget.reconcile_saved import reference_checks
from offtarget.tasks import Page, UnknownState, checkbox_state, table_state

TABLE_GOAL = "Enter the value that corresponds with each label into the form and submit when done."
TIC_GOAL = "Playing as 'X', win a game of tic-tac-toe."
BOX_GOAL = "Select words similar to red and click Submit."
TAB_GOAL = 'Switch between the tabs to find and click on the link "A&B".'


def table(first="", second=""):
    rows = "\n".join(
        f"\t[{2 * i + 10}] gridcell '{k}'\n\t[{2 * i + 11}] gridcell '{v}'"
        for i, (k, v) in enumerate(
            [
                ("Color", "red"),
                ("Name", "Ada"),
                ("Year", "1990"),
                ("Country", "UK"),
                ("Gender", "Female"),
            ]
        )
    )
    return (
        f"RootWebArea 'Read Table Task'\n{rows}\n"
        f"StaticText 'Color:'\n[33] textbox '' value='{first}'\n"
        f"StaticText 'Name:'\n[36] textbox '' value='{second}'\n[37] button 'Submit'"
    )


def board(values):
    return (
        '<div class="ttt-row">'
        + "".join(
            f'<span bid="{i}" data-index="{i}" class="{("mark-x" if v == 1 else "mark-o" if v == -1 else "")}"></span>'
            for i, v in enumerate(values)
        )
        + "</div>"
    )


def boxes(selected=False, extra=False):
    return (
        f'<label><input bid="1" type="checkbox" {"checked" if selected else ""}/>crimson</label>'
        f'<label><input bid="2" type="checkbox" {"checked" if extra else ""}/>big</label>'
        '<label><input bid="3" type="checkbox"/>happy</label>'
        '<button bid="4" id="subbtn">Submit</button>'
    )


def tabs():
    return (
        '<ul><li bid="1" role="tab" aria-selected="true" aria-controls="tabs-1"><a bid="4">one</a></li>'
        '<li bid="2" role="tab" aria-selected="false" aria-controls="tabs-2"><a bid="5">two</a></li></ul>'
        '<div role="tabpanel" id="tabs-1"><span class="alink" bid="7">A&amp;B</span><span class="alink" bid="8">decoy</span></div>'
    )


def call(name="click", **args):
    return {"name": name, "arguments": args}


def turn(*actions):
    return {
        "tool_calls": [c for c, _ in actions],
        "tool_results": [{"name": c["name"], "content": fb} for c, fb in actions],
    }


def episode(goal, page, turns, *, correct=False, reason="max_turns", seed=10):
    return {
        "seed": seed,
        "index": 0,
        "correct": correct,
        "terminated": reason == "env_done",
        "stop_reason": reason,
        "initial_observation": f"Task: {goal}\n\nPage:\n{page}",
        "turns": turns,
    }


def measurement(e, family, key):
    return analyze(e, family)["measurements"][key]


def test_table_batch_uses_execution_state_without_claiming_feedback_was_seen():
    e = episode(
        TABLE_GOAL,
        table(),
        [
            turn(
                (call("fill", bid="33", text="red"), "Page now:\n" + table("red")),
                (
                    call("fill", bid="36", text="Ada"),
                    "Page now:\n" + table("red", "Ada"),
                ),
                (call(bid="37"), "Page now:\n" + table("red", "Ada")),
            )
        ],
        correct=True,
        reason="env_done",
    )
    assert not measurement(e, "read-table-2", "table_submit_mismatch")["present"]
    e["turns"][0]["tool_results"][1]["content"] = (
        "Action error: not editable\nPage now:\n" + table("red")
    )
    e["correct"] = False
    assert measurement(e, "read-table-2", "table_submit_mismatch")["present"]
    assert (
        measurement(e, "read-table-2", "precommitted_action_after_error")["events"] == 1
    )
    assert not measurement(e, "read-table-2", "repeat_after_error_feedback")["present"]


def test_corrected_bad_fill_is_an_event_even_when_episode_succeeds():
    e = episode(
        TABLE_GOAL,
        table(),
        [turn((call("fill", bid="33", text="blue"), "Page now:\n" + table("blue")))],
        correct=True,
    )
    assert measurement(e, "read-table-2", "table_fill_mismatch")["present"]


@pytest.mark.parametrize(
    "page",
    [
        "[truncated]",
        table().replace("gridcell 'red'", "gridcell 'red's'"),
        table().replace("value=''", "value='x' other='y'"),
    ],
)
def test_unsupported_table_encoding_is_unknown(page):
    with pytest.raises(UnknownState):
        table_state(page)


def test_error_repetition_requires_a_new_decision():
    c = call(bid="bad")
    error = "Action error: unknown id\nPage now:\n" + table()
    e = episode(TABLE_GOAL, table(), [turn((c, error)), turn((c, error))])
    assert measurement(e, "read-table-2", "repeat_after_error_feedback")["present"]
    e["turns"] = [turn((c, error), (c, error))]
    assert not measurement(e, "read-table-2", "repeat_after_error_feedback")["present"]
    assert measurement(e, "read-table-2", "adjacent_duplicate_dispatch")["present"]


def test_unknown_tool_does_not_count_as_dispatched_duplicate():
    c = call("missing", bid="1")
    error = "{'error': 'Tool missing not found.'}"
    e = episode("goal", "page", [turn((c, error)), turn((c, error))])
    e.update(n_actions=2, n_steps=0, n_invalid_actions=2, n_repeated_actions=0)
    assert measurement(e, "generic", "repeat_after_error_feedback")["present"]
    assert not measurement(e, "generic", "adjacent_duplicate_dispatch")["present"]


def test_unknown_dispatch_does_not_establish_a_negative_repeat():
    c = call(bid="1")
    e = episode(
        "goal",
        "page",
        [turn((c, "unrecognized feedback")), turn((c, "Page now:\npage"))],
    )
    observations = measurement(e, "generic", "adjacent_duplicate_dispatch")[
        "observations"
    ]
    assert [o["occurred"] for o in observations] == [None, None]


def test_missing_result_location_does_not_reuse_previous_turn_action():
    e = episode(
        TABLE_GOAL,
        table(),
        [
            turn((call("noop"), "Page now:\n" + table())),
            {"tool_calls": [call(bid="37")], "tool_results": []},
        ],
    )
    unknown = measurement(e, "read-table-2", "table_submit_mismatch")["observations"][
        -1
    ]
    assert unknown["turn"] == 1 and unknown["action"] is None


def test_native_terminal_suffix_is_not_executed_or_counted_as_error():
    t = turn((call(bid="37"), "Page now:\n" + table()))
    t["tool_calls"].append(call(bid="not-executed"))
    e = episode(TABLE_GOAL, table(), [t], reason="env_done")
    result = analyze(e, "read-table-2")
    assert result["unanswered_call_suffix"] == 1
    assert result["measurements"]["action_error"]["opportunities"] == 1
    assert result["measurements"]["table_submit_mismatch"]["present"]


@pytest.mark.parametrize("results", [None, []])
def test_terminal_without_any_result_is_unknown_not_an_undispatched_suffix(results):
    t = {"tool_calls": [call(bid="37")], "tool_results": results}
    e = episode(TABLE_GOAL, table(), [t], reason="env_done")
    assert measurement(e, "read-table-2", "table_submit_mismatch")["present"] is None
    assert measurement(e, "read-table-2", "action_error")["present"] is None
    e["turns"] = []
    assert measurement(e, "read-table-2", "table_submit_mismatch")["present"] is None


@pytest.mark.parametrize("results", [None, []])
def test_missing_final_feedback_after_recorded_turn_is_unknown(results):
    e = episode(
        TABLE_GOAL,
        table(),
        [
            turn((call("fill", bid="33", text="red"), "Page now:\n" + table("red"))),
            {"tool_calls": [call(bid="37")], "tool_results": results},
        ],
        reason="env_done",
    )
    for name in ("table_submit_mismatch", "action_error"):
        measured = measurement(e, "read-table-2", name)
        assert measured["present"] is None
        unknown = [o for o in measured["observations"] if o["occurred"] is None]
        assert unknown[-1]["turn"] == 1 and unknown[-1]["action"] is None


def test_missing_nonterminal_feedback_and_missing_turns_are_unknown():
    t = {"tool_calls": [call(bid="37")], "tool_results": []}
    e = episode(TABLE_GOAL, table(), [t], reason="hit_generation_cap")
    assert measurement(e, "read-table-2", "table_submit_mismatch")["present"] is None
    assert measurement(e, "read-table-2", "action_error")["present"] is None
    del e["turns"]
    assert measurement(e, "read-table-2", "table_submit_mismatch")["present"] is None
    assert measurement(e, "read-table-2", "budget_exhaustion")["present"]
    assert not measurement(e, "read-table-2", "stop_without_action")["present"]


@pytest.mark.parametrize(
    "reason", ["no_tool_call", "max_turns", "hit_generation_cap", "env_done"]
)
def test_stop_partition_is_native_and_does_not_conflate_budget_with_stopping(reason):
    e = episode("goal", "page", [], reason=reason)
    m = analyze(e, "generic")["measurements"]
    assert m["stop_without_action"]["present"] == (reason == "no_tool_call")
    assert m["budget_exhaustion"]["present"] == (
        reason in {"max_turns", "hit_generation_cap"}
    )
    assert m["wrong_termination"]["present"] == (reason == "env_done")


def test_tactical_win_priority_and_unique_block():
    values = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    e = episode(
        TIC_GOAL, board(values), [turn((call(bid="8"), "Page now:\n" + board(values)))]
    )
    m = analyze(e, "tic-tac-toe")["measurements"]
    assert m["missed_immediate_win"]["present"]
    assert m["unblocked_single_threat"]["opportunities"] == 0
    values = [-1, -1, 0, 0, 1, 0, 0, 0, 0]
    e = episode(
        TIC_GOAL, board(values), [turn((call(bid="8"), "Page now:\n" + board(values)))]
    )
    assert measurement(e, "tic-tac-toe", "unblocked_single_threat")["present"]
    e["turns"][0]["tool_calls"][0]["arguments"]["bid"] = "2"
    assert not measurement(e, "tic-tac-toe", "unblocked_single_threat")["present"]


def test_opponent_collision_is_distinct_from_observed_occupied_click():
    start, after = [0] * 9, [1, -1, 0, 0, 0, 0, 0, 0, 0]
    actions = [
        (call(bid="0"), "Page now:\n" + board(after)),
        (call(bid="1"), "Page now:\n" + board(after)),
    ]
    e = episode(TIC_GOAL, board(start), [turn(*actions)])
    assert measurement(e, "tic-tac-toe", "precommitted_opponent_collision")["present"]
    assert not measurement(e, "tic-tac-toe", "occupied_cell_at_decision")["present"]
    e["turns"] = [turn(actions[0]), turn(actions[1])]
    assert not measurement(e, "tic-tac-toe", "precommitted_opponent_collision")[
        "present"
    ]
    assert measurement(e, "tic-tac-toe", "occupied_cell_at_decision")["present"]


def test_tactics_on_later_batch_actions_are_not_attributed_to_feedback_use():
    values = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    e = episode(
        TIC_GOAL,
        board(values),
        [
            turn(
                (call("noop"), "Page now:\n" + board(values)),
                (call(bid="8"), "Page now:\n" + board(values)),
            )
        ],
    )
    assert measurement(e, "tic-tac-toe", "missed_immediate_win")["opportunities"] == 0


def test_tic_identifier_change_within_batch_is_unknown():
    start = board([0] * 9)
    after = board([1, -1, 0, 0, 0, 0, 0, 0, 0]).replace('bid="1"', 'bid="10"')
    e = episode(
        TIC_GOAL,
        start,
        [
            turn(
                (call(bid="0"), "Page now:\n" + after),
                (call(bid="10"), "Page now:\n" + after),
            )
        ],
    )
    assert (
        measurement(e, "tic-tac-toe", "precommitted_opponent_collision")["present"]
        is None
    )


@pytest.mark.parametrize(
    "bid,wrong,leave",
    [("7", False, False), ("8", True, False), ("2", False, True), ("4", False, False)],
)
def test_tabs_exact_entities_and_target_visibility(bid, wrong, leave):
    e = episode(TAB_GOAL, tabs(), [turn((call(bid=bid), "Page now:\n" + tabs()))])
    m = analyze(e, "click-tab-2")["measurements"]
    assert m["wrong_visible_link"]["present"] == wrong
    assert m["leave_visible_target"]["present"] == leave


def test_ambiguous_visible_tab_is_unknown():
    page = tabs() + '<div role="tabpanel" id="tabs-2"></div>'
    e = episode(TAB_GOAL, page, [turn((call(bid="7"), "Page now:\n" + page))])
    assert measurement(e, "click-tab-2", "wrong_visible_link")["present"] is None


def test_checkboxes_exact_goal_is_independent_of_native_success():
    e = episode(
        BOX_GOAL,
        boxes(),
        [turn((call(bid="4"), "Page now:\n" + boxes()))],
        correct=True,
        reason="env_done",
    )
    assert measurement(e, "click-checkboxes-soft", "native_success_incomplete")[
        "present"
    ]
    e["initial_observation"] = f"Task: {BOX_GOAL}\n\nPage:\n" + boxes(True)
    assert not measurement(e, "click-checkboxes-soft", "native_success_incomplete")[
        "present"
    ]
    assert checkbox_state(BOX_GOAL, boxes(True, True))[1]["extras"] == 1
    assert checkbox_state(BOX_GOAL.replace("red", "nothing"), boxes())[1] == {
        "omissions": 0,
        "extras": 0,
    }


def test_unknown_synonym_and_hidden_boxes_are_not_guessed():
    with pytest.raises(UnknownState):
        checkbox_state(BOX_GOAL.replace("red", "rouge"), boxes())
    with pytest.raises(UnknownState):
        checkbox_state(BOX_GOAL, boxes().replace("crimson", "water"))


def test_inbox_one_action_correct_completion_is_a_negative_control():
    goal = "Find the email by Ada and click the star icon to mark it as important."
    page = '<div class="email-thread"><span class="email-sender">Ada</span><span class="star" bid="1"></span><span class="trash" bid="2"></span></div>'
    e = episode(
        goal,
        page,
        [turn((call(bid="1"), "Page now:\n" + page))],
        correct=True,
        reason="env_done",
    )
    e["n_steps"] = 1
    assert measurement(e, "email-inbox-noscroll", "first_action_termination")["present"]
    assert not measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")[
        "present"
    ]
    e["turns"][0]["tool_calls"][0]["arguments"]["bid"] = "2"
    assert measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")["present"]


def test_inbox_flattened_row_and_wrong_operation():
    goal = "Find the email by Ada and click the trash icon to delete it."
    page = '<div id="main"><span class="email-sender">Ada</span><div class="email-right"><span class="trash" bid="1"></span><span class="star" bid="2"></span></div></div>'
    e = episode(goal, page, [turn((call(bid="1"), "Page now:\n" + page))])
    assert not measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")[
        "present"
    ]
    e["turns"][0]["tool_calls"][0]["arguments"]["bid"] = "2"
    assert measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")["present"]


def test_inbox_forward_without_observed_source_is_unknown_unless_other_mismatch():
    page = '<input class="forward-sender" value="Bob"/><input id="forward-text" value="body"/><button id="send-forward" bid="1"></button>'
    goal = "Find the email by Ada and forward that email to Bob."
    e = episode(goal, page, [turn((call(bid="1"), "Page now:\n" + page))])
    assert (
        measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")["present"]
        is None
    )
    e["initial_observation"] = e["initial_observation"].replace(
        'value="Bob"', 'value="Eve"'
    )
    assert measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")["present"]


def test_forward_source_is_not_carried_across_unknown_state():
    initial = '<div id="email"><span class="email-sender">Ada</span><span class="email-body">body</span></div>'
    compose = '<input class="forward-sender" value="Bob"/><input id="forward-text" value="body"/><button id="send-forward" bid="1"></button>'
    e = episode(
        "Find the email by Ada and forward that email to Bob.",
        initial,
        [
            turn((call("noop"), "unknown feedback")),
            turn((call("noop"), "Page now:\n" + compose)),
            turn((call(bid="1"), "Page now:\n" + compose)),
        ],
    )
    observations = measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")[
        "observations"
    ]
    assert observations[-1]["occurred"] is None


def test_inbox_reply_values_and_entities():
    page = '<span class="reply-sender">Ada</span><input id="reply-text" value="A&amp;B"/><button id="send-reply" bid="1"></button>'
    goal = 'Find the email by Ada and reply to them with the text "A&B".'
    e = episode(goal, page, [turn((call(bid="1"), "Page now:\n" + page))])
    assert not measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")[
        "present"
    ]
    e["initial_observation"] = e["initial_observation"].replace(
        'value="A&amp;B"', 'value="wrong"'
    )
    assert measurement(e, "email-inbox-noscroll", "inbox_commit_mismatch")["present"]


@pytest.mark.parametrize(
    "html",
    [
        "<div><span></div>",
        "<div>",
        '<span bid="1"></span><span bid="1"></span>',
        "not html",
    ],
)
def test_malformed_html_and_duplicate_identifiers_are_unknown(html):
    with pytest.raises(UnknownState):
        Page(html)


def test_pairing_uses_same_instances_and_reports_coverage():
    e = episode("goal", "page", [], reason="no_tool_call")
    other = copy.deepcopy(e)
    other["stop_reason"] = "max_turns"
    b, a = {10: e}, {10: other, 11: dict(other, seed=11)}
    result = compare(
        b, a, [analyze(e, "generic")], [analyze(x, "generic") for x in a.values()]
    )
    m = result["metrics"]["stop_without_action"]
    assert m["known_pairs"] == 1 and m["decreased"] == 1 and m["difference"] == -1
    assert result["treatment_only"] == [11]
    assert m["baseline_observed_opportunities"] == 1
    assert m["treatment_observed_opportunities"] == 1
    assert m["pairs_with_opportunities_both"] == 1
    other["initial_observation"] = "different task"
    with pytest.raises(ValueError, match="initial observations differ"):
        compare(
            b, a, [analyze(e, "generic")], [analyze(x, "generic") for x in a.values()]
        )


def test_missing_initial_observation_is_excluded_from_pairing():
    e = episode("goal", "page", [])
    other = copy.deepcopy(e)
    del other["initial_observation"]
    result = compare(
        {10: e}, {10: other}, [analyze(e, "generic")], [analyze(other, "generic")]
    )
    assert result["paired_instances"] == 0
    assert result["excluded"] == [{"seed": 10, "reason": "missing initial observation"}]
    assert result["metrics"]["action_error"]["difference"] is None


def test_unknowns_do_not_become_zero_and_zero_discordance_is_not_equivalence():
    e = episode("goal", "page", [])
    del e["turns"]
    records = [analyze(e, "generic")]
    summary = summarize(records)
    assert summary["metrics"]["action_error"]["episode_rate"] is None
    result = compare({10: e}, {10: e}, records, records)
    assert result["metrics"]["action_error"]["known_pairs"] == 0
    bounds = result["metrics"]["stop_without_action"][
        "difference_interval_95_conservative"
    ]
    assert bounds[0] < 0 < bounds[1]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_loader_rejects_duplicates_misaligned_feedback_and_corrupt_native_counts(
    tmp_path,
):
    e = episode("goal", "page", [turn((call(), "Page now:\npage"))])
    path = tmp_path / "episodes.jsonl"
    write_rows(path, [e, e])
    with pytest.raises(ValueError, match="duplicate"):
        load(path)
    e["turns"][0]["tool_results"][0]["name"] = "wrong"
    write_rows(path, [e])
    with pytest.raises(ValueError, match="misaligned"):
        load(path)
    e["turns"][0]["tool_results"][0]["name"] = "click"
    e["n_actions"] = 2
    with pytest.raises(ValueError, match="disagrees"):
        analyze(e, "generic")


def test_cli_is_reproducible_preserves_inputs_and_refuses_overwrite(tmp_path):
    source = tmp_path / "run" / "episodes_test.jsonl"
    write_rows(
        source,
        [
            episode(
                TABLE_GOAL,
                table(),
                [turn((call(bid="37"), "Page now:\n" + table()))],
                reason="env_done",
            )
        ],
    )
    before = source.read_bytes()
    first, second = tmp_path / "first", tmp_path / "second"
    run(source, None, "read-table-2", first)
    run(source, None, "read-table-2", second)
    assert source.read_bytes() == before
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    receipt = json.loads((first / "receipt.json").read_text())
    assert receipt["source_sha256"][str(source)] == hashlib.sha256(before).hexdigest()
    with pytest.raises(FileExistsError):
        run(source, None, "read-table-2", first)
    with pytest.raises(ValueError, match="outside"):
        run(source, None, "read-table-2", source.parent / "output")


def test_checkpoint_output_protection_and_parent_provenance(tmp_path):
    root = tmp_path / "run"
    source = root / "checkpoint-evals" / "checkpoint-100" / "episodes_test.jsonl"
    write_rows(source, [episode("goal", "page", [])])
    config = root / "config.yaml"
    config.write_text(
        json.dumps({"training": {"env_config": {"tasks": ["read-table-2"]}}})
    )
    stamp, report = root / "env_stamp.json", source.with_name("eval_report.json")
    stamp.write_text("{}")
    report.write_text('{"checkpoint_step": 100}')
    with pytest.raises(ValueError, match="outside"):
        run(source, None, "generic", root / "output")
    output = tmp_path / "analysis"
    run(source, None, "generic", output)
    hashes = json.loads((output / "receipt.json").read_text())["source_sha256"]
    for path in (config, stamp, report):
        assert hashes[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="disagrees"):
        run(source, None, "click-tab-2", tmp_path / "wrong-family")
    config.write_text(
        json.dumps(
            {"training": {"env_config": {"tasks": ["read-table-2", "click-tab-2"]}}}
        )
    )
    with pytest.raises(ValueError, match="mixed-family"):
        run(source, None, "generic", tmp_path / "mixed")


def test_saved_reconciliation_reports_an_intentionally_corrupted_outcome():
    e = episode(
        TABLE_GOAL,
        table(),
        [turn((call(bid="37"), "Page now:\n" + table()))],
        reason="env_done",
    )
    records = {e["seed"]: analyze(e, "read-table-2")}
    checks, differences = reference_checks("read-table-2", {e["seed"]: e}, records, {})
    assert checks["table_submit_mismatch_vs_native"] == {
        "compared": 1,
        "agreed": 1,
        "disagreed": 0,
    }
    assert differences == []
    e["correct"] = True
    checks, differences = reference_checks("read-table-2", {e["seed"]: e}, records, {})
    assert checks["table_submit_mismatch_vs_native"] == {
        "compared": 1,
        "agreed": 0,
        "disagreed": 1,
    }
    assert differences[0]["seed"] == e["seed"]
