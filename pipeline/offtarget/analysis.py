"""Trace validation, event extraction and paired off-target summaries."""

import ast
import json
from collections import Counter
from pathlib import Path

from eval.metrics import _wilson_ci
from eval.paired import exact_binomial_p
from offtarget.tasks import TASK_METRICS, UnknownState, detect, inbox_opened

GENERIC_METRICS = (
    "stop_without_action",
    "budget_exhaustion",
    "wrong_termination",
    "first_action_termination",
    "action_error",
    "adjacent_duplicate_dispatch",
    "repeat_after_error_feedback",
    "precommitted_action_after_error",
)
STOP_REASONS = {"env_done", "no_tool_call", "max_turns", "hit_generation_cap"}


def load(path):
    """Reject corrupt records; missing historical fields remain unavailable."""
    episodes = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        episode = json.loads(line)
        if not isinstance(episode, dict) or type(episode.get("seed")) is not int:
            raise ValueError(f"{path}:{line_number}: missing integer seed")
        seed = episode["seed"]
        if seed in episodes:
            raise ValueError(f"{path}:{line_number}: duplicate seed {seed}")
        if type(episode.get("correct")) is not bool:
            raise ValueError(f"{path}:{line_number}: missing native correctness")
        for key in (
            "n_steps",
            "n_actions",
            "n_invalid_actions",
            "n_repeated_actions",
            "n_tokens",
        ):
            value = episode.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"seed {seed}: invalid {key}")
        terminated, reason = episode.get("terminated"), episode.get("stop_reason")
        if terminated is not None and type(terminated) is not bool:
            raise ValueError(f"seed {seed}: invalid terminated flag")
        if reason is not None and (
            not isinstance(reason, str) or reason not in STOP_REASONS
        ):
            raise ValueError(f"seed {seed}: unsupported stop reason {reason}")
        if (
            reason is not None
            and terminated is not None
            and terminated != (reason == "env_done")
        ):
            raise ValueError(f"seed {seed}: contradictory termination fields")
        turns = episode.get("turns")
        if turns is not None:
            if not isinstance(turns, list):
                raise ValueError(f"seed {seed}: turns must be a list")
            for turn in turns:
                if not isinstance(turn, dict) or not isinstance(
                    turn.get("tool_calls"), list
                ):
                    raise ValueError(f"seed {seed}: malformed turn")
                results = turn.get("tool_results")
                if results is not None and (
                    not isinstance(results, list)
                    or len(results) > len(turn["tool_calls"])
                ):
                    raise ValueError(f"seed {seed}: invalid result prefix")
                for call in turn["tool_calls"]:
                    if (
                        not isinstance(call, dict)
                        or not isinstance(call.get("name"), str)
                        or (
                            call.get("arguments") is not None
                            and not isinstance(call["arguments"], dict)
                        )
                    ):
                        raise ValueError(f"seed {seed}: malformed call")
                for call, result in zip(
                    turn["tool_calls"], results or [], strict=False
                ):
                    if (
                        not isinstance(result, dict)
                        or result.get("name") != call["name"]
                        or not isinstance(result.get("content"), str)
                    ):
                        raise ValueError(f"seed {seed}: misaligned tool result")
        episodes[seed] = episode
    if not episodes:
        raise ValueError(f"No episodes in {path}")
    return episodes


def feedback_kind(content):
    """Return (error, dispatched); unknown formats are not assumed successful."""
    if content.startswith("Action error"):
        return True, True
    if content.startswith("Page now:\n"):
        return False, True
    if content.startswith(("{'error':", '{"error":')):
        try:
            value = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None, None
        if isinstance(value, dict) and isinstance(value.get("error"), str):
            return True, False
    return None, None


def analyze(episode, family):
    metrics = {name: [] for name in (*GENERIC_METRICS, *TASK_METRICS[family])}
    turn_index = position = None

    def add(name, occurred, evidence=None):
        metrics[name].append(
            {
                "turn": turn_index,
                "action": position,
                "occurred": occurred,
                "evidence": evidence or {},
            }
        )

    reason, terminated = episode.get("stop_reason"), episode.get("terminated")
    add("stop_without_action", None if reason is None else reason == "no_tool_call")
    add(
        "budget_exhaustion",
        None if reason is None else reason in {"max_turns", "hit_generation_cap"},
    )
    if terminated is not False:
        add("wrong_termination", None if terminated is None else not episode["correct"])
        steps = episode.get("n_steps")
        add(
            "first_action_termination",
            None if terminated is None or steps is None else steps == 1,
        )
    turns = episode.get("turns")
    initial = episode.get("initial_observation")
    goal, page = "", None
    if isinstance(initial, str) and "\n\nPage:\n" in initial:
        goal, page = initial.split("\n\nPage:\n", 1)
        goal = goal.removeprefix("Task: ")
    trace_metrics = (*GENERIC_METRICS[4:], *TASK_METRICS[family])
    unanswered = 0
    last_dispatch = previous = opened = None
    dispatch_history_known = True
    answered = dispatched = errors = repeats = 0
    precommitted = 0
    complete_trace = True
    if turns is None or (
        terminated is True and not any(t.get("tool_results") for t in turns)
    ):
        complete_trace = False
        for name in trace_metrics:
            add(name, None, {"reason": "No saved turns or no recorded terminal action"})
    for turn_index, turn in enumerate(turns or []):
        position = None
        decision_page = page
        calls, results = turn["tool_calls"], turn.get("tool_results")
        if results is None:
            results = []
        postterminal_suffix = (
            reason == "env_done" and turn_index == len(turns) - 1 and bool(results)
        )
        if len(results) < len(calls):
            unanswered += len(calls) - len(results)
            if not postterminal_suffix:
                complete_trace = False
                for name in trace_metrics:
                    add(
                        name,
                        None,
                        {"reason": "Calls without saved results; execution unknown"},
                    )
        for position, (call, result) in enumerate(zip(calls, results, strict=False)):
            content = result["content"]
            error, dispatch = feedback_kind(content)
            answered += 1
            precommitted += position > 0
            add("action_error", error, {"feedback": content.split("Page now:", 1)[0]})
            if error is not None:
                errors += error
            else:
                complete_trace = False
            if previous is not None:
                prior_call, prior_error, prior_turn, prior_position = previous
                if prior_error is True:
                    if prior_turn == turn_index:
                        add(
                            "precommitted_action_after_error",
                            True,
                            {"previous_action": prior_position},
                        )
                    else:
                        add(
                            "repeat_after_error_feedback",
                            call == prior_call,
                            {
                                "previous_turn": prior_turn,
                                "previous_action": prior_position,
                            },
                        )
                elif prior_error is None:
                    add(
                        "repeat_after_error_feedback"
                        if prior_turn != turn_index
                        else "precommitted_action_after_error",
                        None,
                    )
            if dispatch is True:
                dispatched += 1
                duplicate = last_dispatch == call if dispatch_history_known else None
                add("adjacent_duplicate_dispatch", duplicate)
                repeats += duplicate is True
                last_dispatch = call
                dispatch_history_known = True
            elif dispatch is None:
                add("adjacent_duplicate_dispatch", None)
                last_dispatch = None
                dispatch_history_known = False
            if family == "email-inbox-noscroll" and page is not None:
                try:
                    observed = inbox_opened(page)
                    if observed is not None:
                        opened = observed
                except UnknownState:
                    opened = None
            if error is not True:
                try:
                    if error is None:
                        raise UnknownState("Unrecognized tool feedback")
                    detect(
                        family,
                        goal,
                        page,
                        decision_page,
                        call,
                        position,
                        opened,
                        episode["correct"],
                        add,
                    )
                except UnknownState as exc:
                    for name in TASK_METRICS[family]:
                        add(name, None, {"reason": str(exc)})
            if "Page now:\n" in content:
                page = content.split("Page now:\n", 1)[1]
            elif dispatch is not False:
                page = None
                opened = None
            previous = call, error, turn_index, position
        if len(results) < len(calls) and not postterminal_suffix:
            page = None
            previous = last_dispatch = opened = None
            dispatch_history_known = False
    if turns is not None and complete_trace:
        for key, observed in (
            ("n_actions", answered),
            ("n_steps", dispatched),
            ("n_invalid_actions", errors),
            ("n_repeated_actions", repeats),
        ):
            if episode.get(key) is not None and episode[key] != observed:
                raise ValueError(
                    f"seed {episode['seed']}: {key} disagrees with saved actions"
                )
    measurements = {}
    for name, observations in metrics.items():
        positives = sum(o["occurred"] is True for o in observations)
        unknowns = sum(o["occurred"] is None for o in observations)
        measurements[name] = {
            "present": True if positives else None if unknowns else False,
            "events": positives,
            "opportunities": len(observations) - unknowns,
            "unknown_observations": unknowns,
            "observations": observations,
        }
    return {
        "seed": episode["seed"],
        "index": episode.get("index"),
        "native_correct": episode["correct"],
        "stop_reason": reason,
        "unanswered_call_suffix": unanswered,
        "answered_calls": answered if turns is not None else None,
        "precommitted_answered_calls": precommitted if turns is not None else None,
        "assistant_turns": len(turns) if turns is not None else None,
        "measurements": measurements,
    }


def summarize(records):
    output = {}
    for name in records[0]["measurements"]:
        measures = [r["measurements"][name] for r in records]
        known = [m["present"] for m in measures if m["present"] is not None]
        k, n = sum(known), len(known)
        output[name] = {
            "episodes_with_event": k,
            "known_episodes": n,
            "unknown_episodes": len(records) - n,
            "episode_rate": k / n if n else None,
            "wilson_95": list(_wilson_ci(k, n)) if n else None,
            "events": sum(m["events"] for m in measures),
            "opportunities": sum(m["opportunities"] for m in measures),
            "unknown_observations": sum(m["unknown_observations"] for m in measures),
        }
    return {
        "episodes": len(records),
        "answered_calls": sum(r["answered_calls"] or 0 for r in records),
        "precommitted_answered_calls": sum(
            r["precommitted_answered_calls"] or 0 for r in records
        ),
        "episodes_without_turns": sum(r["assistant_turns"] is None for r in records),
        "stop_reasons": dict(Counter(r["stop_reason"] or "unknown" for r in records)),
        "metrics": output,
    }


def compare(base_episodes, arm_episodes, base_records, arm_records):
    """Pair only identical reset observations, then use metric-wise known pairs."""
    common = sorted(base_episodes.keys() & arm_episodes.keys())
    paired, excluded = [], []
    for seed in common:
        left, right = (
            base_episodes[seed].get("initial_observation"),
            arm_episodes[seed].get("initial_observation"),
        )
        if not isinstance(left, str) or not left or not isinstance(right, str):
            excluded.append({"seed": seed, "reason": "missing initial observation"})
        elif left != right:
            raise ValueError(
                f"seed {seed}: initial observations differ; refusing instance pairing"
            )
        else:
            paired.append(seed)
    base, arm = (
        {r["seed"]: r for r in records} for records in (base_records, arm_records)
    )
    output = {}
    for name in base_records[0]["measurements"]:
        pairs = [
            (
                base[s]["measurements"][name]["present"],
                arm[s]["measurements"][name]["present"],
            )
            for s in paired
        ]
        known = [(b, a) for b, a in pairs if b is not None and a is not None]
        opportunity_pairs = [
            (
                base[s]["measurements"][name]["opportunities"],
                arm[s]["measurements"][name]["opportunities"],
            )
            for s, (b, a) in zip(paired, pairs, strict=True)
            if b is not None and a is not None
        ]
        n = len(known)
        increased = sum(not b and a for b, a in known)
        decreased = sum(b and not a for b, a in known)
        b_count, a_count = sum(b for b, _ in known), sum(a for _, a in known)
        # Bonferroni combination of marginal Wilson intervals: an approximate,
        # conservative difference interval, including when all pairs agree.
        bounds = None
        if n:
            b_lo, b_hi = _wilson_ci(b_count, n, ci=0.975)
            a_lo, a_hi = _wilson_ci(a_count, n, ci=0.975)
            bounds = [a_lo - b_hi, a_hi - b_lo]
        output[name] = {
            "known_pairs": n,
            "unknown_pairs": len(paired) - n,
            "baseline_events": b_count,
            "treatment_events": a_count,
            "baseline_observed_opportunities": sum(b for b, _ in opportunity_pairs),
            "treatment_observed_opportunities": sum(a for _, a in opportunity_pairs),
            "pairs_with_opportunities_both": sum(
                b > 0 and a > 0 for b, a in opportunity_pairs
            ),
            "increased": increased,
            "decreased": decreased,
            "difference": (a_count - b_count) / n if n else None,
            "difference_interval_95_conservative": bounds,
            "mcnemar_p_unadjusted": exact_binomial_p(increased, increased + decreased)
            if n
            else None,
        }
    transitions = Counter(
        (base[s]["stop_reason"], arm[s]["stop_reason"]) for s in paired
    )
    return {
        "paired_instances": len(paired),
        "excluded": excluded,
        "baseline_only": sorted(base_episodes.keys() - arm_episodes.keys()),
        "treatment_only": sorted(arm_episodes.keys() - base_episodes.keys()),
        "stop_transitions": [
            {"baseline": b, "treatment": a, "count": n}
            for (b, a), n in transitions.items()
        ],
        "metrics": output,
    }
