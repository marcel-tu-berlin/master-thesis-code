"""Narrow events from the state saved before an action; no model or environment."""

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


class UnknownState(ValueError):
    """The saved observation does not support this measurement."""


@dataclass
class Node:
    tag: str
    attrs: dict
    parent: "Node | None" = field(default=None, repr=False)
    children: list["Node"] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def matches(self, key, value):
        actual = self.attrs.get(key, "") or ""
        return value in actual.split() if key == "class" else actual == value

    def all(self, key, value):
        return [n for n in self.walk() if n.matches(key, value)]

    def one(self, key, value):
        nodes = self.all(key, value)
        if len(nodes) != 1:
            raise UnknownState(f"Expected one {key}={value}, found {len(nodes)}")
        return nodes[0]

    def text(self):
        return " ".join(" ".join(p for n in self.walk() for p in n.parts).split())

    def ancestors(self):
        node: Node | None = self
        while node is not None:
            yield node
            node = node.parent


class Page(HTMLParser):
    """Strict parser for saved, pruned HTML, based on the existing inbox audit."""

    VOID = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", {})
        self.stack = [self.root]
        if not html.lstrip().startswith("<") or "[truncated]" in html:
            raise UnknownState("Missing, truncated, or non-HTML observation")
        self.feed(html)
        self.close()
        if len(self.stack) != 1:
            raise UnknownState("Unclosed HTML element")
        bids = [n.attrs["bid"] for n in self.root.walk() if "bid" in n.attrs]
        if len(set(bids)) != len(bids):
            raise UnknownState("Duplicate element identifiers")

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs), self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag != self.stack[-1].tag:
            raise UnknownState(f"Unmatched HTML closing tag: {tag}")
        self.stack.pop()

    def handle_data(self, data):
        self.stack[-1].parts.append(data)


TASK_METRICS = {
    "read-table-2": ("table_fill_mismatch", "table_submit_mismatch"),
    "email-inbox-noscroll": ("inbox_commit_mismatch",),
    "tic-tac-toe": (
        "occupied_cell_at_decision",
        "missed_immediate_win",
        "unblocked_single_threat",
        "precommitted_opponent_collision",
    ),
    "click-tab-2": ("wrong_visible_link", "leave_visible_target"),
    "click-checkboxes-soft": (
        "incomplete_checkbox_submit",
        "native_success_incomplete",
    ),
    "generic": (),
}
LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
SYNONYM_DATA = json.loads(
    Path(__file__).with_name("checkbox_synonyms.json").read_text()
)
MEANING = {word: i for i, group in enumerate(SYNONYM_DATA["groups"]) for word in group}


def board(page):
    root = Page(page).root
    cells = [n for n in root.walk() if "data-index" in n.attrs and n.tag == "span"]
    if len(cells) != 9 or {n.attrs["data-index"] for n in cells} != set(
        map(str, range(9))
    ):
        raise UnknownState("Incomplete tic-tac-toe board")
    values, bids = [0] * 9, {}
    for node in cells:
        i = int(node.attrs["data-index"])
        x, o = node.matches("class", "mark-x"), node.matches("class", "mark-o")
        if (x and o) or not node.attrs.get("bid"):
            raise UnknownState("Ambiguous board cell")
        values[i] = 1 if x else -1 if o else 0
        bids[node.attrs["bid"]] = i
    return values, bids


def winning_moves(values, player):
    return {
        i
        for i, v in enumerate(values)
        if v == 0
        and any(
            all((player if j == i else values[j]) == player for j in line)
            for line in LINES
        )
    }


def table_state(page):
    # shortcut: accept the recorded simple AXTree grammar only; add another
    # parser when a new observation format is actually used for this family.
    if "[truncated]" in page:
        raise UnknownState("Truncated table")
    cells = re.findall(r"^\s*\[\d+\] gridcell '([^'\n]*)'$", page, re.M)
    fields = re.findall(r"StaticText '([^'\n]*):'\n\s*\[(\d+)\] textbox", page)
    buttons = re.findall(r"\[(\d+)\] button 'Submit'", page)
    if len(cells) != 10 or len(fields) != 2 or len(buttons) != 1:
        raise UnknownState("Unsupported table observation")
    mapping = dict(zip(cells[::2], cells[1::2], strict=True))
    if len(mapping) != 5 or len({bid for _, bid in fields}) != 2:
        raise UnknownState("Ambiguous table labels or fields")
    expected, values = {}, {}
    for label, bid in fields:
        if label not in mapping:
            raise UnknownState("Form label missing from table")
        match = re.search(r"\[" + re.escape(bid) + r"\] textbox ''([^\n]*)", page)
        if match is None:
            raise UnknownState("Missing field state")
        suffix = match[1]
        value = re.fullmatch(r"(?: value='([^'\n]*)')?(?:, focused)?", suffix)
        if value is None:
            raise UnknownState("Unsupported field value encoding")
        expected[bid], values[bid] = mapping[label], value[1] or ""
    return expected, values, buttons[0]


def checkbox_state(goal, page):
    match = re.fullmatch(r"Select words similar to (.*?) and click Submit\.", goal)
    if match is None:
        raise UnknownState("Unsupported checkbox goal")
    words = match[1].split(", ")
    if "nothing" in words and words != ["nothing"]:
        raise UnknownState("Ambiguous empty-selection goal")
    if any(w not in MEANING for w in words if w != "nothing"):
        raise UnknownState("Goal word absent from frozen synonym groups")
    wanted = set() if words == ["nothing"] else {MEANING[w] for w in words}
    root = Page(page).root
    boxes = []
    for label in [n for n in root.walk() if n.tag == "label"]:
        inputs = label.all("type", "checkbox")
        if not inputs:
            continue
        word = label.text()
        if len(inputs) != 1 or word not in MEANING:
            raise UnknownState("Ambiguous checkbox or unknown synonym")
        boxes.append((MEANING[word], "checked" in inputs[0].attrs))
    if (
        not boxes
        or len(set(g for g, _ in boxes)) != len(boxes)
        or not wanted <= {g for g, _ in boxes}
    ):
        raise UnknownState("Incomplete or ambiguous checkbox set")
    button = root.one("id", "subbtn")
    if button.text() != "Submit" or not button.attrs.get("bid"):
        raise UnknownState("Missing checkbox submit control")
    missing = sum(g in wanted and not selected for g, selected in boxes)
    extra = sum(g not in wanted and selected for g, selected in boxes)
    return button.attrs["bid"], {"omissions": missing, "extras": extra}


def inbox_opened(page):
    root = Page(page).root
    opened = root.all("id", "email")
    if not opened:
        return None
    return (
        opened[0].one("class", "email-sender").text(),
        opened[0].one("class", "email-body").text(),
    )


def inbox_commit(goal, page, bid, opened):
    match = re.fullmatch(r"Find the email by (.*?) and (.*)", goal)
    if match is None:
        raise UnknownState("Unsupported inbox goal")
    sender, instruction = match.groups()
    reply = re.fullmatch(r'reply to them with the text "(.*)"\.', instruction)
    forward = re.fullmatch(r"forward that email to (.*)\.", instruction)
    operation = (
        "reply"
        if reply
        else "forward"
        if forward
        else "important"
        if instruction == "click the star icon to mark it as important."
        else "delete"
        if instruction == "click the trash icon to delete it."
        else None
    )
    if operation is None:
        raise UnknownState("Unsupported inbox operation")
    root = Page(page).root
    targets = root.all("bid", bid)
    if not targets:
        raise UnknownState("Clicked inbox element absent from saved page")
    for node in targets[0].ancestors():
        if node.attrs.get("id") in ("send-reply", "send-forward"):
            actual_op = node.attrs["id"].removeprefix("send-")
            if actual_op != operation:
                return True, {
                    "requested_operation": operation,
                    "actual_operation": actual_op,
                }
            if reply:
                actual_sender = root.one("class", "reply-sender").text()
                body = root.one("id", "reply-text").attrs.get("value", "")
                return actual_sender != sender or body != reply[1], {
                    "sender": actual_sender,
                    "expected_sender": sender,
                    "body": body,
                    "expected_body": reply[1],
                }
            recipient = root.one("class", "forward-sender").attrs.get("value", "")
            body = root.one("id", "forward-text").attrs.get("value", "")
            if forward is None:
                raise UnknownState("Missing forward goal")
            details = {
                "recipient": recipient,
                "expected_recipient": forward[1],
                "opened_source": opened,
                "expected_sender": sender,
                "body": body,
            }
            if recipient != forward[1] or (opened is not None and opened[0] != sender):
                return True, details
            return (None if opened is None else body != opened[1]), details
        actual_op = (
            "important"
            if node.matches("class", "star")
            else "delete"
            if node.matches("class", "trash")
            else None
        )
        if actual_op:
            containers = [
                p
                for p in node.ancestors()
                if p.matches("class", "email-thread") or p.matches("id", "email")
            ]
            if containers:
                actual_sender = containers[0].one("class", "email-sender").text()
            else:
                panels = [
                    p for p in node.ancestors() if p.matches("class", "email-right")
                ]
                if not panels or panels[0].parent is None:
                    raise UnknownState("Cannot associate inbox action with a sender")
                panel = panels[0]
                siblings = panel.parent.children
                index = siblings.index(panel)
                if index == 0 or not siblings[index - 1].matches(
                    "class", "email-sender"
                ):
                    raise UnknownState("Unsupported flattened inbox row")
                actual_sender = siblings[index - 1].text()
            return actual_op != operation or actual_sender != sender, {
                "operation": actual_op,
                "expected_operation": operation,
                "sender": actual_sender,
                "expected_sender": sender,
            }
    return None


def detect(family, goal, page, decision_page, call, position, opened, correct, add):
    """Add task events for one answered, non-error action. add captures location."""
    name, args = call["name"], call.get("arguments") or {}
    bid = str(args.get("bid", ""))
    if family == "generic":
        return
    if page is None:
        raise UnknownState("No reliable state before action")
    if family == "read-table-2":
        if (
            goal
            != "Enter the value that corresponds with each label into the form and submit when done."
        ):
            raise UnknownState("Unsupported table goal")
        expected, values, submit = table_state(page)
        if name == "fill" and bid in expected:
            add(
                "table_fill_mismatch",
                args.get("text") != expected[bid],
                {"expected": expected[bid], "submitted": args.get("text")},
            )
        if name == "click" and bid == submit:
            add(
                "table_submit_mismatch",
                values != expected,
                {"expected": expected, "values": values},
            )
    elif family == "click-checkboxes-soft":
        submit, details = checkbox_state(goal, page)
        if name == "click" and bid == submit:
            mismatch = details["omissions"] + details["extras"] > 0
            add("incomplete_checkbox_submit", mismatch, details)
            add("native_success_incomplete", mismatch and correct, details)
    elif family == "email-inbox-noscroll":
        if name == "click":
            result = inbox_commit(goal, page, bid, opened)
            if result is not None:
                add("inbox_commit_mismatch", *result)
    elif family == "tic-tac-toe":
        if goal != "Playing as 'X', win a game of tic-tac-toe.":
            raise UnknownState("Unsupported game goal")
        values, bids = board(page)
        if name != "click" or bid not in bids:
            return
        i = bids[bid]
        if position == 0:
            add(
                "occupied_cell_at_decision",
                values[i] != 0,
                {"board": values, "cell": i},
            )
            wins, threats = winning_moves(values, 1), winning_moves(values, -1)
            if values[i] == 0 and wins:
                add(
                    "missed_immediate_win",
                    i not in wins,
                    {"board": values, "cell": i, "winning_cells": sorted(wins)},
                )
            if values[i] == 0 and not wins and len(threats) == 1:
                add(
                    "unblocked_single_threat",
                    i not in threats,
                    {"board": values, "cell": i, "threat": sorted(threats)},
                )
        else:
            if decision_page is None:
                raise UnknownState("Missing board at batch commitment")
            decision, decision_bids = board(decision_page)
            if decision_bids.get(bid) != i:
                raise UnknownState("Board identifier changed within batch")
            add(
                "precommitted_opponent_collision",
                decision[i] == 0 and values[i] == -1,
                {
                    "board_at_commitment": decision,
                    "board_before_action": values,
                    "cell": i,
                },
            )
    elif family == "click-tab-2" and position == 0:
        match = re.fullmatch(
            r'Switch between the tabs to find and click on the link "(.*?)"\.', goal
        )
        if match is None:
            raise UnknownState("Unsupported tab goal")
        root = Page(page).root
        panels = root.all("role", "tabpanel")
        selected = root.all("aria-selected", "true")
        if (
            len(panels) != 1
            or len(selected) != 1
            or selected[0].attrs.get("aria-controls") != panels[0].attrs.get("id")
        ):
            raise UnknownState("Ambiguous visible tab")
        links = {n.attrs.get("bid"): n.text() for n in panels[0].all("class", "alink")}
        if name != "click":
            return
        if bid in links:
            add(
                "wrong_visible_link",
                links[bid] != match[1],
                {"clicked": links[bid], "target": match[1]},
            )
        tabs = root.all("role", "tab")
        for tab in tabs:
            if tab.all("bid", bid) and match[1] in links.values():
                add(
                    "leave_visible_target",
                    tab is not selected[0],
                    {
                        "target": match[1],
                        "selected_tab": selected[0].attrs.get("aria-controls"),
                    },
                )
