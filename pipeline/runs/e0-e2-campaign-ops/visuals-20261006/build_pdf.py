"""Assemble the table campaign's visual guide from preserved evidence only."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "pipeline"))
from eval import plot_style as style  # noqa: E402

OUT = Path(__file__).parent
OLD = OUT.parent / "deep-review-20260927"
source = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    source[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text())


def components(episode):
    reasoning = sum(t["n_reasoning_tokens"] for t in episode["turns"])
    content = sum(t["n_content_tokens"] for t in episode["turns"])
    return [reasoning, content, episode["n_tokens"] - reasoning - content]


verified = {}
episodes = {}
for arm in ["e0", "e1", "e2"]:
    run = ROOT / "pipeline/runs" / f"{arm}-read-table-2-s4016"
    receipt = read(run / "review.json")
    assert receipt["integrity"] == "pass"
    for path, expected in receipt["input_sha256"].items():
        assert digest(ROOT / path) == expected, path
        verified[path] = expected
    path = run / "episodes_held_out_table.jsonl"
    source[str(path.relative_to(ROOT))] = digest(path)
    episodes[arm] = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["seed"] for e in episodes[arm]] == list(range(4016100000, 4016100200))
    assert [e["initial_observation"] for e in episodes[arm]] == [
        e["initial_observation"] for e in episodes["e0"]
    ]

old_receipt = read(OLD / "analysis_receipt.json")
for path, expected in old_receipt["output_sha256"].items():
    assert digest(ROOT / path) == expected, path
    source[path] = expected
diagnostics = read(OLD / "diagnostics.json")
paired = diagnostics["paired_details"]
joint = [
    (a, b)
    for a, b in zip(episodes["e1"], episodes["e2"], strict=True)
    if a["correct"] and b["correct"]
]
assert len(joint) == paired["n"] == 197
means = np.array(
    [np.mean([components(pair[i]) for pair in joint], axis=0) for i in [0, 1]]
)
assert np.allclose(means, [paired["e1_mean_components"], paired["e2_mean_components"]])
actions = Counter(b["n_actions"] - a["n_actions"] for a, b in joint)
turns = Counter(len(b["turns"]) - len(a["turns"]) for a, b in joint)
one_turn = [
    pair
    for pair in joint
    if all(len(e["turns"]) == 1 and e["n_actions"] == 3 for e in pair)
]
assert len(one_turn) == 186
subset_median = np.median(
    [100 * (b["n_tokens"] / a["n_tokens"] - 1) for a, b in one_turn]
)
example = min(
    one_turn,
    key=lambda pair: (
        abs(100 * (pair[1]["n_tokens"] / pair[0]["n_tokens"] - 1) - subset_median),
        pair[0]["index"],
    ),
)
example_index = example[0]["index"]
assert example_index == 4
assert sum(e["correct"] for e in episodes["e0"]) == 43
assert sum(e["correct"] for e in episodes["e1"]) == 199
assert sum(e["correct"] for e in episodes["e2"]) == 198
assert [
    i
    for i, (a, b) in enumerate(zip(episodes["e1"], episodes["e2"], strict=True))
    if a["correct"] and not b["correct"]
] == [151, 182]

style.apply()
blue, orange = style.PALETTE[:2]
figures = []


def heading(fig, title, subtitle, footnote):
    fig.suptitle(title, x=0.065, y=0.98, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.065, 0.90, subtitle, color=style.MUTED, fontsize=9)
    fig.text(0.065, 0.035, footnote, color=style.MUTED, fontsize=8)


# Page 2: the same success-conditioned decomposition as the inbox guide.
fig, axes = plt.subplots(
    1, 3, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.4, 1, 1]}
)
fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.27, wspace=0.42)
heading(
    fig,
    "Token savings come mostly from generated reasoning",
    "Final checkpoints | Same 197 questions solved correctly by both policies | read-table-2, seed 4016",
    "Final paired median token reduction: 44.93% (95% bootstrap interval: 42.75%-47.33%).\n"
    "Parsed reasoning accounts for 99.07% of the mean saving. It measures emitted text, not internal computation.\n"
    "The residual includes tool-call JSON and template framing. E2 is shorter on 195/197 jointly successful questions.",
)
bottom = np.zeros(2)
for j, (label, color) in enumerate(
    [
        ("Parsed reasoning", "#777777"),
        ("Visible content", "#BFBFBF"),
        ("Tool calls + framing/residual", "#56B4E9"),
    ]
):
    axes[0].bar(
        [0, 1], means[:, j], bottom=bottom, width=0.55, color=color, label=label
    )
    bottom += means[:, j]
axes[0].set(
    title="A  Where the tokens went",
    xticks=[0, 1],
    xticklabels=["E1", "E2"],
    ylabel="Mean assistant tokens",
    ylim=(0, 560),
)
for i, total in enumerate(bottom):
    axes[0].text(i, total + 10, f"{total:.1f}", ha="center", fontweight="bold")
for ax, counts, title in [
    (axes[1], actions, "B  Change in tool actions"),
    (axes[2], turns, "C  Change in decision turns"),
]:
    xs = sorted(counts)
    ax.bar(xs, [counts[x] for x in xs], width=0.6, color=orange)
    for x in xs:
        ax.text(x, counts[x] + 4, str(counts[x]), ha="center")
    ax.set(
        title=title,
        xlabel="E2 minus E1",
        ylabel="Jointly correct questions",
        xticks=xs,
        ylim=(0, 212),
    )
for ax in axes:
    ax.grid(axis="y")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="lower center",
    bbox_to_anchor=(0.53, 0.15),
    ncol=3,
)
figures.append(fig)

# Page 3 replaces the inbox operation breakdown with this family's real variation.
fig, axes = plt.subplots(
    1, 2, figsize=(11.4, 6.5), gridspec_kw={"width_ratios": [1.1, 1]}
)
fig.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.26, wspace=0.32)
heading(
    fig,
    "The values vary; the form and action IDs stay fixed",
    "All 200 held-out pages are distinct | The same two-field lookup workflow repeats",
    "Counts include correct and incorrect episodes. At final evaluation E2 always batches two fills and Submit.\n"
    "The frozen prompt asks for one tool per turn, but both arms' runtime permits batches. This shared mismatch limits interpretation.\n"
    "These results do not establish adaptation to changed controls or decisions that require new information from tool feedback.",
)
axes[0].axis("off")
structural = diagnostics["structural"]
assert structural["textbox_ids"] == {"('33', '36')": 200}
assert structural["submit_ids"] == {"('37',)": 200}
facts = [
    (
        "What varies",
        f"{structural['unique_initial_observations']} distinct initial pages\n{structural['distinct_ordered_target_pairs']} ordered target-field pairs\n{structural['distinct_table_field_orders']} table field orders\n{structural['target_values']} distinct target label/value pairs",
    ),
    (
        "What stays fixed",
        "Textbox IDs: 33 and 36\nSubmit ID: 37\nTwo values to copy, then submit",
    ),
]
for y, (title, text) in zip([0.98, 0.42], facts, strict=True):
    axes[0].text(0, y, title, fontsize=12, fontweight="bold", va="top")
    axes[0].text(0, y - 0.10, text, fontsize=11, linespacing=1.6, va="top")
for arm, color, marker in [("e1", blue, "o"), ("e2", orange, "s")]:
    y = [
        diagnostics["behaviors"][f"{arm}_{s}"]["canonical_in_one_assistant_turn"] / 2
        for s in [100, 200, 300]
    ]
    axes[1].plot([100, 200, 300], y, color=color, marker=marker, label=arm.upper())
    axes[1].annotate(
        f"{int(y[-1] * 2)}/200",
        (300, y[-1]),
        xytext=(-5, 10 if arm == "e2" else -17),
        textcoords="offset points",
        ha="right",
        color=color,
    )
axes[1].set(
    title="Exact fill(33), fill(36), click(37)\nin one model turn",
    ylabel="Episodes (%)",
    xlabel="Training update",
    xticks=[100, 200, 300],
    xlim=(85, 318),
    ylim=(0, 110),
)
axes[1].grid(axis="y")
axes[1].legend(loc="lower right")
figures.append(fig)

# Page 4: one successful example and the complete set of final E2 regressions.
fig, axes = plt.subplots(
    2, 1, figsize=(11.8, 7.4), gridspec_kw={"height_ratios": [1.2, 1]}
)
fig.subplots_adjust(left=0.065, right=0.97, top=0.81, bottom=0.17, hspace=0.31)
heading(
    fig,
    "A shared action sequence, with less generated text",
    "Final checkpoint | Episode 4: copy Religion = Judaism and First name = Meagan, then submit",
    "Success example: nearest the median reduction among 186 pairs with one turn and three actions in both arms.\n"
    "Both final E2 regressions are included. Selection is descriptive, not random. Both policies can batch calls before fill feedback.\n"
    "Passing the declared 5-point success-preservation margin does not mean zero harm.",
)
for ax in axes:
    ax.axis("off")
for x, arm, color in [(0, "e1", blue), (0.53, "e2", orange)]:
    e = episodes[arm][example_index]
    assert e["correct"] and e["n_actions"] == 3 and len(e["turns"]) == 1
    assert [
        (c["name"], c["arguments"]) for t in e["turns"] for c in t["tool_calls"]
    ] == [
        ("fill", {"bid": "33", "text": "Judaism"}),
        ("fill", {"bid": "36", "text": "Meagan"}),
        ("click", {"bid": "37"}),
    ]
    title = f"{arm.upper()}: {e['n_tokens']} tokens, 1 turn, 3 actions"
    axes[0].text(x, 1, title, fontsize=12, fontweight="bold", color=color, va="top")
    lines = []
    for ti, turn in enumerate(e["turns"], 1):
        lines.append(f"Turn {ti}:")
        for call in turn["tool_calls"]:
            args = call["arguments"]
            value = f", '{args['text']}'" if "text" in args else ""
            lines.append(f"    {call['name']}({args['bid']}{value})")
    lines.append("Result: success")
    axes[0].text(
        x,
        0.80,
        "\n".join(lines),
        fontsize=10,
        fontfamily="monospace",
        va="top",
        linespacing=1.5,
    )
axes[1].text(
    0,
    1.02,
    "Both final E2 failures use valid controls and submit the wrong value",
    fontsize=11,
    fontweight="bold",
)
cells = []
for idx, required, wrong in [(151, "Arlyn", "Arlin"), (182, "Judaism", "Punjabi")]:
    a, b = episodes["e1"][idx], episodes["e2"][idx]
    calls = [c for t in b["turns"] for c in t["tool_calls"]]
    assert wrong in [c["arguments"].get("text") for c in calls]
    assert not b["correct"] and b["n_invalid_actions"] == 0
    cells.append([str(idx), required, wrong, f"{a['n_tokens']} / {b['n_tokens']}"])
table = axes[1].table(
    cellText=cells,
    colLabels=["Episode", "Required value", "E2 writes", "E1 / E2 tokens"],
    cellLoc="left",
    colLoc="left",
    bbox=[0, 0.40, 1, 0.49],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
for (row, _col), cell in table.get_celld().items():
    cell.set_linewidth(0.6)
    cell.set_edgecolor(style.FAINT)
    if row == 0:
        cell.set_facecolor("#F2F2F2")
        cell.set_text_props(weight="bold")
axes[1].text(
    0,
    0.22,
    "Episode 182 copies the Language value into Religion. E2 also fixes E1's invalid target in episode 123.",
    fontsize=9,
)
axes[1].text(
    0,
    0.07,
    "Final totals: E1 199/200, E2 198/200. Two regressions, one gain; regression upper bound 3.114% (< 5%).",
    fontsize=9,
)
figures.append(fig)

pdf_path = OUT / "read_table_2_e1_e2_behavior.pdf"
with tempfile.TemporaryDirectory() as temp:
    supplements = Path(temp) / "supplements.pdf"
    with PdfPages(supplements) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    combined = Path(temp) / "combined.pdf"
    subprocess.run(
        [
            "pdfunite",
            str(OLD / "evaluation.pdf"),
            str(supplements),
            str(OLD / "training.pdf"),
            str(combined),
        ],
        check=True,
    )
    info = subprocess.check_output(["pdfinfo", str(combined)], text=True)
    assert "Pages:           5" in info
    # All sources validate before replacing this derived deliverable.
    staged = OUT / ".read_table_2_e1_e2_behavior.pdf.tmp"
    staged.write_bytes(combined.read_bytes())
    os.replace(staged, pdf_path)

source[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
source["pipeline/eval/plot_style.py"] = digest(ROOT / "pipeline/eval/plot_style.py")
receipt = {
    "scope": "Offline five-page visualization of the retained seed-4016 campaign; no new experiments",
    "sealed_run_inputs_verified": len(verified),
    "input_sha256": source,
    "output_sha256": {pdf_path.name: digest(pdf_path)},
    "pages": [
        "Existing evaluation figure",
        "Token components, actions and turns",
        "Fixed interface and batching",
        "Successful example and both final regressions",
        "Existing training figure",
    ],
    "paired_action_deltas": dict(actions),
    "paired_turn_deltas": dict(turns),
    "successful_example_index": example_index,
    "successful_example_selection": "Nearest median percentage change in the 186 jointly correct one-turn/three-action pairs; tie by episode index",
    "pdf_hash_scope": "Exported bytes only; re-exported PDF metadata may change hashes without changing data or rendered content",
    "python_version": sys.version,
    "pdfunite_version": subprocess.check_output(
        ["pdfunite", "-v"], stderr=subprocess.STDOUT, text=True
    ).splitlines()[0],
    "matplotlib_version": matplotlib.__version__,
    "numpy_version": np.__version__,
}
(OUT / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(
    json.dumps(
        {
            "pdf": str(pdf_path.relative_to(ROOT)),
            "pages": 5,
            "sealed_run_inputs_verified": len(verified),
        }
    )
)
