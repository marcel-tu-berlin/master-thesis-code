"""Offline figures from sealed inbox evidence. No environment or model calls."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "pipeline"))
from eval import plot_style as style  # noqa: E402

OUT = Path(__file__).parent
style.apply()
COLORS = {"e1": style.PRIMARY, "e2": style.PALETTE[1]}
LABELS = {"e1": "E1: task reward", "e2": "E2: task + length cost"}
STEPS = [100, 200, 300]
OPS = ["reply", "forward", "delete", "important"]
source = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    source[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text())


def rows(path):
    source[str(path.relative_to(ROOT))] = digest(path)
    return [json.loads(s) for s in path.read_text().splitlines()]


def components(episode):
    reasoning = sum(t["n_reasoning_tokens"] for t in episode["turns"])
    content = sum(t["n_content_tokens"] for t in episode["turns"])
    residual = episode["n_tokens"] - reasoning - content
    assert residual >= 0
    return [reasoning, content, residual]


runs, eps, metrics, audits, training = {}, {}, {}, {}, {}
verified = {}
for arm in ["e0", "e1", "e2"]:
    run = ROOT / "pipeline/runs" / f"{arm}-email-inbox-noscroll-s4021-c4096"
    runs[arm] = run
    receipt = read(run / "review.json")
    assert receipt["integrity"] == "pass"
    for path, expected in receipt["input_sha256"].items():
        assert digest(ROOT / path) == expected, path
        verified[path] = expected
    for step in [0] if arm == "e0" else STEPS:
        folder = (
            run if step in [0, 300] else run / f"checkpoint-evals/checkpoint-{step}"
        )
        key = f"{arm}_{step}"
        eps[key] = rows(folder / "episodes_held_out_inbox.jsonl")
        metrics[key] = read(folder / "eval_report.json")["results"]["held_out_inbox"]
        assert [e["seed"] for e in eps[key]] == list(range(4021100000, 4021100200))
        assert [e["initial_observation"] for e in eps[key]] == [
            e["initial_observation"] for e in eps["e0_0"]
        ]
        assert sum(e["correct"] for e in eps[key]) == metrics[key]["n_correct"]
        for e in eps[key]:
            assert sum(t["n_tokens"] for t in e["turns"]) == e["n_tokens"]
    if arm != "e0":
        audits[arm] = read(run / "technical_review.json")["evaluation_audits"]
        training[arm] = [r for r in read(run / "train_log.json") if "loss" in r]
        assert [r["step"] for r in training[arm]] == list(range(1, 301))

comparison = read(runs["e2"] / "initial_comparison.json")
behavior = read(runs["e2"] / "behavior_review.json")
joint = [
    (a, b)
    for a, b in zip(eps["e1_300"], eps["e2_300"], strict=True)
    if a["correct"] and b["correct"]
]
assert len(joint) == 182
means = np.array([np.mean([components(p[i]) for p in joint], axis=0) for i in [0, 1]])
action_deltas = Counter(b["n_actions"] - a["n_actions"] for a, b in joint)
turn_deltas = Counter(len(b["turns"]) - len(a["turns"]) for a, b in joint)
behavior_counts = {}
for key, episodes in eps.items():
    invalid_positions = Counter()
    repeated_positions = Counter()
    for e in episodes:
        previous, previous_turn = None, None
        assert sum(len(t["tool_results"]) for t in e["turns"]) == e["n_actions"]
        for ti, turn in enumerate(e["turns"]):
            # Evaluation stops dispatch immediately after env_done. Any remaining
            # emitted calls have no result and are not counted as attempted actions.
            dispatched = turn["tool_calls"][: len(turn["tool_results"])]
            if len(dispatched) < len(turn["tool_calls"]):
                assert e["stop_reason"] == "env_done" and ti == len(e["turns"]) - 1
            for ci, (call, result) in enumerate(
                zip(dispatched, turn["tool_results"], strict=True)
            ):
                assert call["name"] == result["name"]
                signature = (
                    call["name"],
                    json.dumps(call["arguments"], sort_keys=True),
                )
                # Unknown tools and argument-binding failures never reach the
                # environment; the existing repeated-action metric excludes them.
                if not result["content"].startswith("{'error':"):
                    if signature == previous:
                        repeated_positions[
                            "same_batch" if ti == previous_turn else "later_turn"
                        ] += 1
                    previous, previous_turn = signature, ti
                error = result["content"].startswith(("Action error", "{'error':"))
                if error:
                    invalid_positions[f"turn_{ti + 1}_call_{ci + 1}"] += 1
    assert sum(invalid_positions.values()) == sum(
        e["n_invalid_actions"] for e in episodes
    )
    assert sum(repeated_positions.values()) == sum(
        e["n_repeated_actions"] for e in episodes
    )
    behavior_counts[key] = {
        "invalid_action_positions": dict(invalid_positions),
        "consecutive_exact_repeats": dict(repeated_positions),
        "assistant_turn_distribution": dict(Counter(len(e["turns"]) for e in episodes)),
    }
assert behavior_counts["e2_300"]["invalid_action_positions"]["turn_1_call_2"] == 200
assert behavior_counts["e2_300"]["consecutive_exact_repeats"] == {"same_batch": 66}

diagnostics = {
    "scope": "Descriptive offline analysis of the reviewed seed-4021 corrected campaign",
    "sealed_inputs_verified": len(verified),
    "joint_correct_final": {
        "n": len(joint),
        "all_e2_shorter": all(b["n_tokens"] < a["n_tokens"] for a, b in joint),
        "components": [
            "parsed_reasoning",
            "parsed_content",
            "tool_call_framing_and_residual",
        ],
        "mean_components_e1_e2": means.tolist(),
        "e2_minus_e1_actions": dict(action_deltas),
        "e2_minus_e1_turns": dict(turn_deltas),
        "mean_actions_e1_e2": [
            float(np.mean([p[i]["n_actions"] for p in joint])) for i in [0, 1]
        ],
        "mean_turns_e1_e2": [
            float(np.mean([len(p[i]["turns"]) for p in joint])) for i in [0, 1]
        ],
    },
    "behavior_counts": behavior_counts,
}
(OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

figures = []


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf")
    figures.append(fig)


def finish(fig, title, subtitle, footnote):
    fig.suptitle(title, x=0.065, ha="left", y=0.98, fontsize=15, fontweight="bold")
    fig.text(0.065, 0.90, subtitle, color=style.MUTED, fontsize=9)
    fig.text(0.065, 0.025, footnote, color=style.MUTED, fontsize=8)


def arm_lines(ax, metric, scale=100, intervals=False):
    for arm, marker in [("e1", "o"), ("e2", "s")]:
        ms = [metrics[f"{arm}_{s}"] for s in STEPS]
        y = np.array([m[metric] * scale for m in ms])
        err = None
        if intervals:
            err = [
                y - [m[metric + "_ci_low"] * scale for m in ms],
                np.array([m[metric + "_ci_high"] * scale for m in ms]) - y,
            ]
        ax.errorbar(
            STEPS, y, yerr=err, color=COLORS[arm], marker=marker, label=LABELS[arm]
        )
    ax.set(xticks=STEPS, xlabel="Training update", xlim=(82, 318), ylim=(-2, 105))
    ax.grid(axis="y")


# 1. The opposing learning curves are the central result.
fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.3))
fig.subplots_adjust(
    left=0.08, right=0.97, bottom=0.14, top=0.84, hspace=0.50, wspace=0.27
)
finish(
    fig,
    "Compression improves as action errors increase",
    "Same 200 held-out inbox tasks at every checkpoint | E1 and E2 trained independently from the base",
    "Success: Wilson 95% intervals. Compression: paired median with 95% bootstrap intervals.\n"
    "Lines connect three measured checkpoints; they do not describe unmeasured updates. One training seed; update 300 is primary.",
)
arm_lines(axes[0, 0], "accuracy", intervals=True)
axes[0, 0].axhline(51.5, color=style.MUTED, ls=":", lw=1)
axes[0, 0].text(85, 47, "E0: 51.5%", color=style.MUTED, fontsize=8)
axes[0, 0].set(title="A  Task success", ylabel="Correct episodes (%)", ylim=(40, 107))
for arm, offsets in [("e1", [-15, -15, -16]), ("e2", [8, -15, 8])]:
    for step, offset in zip(STEPS, offsets, strict=True):
        value = metrics[f"{arm}_{step}"]["accuracy"] * 100
        axes[0, 0].annotate(
            f"{value:g}%",
            (step, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            color=COLORS[arm],
            fontsize=8,
        )
ax = axes[0, 1]
for step in STEPS:
    comp = comparison["comparisons"][f"e1_{step}_to_e2_{step}"]
    median, lo, hi = -np.array(comp["median_percent_ci95"])
    ax.errorbar(
        step,
        median,
        yerr=[[median - hi], [lo - median]],
        marker="s",
        color=COLORS["e2"],
    )
    ax.annotate(
        f"{median:.1f}%\nn={comp['n_joint_correct']}",
        (step, median),
        xytext=(17, 0) if step == 100 else (0, -31),
        textcoords="offset points",
        ha="left" if step == 100 else "center",
        fontsize=8,
    )
ax.axhline(10, color=style.MUTED, ls=":", lw=1)
ax.text(87, 13, "Declared 10% target", color=style.MUTED, fontsize=8)
ax.set(
    title="B  E2 token reduction relative to E1",
    ylabel="Paired reduction on jointly correct tasks (%)",
    xlabel="Training update",
    xticks=STEPS,
    xlim=(82, 318),
    ylim=(0, 105),
)
ax.grid(axis="y")
for ax, metric, title in [
    (axes[1, 0], "invalid_action_rate", "C  Episodes containing an invalid action"),
    (
        axes[1, 1],
        "repeated_action_rate",
        "D  Episodes containing an exact repeated action",
    ),
]:
    arm_lines(ax, metric)
    ax.set(title=title, ylabel="Episodes (%)")
    for arm, dy in [("e1", 8), ("e2", -16)]:
        value = metrics[f"{arm}_300"][metric] * 100
        ax.annotate(
            f"{value:g}%",
            (300, value),
            xytext=(-9, dy),
            textcoords="offset points",
            ha="right",
            color=COLORS[arm],
            fontsize=9,
        )
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="lower center",
    bbox_to_anchor=(0.52, 0.072),
    ncol=2,
)
save(fig, "01_learning_and_behavior")

# 2. Hold the question set and success condition fixed while explaining compression.
fig, axes = plt.subplots(
    1, 3, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1.45, 1, 1]}
)
fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.26, wspace=0.43)
finish(
    fig,
    "Token savings come from generated reasoning",
    "Final checkpoints | The same 182 questions solved correctly by both policies",
    "Parsed reasoning is generated text, not a measurement of internal computation. The residual includes tool calls and template framing.\n"
    "All 182 E2 responses are shorter. No jointly correct question uses fewer E2 actions; 148/182 use the same number of decision turns.",
)
bottom = np.zeros(2)
parts = [
    ("Parsed reasoning", "#777777"),
    ("Visible content", "#BFBFBF"),
    ("Tool calls + framing/residual", "#56B4E9"),
]
for j, (label, color) in enumerate(parts):
    axes[0].bar(
        [0, 1], means[:, j], bottom=bottom, width=0.55, color=color, label=label
    )
    bottom += means[:, j]
axes[0].set(
    title="A  Where the tokens went",
    xticks=[0, 1],
    xticklabels=["E1", "E2"],
    ylabel="Mean assistant tokens",
    ylim=(0, 1750),
)
axes[0].grid(axis="y")
for x, total in enumerate(bottom):
    axes[0].text(x, total + 35, f"{total:,.1f}", ha="center", fontweight="bold")
axes[0].text(
    0.48,
    1100,
    "Reasoning text\n1,386 -> 17 tokens",
    fontsize=9,
    ha="left",
    color=style.INK,
)
for ax, counts, title in [
    (axes[1], action_deltas, "B  Change in tool actions"),
    (axes[2], turn_deltas, "C  Change in decision turns"),
]:
    xs = sorted(counts)
    ax.bar(xs, [counts[x] for x in xs], width=0.6, color=COLORS["e2"])
    for x in xs:
        ax.text(x, counts[x] + 4, str(counts[x]), ha="center", fontsize=9)
    ax.set(
        title=title,
        xlabel="E2 minus E1",
        ylabel="Jointly correct questions",
        xticks=xs,
        ylim=(0, 185),
    )
    ax.grid(axis="y")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="lower center",
    bbox_to_anchor=(0.53, 0.14),
    ncol=3,
)
save(fig, "02_tokens_actions_and_turns")

# 3. Operation mix must not conceal a failed longer workflow.
fig, axes = plt.subplots(1, 2, figsize=(10.5, 6))
fig.subplots_adjust(left=0.14, right=0.9, top=0.79, bottom=0.19, wspace=0.35)
finish(
    fig,
    "The temporary quality loss is concentrated in longer workflows",
    "Correct tasks / tasks of that operation | Fixed questions and denominators across checkpoints",
    "Descriptive operation breakdown, not four independent equivalence tests. At update 200, E2 loses 38 E1 successes and gains 18.\n"
    "By update 300, E2 solves 60/60 forwards and 53/54 replies; its remaining failure drops a required period.",
)
for ax, arm in zip(axes, ["e1", "e2"], strict=True):
    values, cells = [], []
    for op in OPS:
        row, labels = [], []
        for step in STEPS:
            selected = [a for a in audits[arm][str(step)] if a["operation"] == op]
            n, correct = len(selected), sum(a["correct"] for a in selected)
            row.append(100 * correct / n)
            labels.append(f"{correct}/{n}\n{100 * correct / n:.1f}%")
        values.append(row)
        cells.append(labels)
    im = ax.imshow(values, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    for i in range(4):
        for j in range(3):
            ax.text(
                j,
                i,
                cells[i][j],
                ha="center",
                va="center",
                color="white" if values[i][j] < 57 else style.INK,
                fontsize=10,
            )
    ax.set(
        title=LABELS[arm],
        xticks=range(3),
        xticklabels=STEPS,
        yticks=range(4),
        yticklabels=["Reply (54)", "Forward (60)", "Delete (41)", "Important (45)"],
        xlabel="Training update",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="Success (%)")
save(fig, "03_operation_breakdown")

# 4. A deliberately simple, identified case shows the order of feedback and decisions.
fig, ax = plt.subplots(figsize=(11.8, 6.9))
fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.19)
ax.set(xlim=(0, 12), ylim=(0, 5))
ax.axis("off")
finish(
    fig,
    "A shorter successful trace can contain an avoidable failed action",
    "Final checkpoint, episode 0 | Request: find Floris's email and mark it important",
    "Boxes group calls emitted in one model turn. E2 chooses both first-turn calls before processing either result.\n"
    "The error does not change the saved visible page. This shows an action/state mismatch followed by recovery; intent is unobserved.",
)


def box(x, y, width, height, text, color, fill="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.10,rounding_size=0.07",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.3,
        )
    )
    ax.text(x + 0.16, y + height - 0.2, text, va="top", fontsize=10, linespacing=1.55)


ax.text(0.1, 4.65, "E1", color=COLORS["e1"], fontsize=14, fontweight="bold")
ax.text(
    0.1,
    4.1,
    "912 tokens\n864 reasoning\n2 actions",
    fontsize=10,
    va="top",
    linespacing=1.5,
)
box(
    1.8, 3.1, 3.55, 1.4, "Turn 1\nOpen Floris's email\nclick(28): success", COLORS["e1"]
)
box(
    7.2,
    3.1,
    3.75,
    1.4,
    "Turn 2\nClick the detail-page star\nclick(62): task success",
    COLORS["e1"],
)
ax.annotate(
    "",
    xy=(7.0, 3.77),
    xytext=(5.55, 3.77),
    arrowprops={"arrowstyle": "->", "color": style.MUTED, "lw": 1.2},
)
ax.text(6.27, 4.13, "Read new\npage", ha="center", fontsize=9, color=style.MUTED)
ax.text(0.1, 2.15, "E2", color=COLORS["e2"], fontsize=14, fontweight="bold")
ax.text(
    0.1,
    1.65,
    "81 tokens\n13 reasoning\n3 actions",
    fontsize=10,
    va="top",
    linespacing=1.5,
)
box(
    1.8,
    0.25,
    3.55,
    1.95,
    "Turn 1: two-call batch\n1. Open email: click(30), success\n2. Old list star: click(37), ERROR\n    The list is now hidden",
    COLORS["e2"],
)
box(
    7.2,
    0.45,
    3.75,
    1.4,
    "Turn 2\nClick the detail-page star\nclick(62): task success",
    COLORS["e2"],
)
ax.annotate(
    "",
    xy=(7.0, 1.15),
    xytext=(5.55, 1.15),
    arrowprops={"arrowstyle": "->", "color": style.MUTED, "lw": 1.2},
)
ax.text(6.27, 1.52, "Read page\nand error", ha="center", fontsize=9, color=style.MUTED)
fig.text(
    0.065,
    0.12,
    "General pattern: all 200 final E2 episodes begin with a valid email-opening click followed by an invalid action in the same batch.",
    fontsize=9,
    fontweight="bold",
)
save(fig, "04_same_task_different_strategy")

# 5. Logged task success and gradient availability, without confusing shaped reward with accuracy.
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
fig.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.22, wspace=0.26)
finish(
    fig,
    "Length shaping also changes how often groups provide a learning signal",
    "Training logs | Trailing 20-update means, shown from update 20 | These are sampled training rollouts",
    "Active group = nonzero composed-reward variation (1 - frac_reward_zero_std). It does not quantify gradient magnitude.\n"
    "This is not held-out success or an independent replication. The current pair cannot isolate length preference from extra gradient availability.",
)
for arm in ["e1", "e2"]:
    log = training[arm]
    for ax, key, title in [
        (axes[0], "reward/EnvReward/raw_mean", "A  Sampled task success"),
        (axes[1], "frac_reward_zero_std", "B  Groups with varying composed reward"),
    ]:
        y = np.array([r[key] for r in log])
        if key == "frac_reward_zero_std":
            y = 1 - y
        ax.plot(
            range(20, 301),
            np.convolve(y, np.ones(20) / 20, mode="valid") * 100,
            color=COLORS[arm],
            label=LABELS[arm],
        )
        ax.set(
            title=title,
            xlabel="Training update",
            ylabel="Percent",
            ylim=(0, 105),
            xlim=(0, 307),
            xticks=[0, 100, 200, 300],
        )
        ax.grid(axis="y")
        for step in [100, 200]:
            ax.axvline(step, color=style.FAINT, ls=":", lw=0.8)
axes[1].text(18, 12, "Whole-run active groups:\nE1: 49.7%     E2: 92.2%", fontsize=10)
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="lower center",
    bbox_to_anchor=(0.52, 0.12),
    ncol=2,
)
save(fig, "05_training_signal")

with PdfPages(OUT / "inbox_e1_e2_behavior.pdf") as pdf:
    for fig in figures:
        pdf.savefig(fig)
        plt.close(fig)

source[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
style_path = ROOT / "pipeline/eval/plot_style.py"
source[str(style_path.relative_to(ROOT))] = digest(style_path)
receipt = {
    "integrity": "pass",
    "analysis": "offline descriptive figures; no new experiment or endpoint selection",
    "sealed_input_count": len(verified),
    "inputs_sha256": source,
    "outputs_sha256": {
        p.name: digest(p)
        for p in sorted(OUT.iterdir())
        if p.suffix in [".png", ".pdf"] or p.name == "diagnostics.json"
    },
    "versions": {
        "python": sys.version,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
    },
}
(OUT / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(
    json.dumps(
        {
            "sealed_inputs_verified": len(verified),
            "paired": diagnostics["joint_correct_final"],
            "figures": len(figures),
        },
        indent=2,
    )
)
