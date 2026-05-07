import glob
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    _MPL_MISSING_MSG = (
        "Warning: matplotlib not installed — eval plots will be skipped. "
        "Install with: pip install matplotlib"
    )


def _find_latest_trainer_state(run_dir: str) -> str | None:
    files = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if not files:
        return None

    def _step(p):
        try:
            return int(os.path.basename(os.path.dirname(p)).split("-")[-1])
        except ValueError:
            return 0

    return max(files, key=_step)


def plot_training_curves(run_dir: str, out_dir: str) -> None:
    if not _MPL_AVAILABLE:
        return

    state_path = _find_latest_trainer_state(run_dir)
    if state_path is None:
        return

    with open(state_path) as f:
        log_history = json.load(f).get("log_history", [])

    entries = [e for e in log_history if "step" in e and "loss" in e]
    if not entries:
        return

    steps = [e["step"] for e in entries]
    rewards = [e.get("reward") for e in entries]
    # 0.0 default is safe: orphaned std entries are filtered downstream by zip
    # against `y_vals` (skipped when reward is None).
    reward_stds = [e.get("reward_std", 0.0) for e in entries]
    kls = [e.get("kl") for e in entries]
    lengths = [e.get("completion_length") for e in entries]
    losses = [e.get("loss") for e in entries]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Curves", fontsize=14)

    def _panel(ax, y_vals, title, ylabel, color, std_vals=None):
        valid = [(s, y) for s, y in zip(steps, y_vals) if y is not None]
        if not valid:
            ax.set_visible(False)
            return
        xs, ys = zip(*valid)
        ax.plot(xs, ys, color=color, linewidth=1.5)
        if std_vals is not None:
            stds = [sv for sv, y in zip(std_vals, y_vals) if y is not None]
            ys_arr, stds_arr = np.array(ys), np.array(stds)
            ax.fill_between(xs, ys_arr - stds_arr, ys_arr + stds_arr, alpha=0.25, color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _panel(axes[0, 0], rewards, "Reward", "Reward", "steelblue", std_vals=reward_stds)
    _panel(axes[0, 1], kls, "KL Divergence", "KL", "coral")
    _panel(axes[1, 0], lengths, "Completion Length", "Tokens", "seagreen")
    _panel(axes[1, 1], losses, "Policy Loss", "Loss", "mediumpurple")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def plot_accuracy_bars(report_dict: dict, out_dir: str) -> None:
    if not _MPL_AVAILABLE:
        return

    split_labels = [
        ("id_split", "ID"),
        ("near_ood", "Near-OOD"),
        ("far_ood", "Far-OOD"),
        ("capability_floor", "Capability Floor"),
    ]

    results = report_dict.get("results", {})
    names, accs, err_lo, err_hi = [], [], [], []
    for key, label in split_labels:
        split = results.get(key)
        if not split or split.get("accuracy") is None:
            continue
        acc = split["accuracy"]
        ci = split.get("accuracy_ci", [acc, acc])
        names.append(label)
        accs.append(acc)
        err_lo.append(acc - ci[0])
        err_hi.append(ci[1] - acc)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 1.3)))
    y_pos = list(range(len(names)))
    bars = ax.barh(y_pos, accs, xerr=[err_lo, err_hi], capsize=4,
                   color="steelblue", alpha=0.8, error_kw={"linewidth": 1.5})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Accuracy by Split — {report_dict.get('experiment_id', '')}")
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(min(acc + 0.02, 1.02), bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eval_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def plot_token_distribution(ood_results, out_dir: str) -> None:
    if not _MPL_AVAILABLE:
        return

    id_metrics = ood_results.id_split
    if id_metrics is None or not id_metrics.raw:
        return

    correct_tokens = [r.n_tokens for r in id_metrics.raw if r.correct]
    incorrect_tokens = [r.n_tokens for r in id_metrics.raw if not r.correct]

    fig, ax = plt.subplots(figsize=(8, 4))
    if correct_tokens:
        ax.hist(correct_tokens, bins=30, alpha=0.6, color="seagreen",
                label=f"Correct (n={len(correct_tokens)})")
    if incorrect_tokens:
        ax.hist(incorrect_tokens, bins=30, alpha=0.6, color="tomato",
                label=f"Incorrect (n={len(incorrect_tokens)})")

    ax.set_xlabel("Token Count")
    ax.set_ylabel("Count")
    ax.set_title("Token Distribution by Correctness (ID Split)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "token_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def plot_difficulty_scatter(ood_results, out_dir: str) -> None:
    if not _MPL_AVAILABLE:
        return

    id_metrics = ood_results.id_split
    if id_metrics is None or not id_metrics.raw:
        return

    samples = [r for r in id_metrics.raw if r.difficulty is not None]
    if len(samples) < 5:
        return

    diffs = [r.difficulty for r in samples]
    tokens = [r.n_tokens for r in samples]
    colors = ["seagreen" if r.correct else "tomato" for r in samples]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(diffs, tokens, c=colors, alpha=0.6, edgecolors="none", s=30)

    pr = id_metrics.pearson_difficulty_length
    pv = id_metrics.pearson_p_value
    if pr is not None:
        ann = f"r = {pr:.3f}" + (f", p = {pv:.3f}" if pv is not None else "")
        ax.annotate(ann, xy=(0.05, 0.93), xycoords="axes fraction", fontsize=10)

    ax.legend(handles=[
        mpatches.Patch(color="seagreen", label="Correct"),
        mpatches.Patch(color="tomato", label="Incorrect"),
    ])
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Token Count")
    ax.set_title("Difficulty vs Token Count (ID Split)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "difficulty_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def plot_all(run_dir: str, ood_results, report_dict: dict, out_dir: str) -> None:
    if not _MPL_AVAILABLE:
        print(_MPL_MISSING_MSG)
        return
    os.makedirs(out_dir, exist_ok=True)
    for fn, args in [
        (plot_training_curves, (run_dir, out_dir)),
        (plot_accuracy_bars, (report_dict, out_dir)),
        (plot_token_distribution, (ood_results, out_dir)),
        (plot_difficulty_scatter, (ood_results, out_dir)),
    ]:
        try:
            fn(*args)
        except Exception as exc:
            print(f"Warning: {fn.__name__} failed: {exc}")
