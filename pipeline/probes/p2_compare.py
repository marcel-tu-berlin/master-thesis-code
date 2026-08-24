"""Pair Phase-2 probe arms against a base arm on their train_log.json.

Same seed means the same question order, so step k of every arm saw the same
prompts (num_iterations > 1 excepted: there step 2k-1 and 2k share one rollout
batch, and the pairing is by rollout batch). Reports per-arm mean step_time and
the paired per-step reward difference against the base arm with a sign count
and an SE, the readout the audit's probe ladder used.

    python -m probes.p2_compare runs/probe-p2-base runs/probe-p2-liger ...
"""
import json
import math
import os
import sys


def _steps(run_dir):
    """(all timed steps, generation steps). Under num_iterations > 1 the reuse
    steps log a step_time but no reward; they count for s/it, not pairing."""
    with open(os.path.join(run_dir, "train_log.json")) as f:
        log = json.load(f)
    timed = [x for x in log if "step_time" in x]
    return timed, [x for x in timed if "reward" in x]


def _paired(base, arm):
    n = min(len(base), len(arm))
    d = [arm[i]["reward"] - base[i]["reward"] for i in range(n)]
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    pos = sum(x > 0 for x in d)
    neg = sum(x < 0 for x in d)
    return n, mean, se, pos, neg, n - pos - neg


def main(argv):
    if len(argv) < 2:
        sys.exit("usage: p2_compare.py <base_run_dir> <arm_run_dir> [...]")
    _, base = _steps(argv[0])
    print(f"{'arm':22s} {'steps':>5s} {'s/it':>7s} {'reward':>7s} {'clip':>6s} {'len':>6s}  "
          f"paired d reward (pos/neg/tie, SE)")
    for path in argv:
        timed, arm = _steps(path)
        n = len(arm)
        sit = sum(x["step_time"] for x in timed) / len(timed)
        rew = sum(x["reward"] for x in arm) / n
        clip = sum(x.get("completions/clipped_ratio", 0.0) for x in arm) / n
        ln = sum(x.get("completions/mean_length", 0.0) for x in arm) / n
        line = f"{os.path.basename(path.rstrip('/')):22s} {n:5d} {sit:7.1f} {rew:7.3f} {clip:6.3f} {ln:6.0f}"
        if path != argv[0]:
            m, mean, se, pos, neg, tie = _paired(base, arm)
            line += f"  {mean:+.3f} ({pos}/{neg}/{tie}, SE {se:.3f}, n={m})"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
