"""Deterministic task generation for the repl env.

repl ships no task corpus, so we mint one. Kept in its own module so both the
domain (build_seed_dataset) and the adapter (reset) derive the SAME task from a
seed without a circular import - training and eval both call reset(seed=N), so
the task must be a pure function of the seed, identical across rollout slots.

The smoke family is arithmetic-over-a-list: the model must write code to compute
a reduction and print FINAL(<number>). Intentionally simple - the point is a real
automatic exact-match reward and real token slack (terse vs verbose solutions),
not task difficulty. Upgrade target: a richer task source (e.g. reasoning_gym).
"""
import random as _random

_OPS = {"sum": sum, "maximum": max, "minimum": min}
_OP_NAMES = ["sum", "maximum", "minimum"]


def make_task(seed: int):
    """seed -> (context, task_prompt, expected_answer). Pure function of seed."""
    s = int(seed)
    rng = _random.Random(s)
    op = _OP_NAMES[s % len(_OP_NAMES)]
    k = 4 + (s % 5)
    nums = [rng.randint(1, 99) for _ in range(k)]
    answer = _OPS[op](nums)
    task = (
        f"Compute the {op} of these integers: {nums}. "
        f"Then print the result as FINAL(<number>)."
    )
    return "", task, str(answer)
