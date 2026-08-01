"""Is BrowserGym reset(seed=N) deterministic, and does task_name switch per reset?

Both are hard requirements. GRPO repeats a prompt num_generations times across
rollout slots; if reset(seed=N) is not a pure function of N the slots get
different pages and the group is not a group. FinQA needed a patch for exactly
this.
"""
import sys, time, subprocess, os, urllib.request
sys.path.insert(0, "/workspace/OpenEnv/envs")
sys.path.insert(0, "/tmp")
from bg_probe import start_server
from browsergym_env import BrowserGymEnv

assert start_server("click-option"), "server failed"
env = BrowserGymEnv("http://localhost:8100").sync()


def page(seed=None, task=None):
    kw = {}
    if seed is not None:
        kw["seed"] = seed
    if task is not None:
        kw["task_name"] = task
    o = env.reset(**kw).observation
    goal = (getattr(o, "goal", "") or "").strip()
    ax = (getattr(o, "axtree_txt", "") or getattr(o, "text", "") or "").strip()
    return goal, ax


print("=== same seed twice ===")
g1, a1 = page(seed=7)
g2, a2 = page(seed=7)
print(f"  goal match: {g1 == g2}")
print(f"  axtree match: {a1 == a2}")
print(f"  goal(7): {g1[:70]}")

print("=== different seed ===")
g3, a3 = page(seed=8)
print(f"  goal differs from seed 7: {g3 != g1}")
print(f"  goal(8): {g3[:70]}")

print("=== seed 7 again, after an intervening reset ===")
g4, a4 = page(seed=7)
print(f"  still matches seed 7: {g4 == g1 and a4 == a1}")

print("=== task_name switch per reset ===")
try:
    g5, a5 = page(seed=7, task="enter-text")
    print(f"  switched ok: {g5[:70]}")
    print(f"  differs from click-option: {g5 != g1}")
    g6, _ = page(seed=7, task="click-option")
    print(f"  switched back, matches original: {g6 == g1}")
except Exception as e:
    print(f"  task switch FAILED: {type(e).__name__}: {e}")
