#!/usr/bin/env python3
"""Apply local patches to a meta-pytorch/OpenEnv clone so its env servers meet
the pipeline's contract. Idempotent: re-running is a no-op once applied.

Why this exists: the OpenEnv env servers live in a separate clone (not vendored
into this repo), so a re-clone or `git pull` there reverts these edits. Keep this
script as the canonical, re-appliable record and run it after (re)cloning.

Patched envs:
- finqa (determinism): make reset(seed) deterministic. As shipped, finqa picks
  questions via an unseeded global shuffle, so reset() ignores `seed`. That breaks
  both the EnvDomain seed->question contract and GRPO's same-question-per-group
  invariant on a shared server (each concurrent rollout would otherwise get a
  different question). The patch makes reset(seed=N) select questions[N % len].
- finqa (concurrency): pass MAX_CONCURRENT_ENVS into create_app. Shipped finqa
  omits max_concurrent_envs, so create_app defaults it to 1 and the server closes
  every WebSocket session after the first - fatal for the pipeline's one-server /
  many-rollout-slot model (the second client's reset dies with ConnectionClosedOK).
  reasoning_gym/textarena already wire the env var through; finqa did not.

Usage: python patch_openenv.py [OPENENV_ENVS_PATH]   (default /workspace/OpenEnv/envs)
"""
import sys
from pathlib import Path

MARKER = "PATCHED (master-thesis-code)"

FINQA_OLD = "        question = self._get_next_question()\n"
FINQA_NEW = (
    "        # " + MARKER + ": seed-addressable question selection.\n"
    "        # Shipped finqa ignores `seed` (unseeded shuffle); GRPO needs\n"
    "        # reset(seed=N) to be a deterministic function of N, identical\n"
    "        # across rollout slots sharing one server.\n"
    "        if seed is not None:\n"
    "            question = self.questions[seed % len(self.questions)]\n"
    "        else:\n"
    "            question = self._get_next_question()\n"
)

FINQA_APP_OLD = (
    "app = create_app(\n"
    "    _env_factory, FinQACallToolAction, CallToolObservation, env_name=\"finqa_env\"\n"
    ")\n"
)
FINQA_APP_NEW = (
    "# " + MARKER + ": pass MAX_CONCURRENT_ENVS so one server serves all rollout\n"
    "# slots. Shipped finqa omits max_concurrent_envs, so create_app defaults it to\n"
    "# 1 and the server closes every WS session after the first (the pipeline runs\n"
    "# one server for many concurrent rollout-slot clients). reasoning_gym/textarena\n"
    "# already wire this env var through.\n"
    "app = create_app(\n"
    "    _env_factory, FinQACallToolAction, CallToolObservation, env_name=\"finqa_env\",\n"
    "    max_concurrent_envs=int(os.getenv(\"MAX_CONCURRENT_ENVS\", \"8\")),\n"
    ")\n"
)

# Mark FinQAEnvironment concurrency-safe. create_app refuses max_concurrent_envs>1
# unless the env class sets this flag; finqa is safe because the server builds a
# fresh, isolated FinQAEnvironment per session (own mcp_client + state + questions)
# - reasoning_gym/textarena set the same flag. Anchored after the class docstring.
FINQA_CONC_OLD = '    """\n\n    def __init__(\n'
FINQA_CONC_NEW = (
    '    """\n\n'
    "    SUPPORTS_CONCURRENT_SESSIONS = True  # " + MARKER + ": per-session env isolation\n\n"
    "    def __init__(\n"
)


def patch_file(path: Path, old: str, new: str, sentinel: str) -> str:
    # Idempotency is per-patch (via `sentinel`, a phrase unique to this patch's
    # new text), not per-file: a file may carry more than one patch (finqa's
    # finqa_environment.py gets both the determinism and concurrency-flag edits).
    if not path.is_file():
        return f"FAIL (missing): {path}"
    text = path.read_text()
    if sentinel in text:
        return f"skip (already patched): {path.name} [{sentinel[:34]}]"
    if old not in text:
        return f"FAIL (anchor not found): {path.name} [{sentinel[:34]}]"
    path.write_text(text.replace(old, new, 1))
    return f"patched: {path.name} [{sentinel[:34]}]"


def main() -> int:
    envs = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/OpenEnv/envs")
    finqa_env_py = envs / "finqa_env" / "server" / "finqa_environment.py"
    results = [
        patch_file(finqa_env_py, FINQA_OLD, FINQA_NEW, "seed-addressable question selection"),
        patch_file(envs / "finqa_env" / "server" / "app.py", FINQA_APP_OLD, FINQA_APP_NEW,
                   "pass MAX_CONCURRENT_ENVS so one server"),
        patch_file(finqa_env_py, FINQA_CONC_OLD, FINQA_CONC_NEW, "per-session env isolation"),
    ]
    for r in results:
        print(r)
    return 1 if any(r.startswith("FAIL") for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
