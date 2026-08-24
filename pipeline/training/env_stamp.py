"""Record the stack a run phase actually executed against.

The lockfile says what a fresh setup installs; it does not say what was installed
on the box the day a run trained. A hand `pip install` between two arms of a
comparison is exactly the silent change this repo keeps paying for, and nothing
on disk would show it. The stamp is that record: it lives next to the frozen
config, so `runs/<exp>/` answers "which stack produced these numbers".

Never fatal - a missing package or a clone that is not a git checkout is recorded
as null. The pin check in env_server.verify_openenv_pin is the loud one.
"""

import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

from training.env_server import clone_head, openenv_pin

STAMP_FILE = "env_stamp.json"

# The packages whose behaviour a run's numbers depend on: the trainer, the
# generation path, the quantizer, the envs. Not the full lock - that is what
# requirements.lock.txt is for.
STAMPED_PACKAGES = (
    "trl", "transformers", "torch", "vllm", "peft", "accelerate", "datasets",
    "bitsandbytes", "reasoning-gym", "browsergym-core", "playwright", "numpy",
)


def collect_env_stamp(repo_envs_path=None) -> dict:
    packages = {}
    for name in STAMPED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    openenv = {"pin": None, "head": None}
    try:
        openenv["pin"] = openenv_pin()
    except OSError:
        pass
    if repo_envs_path:
        try:
            openenv["head"] = clone_head(repo_envs_path)
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "openenv": openenv,
        "packages": packages,
    }


def write_env_stamp(run_dir, phase, repo_envs_path=None) -> dict:
    """Merge this phase's stamp into runs/<exp>/env_stamp.json.

    Keyed by phase ("train" / "eval") because they can run weeks apart on
    different stacks, and overwriting would lose the one that trained.
    """
    path = os.path.join(run_dir, STAMP_FILE)
    stamps = {}
    if os.path.exists(path):
        with open(path) as f:
            stamps = json.load(f)
    stamps[phase] = collect_env_stamp(repo_envs_path)
    with open(path, "w") as f:
        json.dump(stamps, f, indent=2, sort_keys=True)
    return stamps[phase]
