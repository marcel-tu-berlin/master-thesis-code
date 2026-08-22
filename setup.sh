#! /usr/bin/env bash

set -e

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

if [ -d .venv ]; then
    rm -rf .venv
fi

uv venv .venv --python 3.12
source .venv/bin/activate
# Every version comes from the lockfile, frozen off the GPU box. A floating
# install resolves a different stack every week, and a library change (trl's
# sequence_mask, transformers' tool-call rendering) is a silent change to what
# the runs measure. Upgrade deliberately: install, verify, re-freeze the lock.
uv pip install -r requirements.lock.txt --torch-backend=auto

# OpenEnv env servers (reasoning_gym and friends) are not on PyPI - they live in
# the meta-pytorch/OpenEnv repo and run as local HTTP servers. Agentic-mode
# training launches one as a subprocess, so clone the repo. Its path is read
# from training.env_server.repo_path in the config (default /workspace/OpenEnv/envs).
OPENENV_DIR="${OPENENV_DIR:-/workspace/OpenEnv}"
OPENENV_COMMIT="024eedc90305cc8bd7a5b44f44d1b987102e957b"  # v0.4.1-67-g024eedc
if [ ! -d "$OPENENV_DIR" ]; then
  git clone https://github.com/meta-pytorch/OpenEnv "$OPENENV_DIR"
fi
# Pin the clone too: it carries the env servers and their wire contract, so its
# HEAD is as load-bearing as any pinned wheel.
git -C "$OPENENV_DIR" fetch --quiet origin
git -C "$OPENENV_DIR" checkout --quiet "$OPENENV_COMMIT"

# Install the core from the same clone that provides the env servers, NOT the
# PyPI `openenv-core`. The two ship the same `openenv` import name, and a version
# skew between them fails at RUNTIME, not import time: the repo's env clients pass
# `metadata=` to `StepResult`, which older cores reject with a TypeError on the
# first reset. Installing both means whichever shadows the other decides the
# version, so `openenv-core` must be absent.
uv pip uninstall openenv-core 2>/dev/null || true
uv pip install -e "$OPENENV_DIR"
