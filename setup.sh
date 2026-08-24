#! /usr/bin/env bash

set -e

# Everything below is repo-relative, so run from the repo root whatever the
# caller's cwd was. `ssh gpu-l4 'bash /workspace/master-thesis-code/setup.sh'`
# starts in $HOME, where a cwd-relative `rm -rf .venv` would delete someone
# else's venv and then fail on the missing lockfile.
cd "$(dirname "$0")"

# `deactivate` is a function of the *parent* shell and does not exist in this
# one, so calling it aborted the script (127) exactly when a venv was active.
# Dropping VIRTUAL_ENV is all the activate below needs.
unset VIRTUAL_ENV

if [ -d .venv ]; then
    rm -rf .venv
fi

# Pin the interpreter, not just the series: the lock was frozen against 3.12.3
# and `--python 3.12` floats to whatever 3.12.x the box happens to offer.
uv venv .venv --python 3.12.3
source .venv/bin/activate
# Every version comes from the lockfile, frozen off the GPU box. A floating
# install resolves a different stack every week, and a library change (trl's
# sequence_mask, transformers' tool-call rendering) is a silent change to what
# the runs measure. Upgrade deliberately: install, verify, re-freeze the lock.
# --torch-backend=cu130 matches the +cu130 wheels the lock pins; `auto` picks a
# variant per machine and can disagree with them.
uv pip install -r requirements.lock.txt --torch-backend=cu130

# Move a pinned clone onto its commit, refusing anything that would lose work.
# Untracked files are left alone (the box clone carries some); tracked
# modifications are not, because a checkout would silently revert them.
pin_clone() {
  local dir="$1" url="$2" commit="$3" before after
  if [ ! -d "$dir" ]; then
    git clone "$url" "$dir"
  fi
  # Only fetch when the pin is not already in the clone: an unconditional fetch
  # makes a re-run on an offline box die here, after .venv has been deleted.
  git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null || \
    git -C "$dir" fetch --quiet origin
  if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
    echo "$dir has uncommitted changes; commit or stash them before setup moves it to $commit" >&2
    exit 1
  fi
  before="$(git -C "$dir" rev-parse HEAD)"
  git -C "$dir" checkout --quiet "$commit"
  after="$(git -C "$dir" rev-parse HEAD)"
  if [ "$after" != "$commit" ]; then
    echo "$dir is at $after, not the pin $commit" >&2
    exit 1
  fi
  [ "$before" = "$after" ] || echo "$dir moved $before -> $after"
}

# OpenEnv env servers (reasoning_gym and friends) are not on PyPI - they live in
# the OpenEnv repo and run as local HTTP servers. Agentic-mode training launches
# one as a subprocess, so clone the repo. Its path is read from
# training.env_server.repo_path in the config (default /workspace/OpenEnv/envs).
# The clone carries the env servers and their wire contract, so its HEAD is as
# load-bearing as any pinned wheel. The pin lives in pipeline/ so it travels with
# the rsync that pushes the pipeline to the box, and so the training run can
# check the clone it is about to launch against it.
OPENENV_DIR="${OPENENV_DIR:-/workspace/OpenEnv}"
OPENENV_COMMIT="$(cat pipeline/OPENENV_COMMIT)"
# meta-pytorch/OpenEnv is a redirect to this repo; use the target directly, so a
# recreated meta-pytorch/OpenEnv could not quietly become the source.
pin_clone "$OPENENV_DIR" https://github.com/huggingface/OpenEnv "$OPENENV_COMMIT"

# Install the core from the same clone that provides the env servers, NOT the
# PyPI `openenv-core`. The two ship the same `openenv` import name, and a version
# skew between them fails at RUNTIME, not import time: the repo's env clients pass
# `metadata=` to `StepResult`, which older cores reject with a TypeError on the
# first reset. Installing both means whichever shadows the other decides the
# version, so `openenv-core` must be absent.
uv pip uninstall openenv-core 2>/dev/null || true
# --no-deps: the lock is the whole dependency set, and it already carries
# openenv's. Resolving them here instead would upgrade packages the lock just
# pinned - OpenEnv declares them as floating `>=` bounds.
uv pip install --no-deps -e "$OPENENV_DIR"

# browsergym needs two things pip does not install: playwright's browser binary
# (matched to the pinned playwright), and MiniWoB's HTML, which browsergym-miniwob
# does not ship. The env raises "core is not defined" at the first reset without
# the latter. Serving it is a launch step, not a setup one - see LAB_NOTES.md.
playwright install chromium
pin_clone "${MINIWOB_DIR:-/workspace/miniwob-plusplus}" \
  https://github.com/Farama-Foundation/miniwob-plusplus \
  eb59fed60fabe8951350275ba8650633b740013b
