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
uv pip install \
  "trl>=0.26" peft bitsandbytes accelerate \
  https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl \
  openenv-core reasoning-gym \
  "textarena>=0.6.1" nltk \
  datasets scipy matplotlib ipywidgets \
  --torch-backend=auto

# OpenEnv env servers (reasoning_gym and friends) are not on PyPI - they live in
# the meta-pytorch/OpenEnv repo and run as local HTTP servers. Agentic-mode
# training launches one as a subprocess, so clone the repo. Its path is read
# from training.env_server.repo_path in the config (default /workspace/OpenEnv/envs).
OPENENV_DIR="${OPENENV_DIR:-/workspace/OpenEnv}"
if [ ! -d "$OPENENV_DIR" ]; then
  git clone --depth 1 https://github.com/meta-pytorch/OpenEnv "$OPENENV_DIR"
fi

# TextArena word games (Wordle) need NLTK corpora. Pre-fetch so the first
# env-server start does not block on a download mid-training. Best-effort:
# textarena also self-downloads on first start if these are missing and the box
# has network. Exact corpora confirmed against the env at the first smoke.
python -c "import nltk; nltk.download('words'); nltk.download('brown')" || true
