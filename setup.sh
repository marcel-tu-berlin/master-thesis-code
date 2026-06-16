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
