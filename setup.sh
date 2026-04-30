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
uv pip install unsloth https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl ipykernel ipywidgets scipy --torch-backend=auto
