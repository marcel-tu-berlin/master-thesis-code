#!/bin/sh
# The merge gate, as one command. Run by the Claude Stop and commit hooks, and
# by CI, so both execute exactly this.
#
# CPU-only and a few seconds: it covers the pipeline's logic and setup.sh's
# harness. Training and eval need the GPU box and are not part of the gate.
set -eu
cd "$(dirname "$0")/.."

VENV=.venv-test/bin/python
if [ ! -x "$VENV" ]; then
  echo "check: $VENV is missing. Create it with:" >&2
  echo "  uv venv .venv-test --python 3.12.6" >&2
  echo "  uv pip install --python .venv-test/bin/python -r requirements-test.lock.txt" >&2
  exit 1
fi

( cd pipeline && ../"$VENV" -m pytest tests/ -q )
bash tests/test_setup.sh
