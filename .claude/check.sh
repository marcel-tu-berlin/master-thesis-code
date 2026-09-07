#!/bin/sh
# The merge gate, as one command. Run by the Claude Stop and commit hooks, and
# by CI, so both execute exactly this.
#
# CPU-only and a few seconds: format, lint, types, the pipeline's logic tests and
# setup.sh's harness. Training and eval need the GPU box and are not part of the
# gate. Ordered cheapest-first so a formatting slip fails in under a second.
set -eu
cd "$(dirname "$0")/.."

VENV=.venv-test/bin
if [ ! -x "$VENV/python" ]; then
  echo "check: $VENV/python is missing. Create it with:" >&2
  echo "  uv venv .venv-test --python 3.12.6" >&2
  echo "  uv pip install --python .venv-test/bin/python -r requirements-test.lock.txt" >&2
  exit 1
fi

"$VENV/ruff" format --check pipeline
"$VENV/ruff" check pipeline
"$VENV/mypy" pipeline
( cd pipeline && ../"$VENV/python" -m pytest tests/ -q )
bash tests/test_setup.sh
