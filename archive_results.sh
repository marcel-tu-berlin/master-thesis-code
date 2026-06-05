#!/usr/bin/env bash
#
# Pack experiment results into a gzipped tarball for easy transfer off a
# training box (mail / scp / rsync). Includes plots, configs, eval reports
# and logs; excludes trained model weights and caches (the multi-GB part).
#
# Usage:
#   ./archive_results.sh [OUTPUT.tar.gz] [RUN_DIR ...]
#
#   OUTPUT.tar.gz   Archive path (default: results-<timestamp>.tar.gz in CWD).
#   RUN_DIR ...     Specific run dirs to include (default: everything under
#                   $RUNS_DIR). Useful to grab a single experiment.
#
# Env:
#   RUNS_DIR        Results root (default: pipeline/runs).
#
# Examples:
#   ./archive_results.sh                              # all runs -> timestamped archive
#   ./archive_results.sh thesis-results.tar.gz        # all runs -> named archive
#   ./archive_results.sh e1.tar.gz pipeline/runs/e1-token-entropy
#
set -euo pipefail

RUNS_DIR="${RUNS_DIR:-pipeline/runs}"

# What "gigabytes" looks like: model weights + framework caches. Pruned by
# directory name (whole subtree skipped) and, defensively, by file extension
# so weights in merged/GGUF export dirs are dropped regardless of dir name.
PRUNE_DIRS=('checkpoint-*' 'unsloth_*')
WEIGHT_EXTS=(safetensors bin pt pth ckpt gguf onnx h5 msgpack)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '3,21p' "$0"; exit 0
fi

# --- args ---------------------------------------------------------------
out="${1:-results-$(date +%Y%m%d-%H%M%S).tar.gz}"
[[ $# -gt 0 ]] && shift || true
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac   # absolutise before cd

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "error: results dir not found: $RUNS_DIR (set RUNS_DIR=...)" >&2
  exit 1
fi

# Archive entries are relative to the results parent, so the tree extracts as
# e.g. runs/<exp>/... — run find from there to avoid GNU-only path rewriting.
parent="$(cd "$(dirname "$RUNS_DIR")" && pwd)"
base="$(basename "$RUNS_DIR")"
cd "$parent"

# Targets relative to $parent: explicit run dirs, or the whole results root.
targets=()
if [[ $# -gt 0 ]]; then
  for d in "$@"; do
    abs="$(cd "$d" 2>/dev/null && pwd)" || { echo "error: not a directory: $d" >&2; exit 1; }
    rel="${abs#"$parent"/}"
    [[ "$rel" == "$abs" ]] && { echo "error: $d is not under $RUNS_DIR" >&2; exit 1; }
    targets+=("$rel")
  done
else
  targets=("$base")
fi

# --- file list via find (POSIX semantics, identical on GNU/BSD) ----------
build_find() {
  local prune=() exts=() first=1 d ext
  for d in "${PRUNE_DIRS[@]}"; do
    [[ $first == 1 ]] && first=0 || prune+=(-o)
    prune+=(-name "$d")
  done
  for ext in "${WEIGHT_EXTS[@]}"; do exts+=(! -name "*.$ext"); done
  find "${targets[@]}" \
    \( -type d \( "${prune[@]}" \) -prune \) -o \
    \( -type f "${exts[@]}" -print0 \)
}

count="$(build_find | tr -cd '\0' | wc -c | tr -d ' ')"
if [[ "$count" -eq 0 ]]; then
  echo "warning: no result files found under $RUNS_DIR — nothing archived." >&2
  exit 1
fi

build_find | tar --null -czf "$out" -T -

# --- summary ------------------------------------------------------------
size="$(du -h "$out" | cut -f1)"
echo "wrote $out  ($size, $count files)"
echo "runs included:"
tar -tzf "$out" | sed -E "s#^($base/[^/]+)/.*#\1#" | sort -u | sed 's/^/  /'
