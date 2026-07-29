# Running

What is executing on the GPU box (`ssh gpu-l4`). Update rules are in CLAUDE.md.

Updated: 2026-07-29 18:19 UTC (box time)

| exp | phase | status | started (box) | ends approx |
|-----|-------|--------|---------------|-------------|
| e24-poly-4k-env-only-qwen3-1_7b | train, 300 steps | done, 9h49m | 07-29 07:47 | 07-29 17:36 |
| e24-poly-4k-env-only-qwen3-1_7b | eval, 100 ep | running | 07-29 17:36 | ~07-29 19:30 |
| e25-poly-4k-cosine-w16-qwen3-1_7b | train, 300 steps | queued | after e24 eval | ~07-30 05:20 |
| e25-poly-4k-cosine-w16-qwen3-1_7b | eval, 100 ep | queued | after e25 train | ~07-30 07:00 |

Launcher, box pid 166878, cwd `/workspace/master-thesis-code/pipeline`:

```bash
python -m training.batch \
  configs/e24-poly-4k-env-only-qwen3-1_7b.yaml \
  configs/e25-poly-4k-cosine-w16-qwen3-1_7b.yaml \
  --train --eval
```

Logs: `runs/<exp>/batch_train.log` and `runs/<exp>/batch_eval.log` on the box.
