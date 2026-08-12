# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-12 13:20 UTC (box time)

| run | phase | pid | started (UTC) | ETA | log |
|---|---|---|---|---|---|
| probe-b1-lr2e5-tokentruncate (50-step B1 lr probe, plan Phase 2 item 4) | train | 769822 | 2026-08-12 13:15 | ~18:30 | /workspace/probe_tis/b1.log |

B1 knob probe for `docs/plans/no-arm-beats-e0-audit.md`: e27bs4 recipe at
`learning_rate: 2e-5` (4x baseline) under `token_truncate`, same seed/dataset order
as the verification run so per-step rewards pair. Dump at
/workspace/probe_tis/b1_dump.jsonl. Read: reward slope, clip ratio, KL. The Phase-1
verification run (probe-tis-tokentruncate-50) finished 2026-08-11 ~15:40 UTC and is
harvested: NO-GO, fix alone gives no slope at baseline knobs (paired per-step diff
vs e27bs4 +0.009 mean, median 0.000).
