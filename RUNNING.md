# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-11 09:47 UTC (box time)

| run | phase | pid | started (UTC) | ETA | log |
|---|---|---|---|---|---|
| probe-tis-filter (3-step instrumented probe, plan Phase 1 item 1) | train | launching | 2026-08-11 ~09:50 | ~10:20 | /workspace/probe_tis/probe.log |

Probe for `docs/plans/no-arm-beats-e0-audit.md`: 3 real e27bs4-recipe steps with
per-episode ISR/reward instrumentation, dump at /workspace/probe_tis/probe_dump.jsonl.
