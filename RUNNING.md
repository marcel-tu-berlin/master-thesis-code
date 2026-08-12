# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-12 22:46 UTC (box time)

| run | phase | pid | started (UTC) | ETA | log |
|---|---|---|---|---|---|
| probe-b3-menuonly-lr5e5 (50-step B3 family-isolation probe, plan Phase 2 item 6) | train | 1259160 | 2026-08-12 22:46 | ~03:45 | /workspace/probe_tis/b3.log |

B3 probe from `docs/plans/no-arm-beats-e0-audit.md`: click-menu-2 only
(`tasks: [click-menu-2]`), lr 5e-5, token_truncate, same seed - even-seed
episodes are the same questions as b1b's menu episodes, so those pair; odd
seeds add new menu questions. Dump at /workspace/probe_tis/b3_dump.jsonl.
Read: does menu-2 climb when it owns the whole batch at the best-known lr.
B1b (lr 5e-5, pid 1014057) finished 2026-08-12 23:30-ish and is harvested:
monotone lr dose-response confirmed - 5e-5 vs 5e-6 paired +0.081 mean (36/5/9,
6.8 SE), 5e-5 vs 2e-5 +0.036 (3.4 SE), KL healthy at 0.05, menu-2 decline
halves at 5e-5. Verdict in LAB_NOTES.
