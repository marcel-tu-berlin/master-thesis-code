# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-13 22:20 UTC (box time)

| run | phase | pid | log | ETA (box time) |
|---|---|---|---|---|
| e30 gate (e0m base eval -> e30 train -> e30 eval) | e30 train, step 46/150 at 492 s/it, healthy (reward 0.56 -> 0.63+, ISR 1.0) | 1752886 (script), 1809005 (train) | /workspace/e30_gate.log | train ~12:30 Aug 14; eval ~17:00 Aug 14 |
