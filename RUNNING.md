# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-13 16:04 UTC (box time)

| run | phase | pid | log | ETA (box time) |
|---|---|---|---|---|
| e30 gate (e0m base eval -> e30 train -> e30 eval) | e30 train, 150 steps (e0m done 16:04, harvested: held_out 0.595) | 1752886 (script), 1809005 (train) | /workspace/e30_gate.log | train ~10:00 Aug 14; eval ~14:30 Aug 14 |
