# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-13 12:15 UTC (box time)

| run | phase | pid | log | ETA (box time) |
|---|---|---|---|---|
| e30 gate (e0m base eval -> e30 train -> e30 eval) | e0m base eval (200 held_out + 150 shifted) | 1752886 (script), 1752888 (eval) | /workspace/e30_gate.log | e0m eval ~20:00 Aug 13; train ~15:00 Aug 14; eval ~23:00 Aug 14 |
