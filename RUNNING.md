# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-18 15:11 UTC (box time)

| Run | Phase | PID | Started (UTC) | ETA (UTC) | Notes |
|---|---|---|---|---|---|
| e35-browsergym-e3-nontermination-w025 | train 6/150 | - | Aug 18 14:19 | train ~Aug 19 10:40, eval done ~16:00 | e33-e36 lambda-sweep batch (batch pid 4184053, log `/workspace/e33_e36_batch.log`); e33 + e34 done and harvested Aug 18 |
| e36-browsergym-e3-nontermination-w05 | queued | - | - | ~Aug 20 evening | same batch; E3 arms run ~490 s/it like e32 |
