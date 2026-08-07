# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-07 19:31 UTC (box time)

The E1/E2/E3 browsergym campaign, all three arms at `batch_size 4`, 150 steps,
one batch process running them in order. Parent pid `2309661`, log
`/workspace/e27bs4_e28bs4_e29bs4_batch.log`, per-arm logs at
`runs/<exp>/batch_{train,eval}.log`.

| Run | Phase | Pid | Started (UTC) | ETA (UTC) |
|---|---|---|---|---|
| e27bs4-browsergym-e1-baseline-qwen3-1_7b | train | 2309662 | Aug 7 19:30 | Aug 8 ~10:45 |
| e27bs4 | eval (2x100) | queued | - | Aug 8 ~13:20 |
| e28bs4-browsergym-e2-cosine-qwen3-1_7b | train | queued | - | Aug 9 ~04:30 |
| e28bs4 | eval (2x100) | queued | - | Aug 9 ~07:10 |
| e29bs4-browsergym-e3-nontermination-qwen3-1_7b | train | queued | - | Aug 9 ~22:20 |
| e29bs4 | eval (2x100) | queued | - | Aug 10 ~01:00 |

ETAs assume ~365 s/it (interpolated from 94 s/it at bs=1 and 712 s/it at bs=8 on
this env) and the 2h33m eval e27 measured over the same two splits. First step
timing lands ~19:40 Aug 7 and is what to check the whole schedule against.
