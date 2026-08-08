# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-08 15:09 UTC (box time)

The E1/E2/E3 browsergym campaign, all three arms at `batch_size 4`, 150 steps,
one batch process running them in order. Parent pid `2309661`, log
`/workspace/e27bs4_e28bs4_e29bs4_batch.log`, per-arm logs at
`runs/<exp>/batch_{train,eval}.log`.

| Run | Phase | Started (UTC) | ETA (UTC) |
|---|---|---|---|
| e28bs4-browsergym-e2-cosine-qwen3-1_7b | train, step 23/150 | Aug 8 12:55 | Aug 9 ~03:10 |
| e28bs4 | eval (2x100) | queued | Aug 9 ~05:45 |
| e29bs4-browsergym-e3-nontermination-qwen3-1_7b | train | queued | Aug 9 ~20:25 |
| e29bs4 | eval (2x100) | queued | Aug 9 ~23:00 |

e27bs4 is done and harvested, so its rows are gone. Measured cost per arm, which
is what the ETAs above now use: 14h55m of training at 358 s/it plus a 2h30m
two-split eval.
