# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-09 06:41 UTC (box time)

The E1/E2/E3 browsergym campaign, all three arms at `batch_size 4`, 150 steps,
one batch process running them in order. Parent pid `2309661`, log
`/workspace/e27bs4_e28bs4_e29bs4_batch.log`, per-arm logs at
`runs/<exp>/batch_{train,eval}.log`.

| Run | Phase | Started (UTC) | ETA (UTC) |
|---|---|---|---|
| e29bs4-browsergym-e3-nontermination-qwen3-1_7b | train | Aug 9 06:36 | Aug 9 ~22:00 |
| e29bs4 | eval (2x100) | queued | Aug 10 ~00:35 |

e27bs4 and e28bs4 are done and harvested, so their rows are gone. Measured cost
per arm, which is what the ETA above uses: about 15h30m of training at 358-374
s/it plus a 2h30m two-split eval.
