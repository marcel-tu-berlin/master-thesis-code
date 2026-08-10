# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-10 11:02 UTC (box time)

| Run | Phase | pid | Started (UTC) | ETA | Log |
|---|---|---|---|---|---|
| e0-browsergym-base-qwen3-1_7b | eval (`--base-model`, 200 episodes) | 422924 | 10:58 | ~14:00 | `/workspace/e0_reeval.log` |

The E1/E2/E3 arms are harvested and their findings are written; the box copy of the
old e0 report (old seed scheme, pre-fix metrics) is archived at
`runs/_archive-e0-browsergym-base-oldseeds-20260803/`.
