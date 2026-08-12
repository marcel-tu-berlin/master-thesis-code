# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-12 18:16 UTC (box time)

| run | phase | pid | started (UTC) | ETA | log |
|---|---|---|---|---|---|
| probe-b1b-lr5e5-tokentruncate (50-step 4b lr rung, plan Phase 2 item 4b) | train | 1014057 | 2026-08-12 18:15 | ~23:30 | /workspace/probe_tis/b1b.log |

4b rung of the lr ladder in `docs/plans/no-arm-beats-e0-audit.md`: e27bs4 recipe
at `learning_rate: 5e-5` (10x baseline, the LoRA-lr heuristic) under
`token_truncate`, same seed/dataset order as B1 and the verification run so
per-step rewards pair three ways. Dump at /workspace/probe_tis/b1b_dump.jsonl.
Read: paired reward diff vs B1, KL health (B1 hit max 0.013 - watch for
instability at 5e-5), menu-2 family trajectory. B1 (lr 2e-5, pid 769822)
finished 2026-08-12 18:09 and is harvested: GO signal, paired +0.046 mean /
+0.063 median over lr 5e-6 on identical prompts (30/11/9, 3.6 SE), diff growing
over training; verdict in LAB_NOTES.
