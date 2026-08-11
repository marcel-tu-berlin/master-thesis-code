# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-11 10:21 UTC (box time)

| run | phase | pid | started (UTC) | ETA | log |
|---|---|---|---|---|---|
| probe-tis-tokentruncate-50 (50-step verification, plan Phase 1 item 3) | train | 508770 | 2026-08-11 10:21 | ~15:40 | /workspace/probe_tis/verify.log |

Verification run for `docs/plans/no-arm-beats-e0-audit.md`: the e27bs4 recipe with
`vllm_importance_sampling_mode: token_truncate`, instrumented per-episode dump at
/workspace/probe_tis/verify_dump.jsonl. Go/no-go = EnvReward slope (e27bs4 was flat
0.62-0.79 all 150 steps under sequence_mask). The 3-step diagnostic probe finished
10:12 UTC and is harvested; C1-C3 env-alignment results harvested to
/workspace/probe_tis/env_alignment.json + local scratchpad.
