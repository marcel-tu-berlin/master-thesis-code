# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-13 05:03 UTC (box time)

| run | phase | pid | started (UTC) | ETA | log |
|---|---|---|---|---|---|
| probe-b2-temp12-lr5e5 (50-step B2 exploration probe, plan Phase 2 item 5) | train | 1505327 | 2026-08-13 05:02 | ~10:15 | /workspace/probe_tis/b2.log |

B2 probe from `docs/plans/no-arm-beats-e0-audit.md`: rollout temperature 1.2,
otherwise b1b unchanged (mixed tasks, lr 5e-5, token_truncate, same seed - pairs
step-for-step with b1b). The A6 handling is active and confirmed in the log
("probe: vllm.LLM patched with logprobs_mode=processed_logprobs"). Dump at
/workspace/probe_tis/b2_dump.jsonl. Read: reward slope vs b1b,
frac_reward_zero_std, and the ISR mean staying ~1.0 (drift off 1.0 = wrong
logprob pairing = void). B3 (menu-only, pid 1259160) finished 2026-08-13 ~04:45
and is harvested: menu-2 CLIMBS when it owns the batch - 0.588 -> 0.756 across
buckets, OLS +0.0038/step, the audit's first positive within-run slope. Verdict
in LAB_NOTES.
