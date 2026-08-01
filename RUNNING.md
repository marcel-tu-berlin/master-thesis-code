# Running

What is executing on the GPU box (`ssh gpu-l4`). Rows leave the table once the
results are harvested. Update rules are in CLAUDE.md.

Updated: 2026-08-01 10:40 UTC (box time)

Nothing running. GPU free (0 MiB, 0% util).

e26 (finqa E1 qualification) finished and is harvested: train 300 steps in 8h08m,
eval 60 episodes at 08:38 UTC. Findings in
`pipeline/runs/e26_finqa_qualification_findings.md`. Verdict: finqa does not
qualify (accuracy 0/60, per-rollout training accuracy 1%, gradient on 10 of 300
steps); the off-target panel instrumentation does. Env choice is the open decision.

The e24/e25 4k pair finished and is harvested. Reports, configs and train logs are in
`pipeline/runs/e24-poly-4k-env-only-qwen3-1_7b/` and
`pipeline/runs/e25-poly-4k-cosine-w16-qwen3-1_7b/`; findings write-up in
`pipeline/runs/e24_e25_4k_pair_findings.md`; batch summary in
`pipeline/runs/batch_summary_20260730_042707.md`.

## Correction to the 2026-08-01 06:40 UTC entry

An earlier version of this file recorded the box as down and pid 171589 as
"almost certainly dead". Wrong: `ssh` refused on port 30236 for about six minutes
(06:40-06:46 UTC) and then recovered. The node has 79 days uptime and the training
process ran to completion throughout.
