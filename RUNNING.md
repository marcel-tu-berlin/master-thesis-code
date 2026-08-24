# Running

What is executing on the GPU box (`ssh gpu-l4`), and nothing else. Rows leave the
table once the results are harvested. Update rules are in CLAUDE.md; everything that
is not a live run belongs in `LAB_NOTES.md`.

Updated: 2026-08-24 20:00 UTC (box time)

| Run | Phase | pid | Started (UTC) | ETA | Notes |
|---|---|---|---|---|---|
| `probe-p2-base` -> `probe-p2-liger` -> `probe-p2-sleep` -> `probe-p2-iter2` | train, 50 steps each, sequential chain | 3215979 (chain), base arm 3215982 | 2026-08-24 20:00 | ~6.5h/arm at e30's 460 s/it, ~26h total (liger/sleep arms expected faster) | Phase-2 trainer-knob A/B (LAB_NOTES "recipe defaults"). Configs `/workspace/probe-p2-*.yaml`, logs `/workspace/probe-p2-*.log`, chain `/workspace/probe_p2_chain.log`. Read with `python -m probes.p2_compare`. |
