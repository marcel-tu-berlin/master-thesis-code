# Approve a bounded flight budget diagnostic

Decided: 2026-10-06. The user approved the diagnostic proposed in
`pipeline/runs/flight_s4022_scroll_findings.md` and requested launch.

This is a narrow exception to decision 0025 for flight qualification only:

| Setting | Diagnostic value |
| --- | ---: |
| Prompt allowance | 4,096 tokens |
| Whole trajectory, including tool feedback | 24,576 tokens |
| Model context | 28,672 tokens |
| Assistant turns | 32 |
| Training batch size / rollouts | 4 / 8 |
| Training micro-batch | 1 |

First run at most 16 greedy base-model episodes for `book-flight-nodelay`, using
visible HTML, fill and scroll. Use seed block 4023, evaluation offset 700000:
question seeds 4023700000-4023700015. These are disjoint from the scripted
qualification and retained campaigns. The run ID is
`flight-base-screen-s4023-c24576`.

After integrity and scientific review of that screen, up to three ordinary
training updates from the original base are authorized on fresh seed block
4024, using run ID `flight-native-check-s4024-c24576`. The user subsequently
requested immediate progression when the evaluation finishes, after being told
that the first nine episodes all failed and long-context training may not fit.
An intact, completed screen therefore admits the bounded resource test even
with zero successes. This is not approval for a training campaign. Preserve the
existing bf16, LoRA and training recipe, including the 0.24 vLLM reservation;
record a cache-initialization failure if that is the first limit reached. Measure
resource use; survival at short realized lengths does not prove long-sequence
capacity. No trained checkpoint from this diagnostic may warm-start a campaign.

Use only physical GPU 1. The combined diagnostic is capped at two allocated
GPU-hours, including setup and environment waits while the phase owns the GPU.
Stop on OOM, an integrity error or exhaustion of the remaining time. Retain
failed attempts and costs; do not silently retry or change budgets, precision,
rollout geometry or family. No full E0/E1/E2 campaign is authorized.

Freeze source, stack, config, exact seeds, commands and exit evidence. Diagnostic
configs live with their run operations, outside active campaign configs. The
active-campaign budget regression test remains unchanged. Existing table and
inbox budgets, source-bound results and artifacts remain intact. A hardware
request and any family fallback remain separate decisions under decision 0026.
