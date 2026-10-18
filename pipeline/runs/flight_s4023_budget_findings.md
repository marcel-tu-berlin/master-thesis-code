# Flight policy and training-budget diagnostic

Completed 2026-10-06 under decision 0027. Both phases used Qwen3-1.7B,
bf16, visible HTML, fill and scroll, a 4,096-token prompt allowance,
24,576-token trajectory allowance including feedback, 28,672-token context,
and 32 assistant turns. Only physical GPU 1, an L4, was used.

## Base-policy screen

Run: `flight-base-screen-s4023-c24576`. Greedy base policy, no adapter,
16 development questions, seeds 4023700000-4023700015.

| Measurement | Result |
| --- | ---: |
| Correct | 0/16 |
| Wilson 95% interval | 0.0%-19.4% |
| Logged trajectory-cap stops | 9 |
| Logged turn-cap stops | 4 |
| Logged no-tool-call stops | 3 |
| Total actions / invalid / repeated | 317 / 54 / 174 |
| Assistant tokens per episode, range | 2,151-22,446 |
| Sampled peak device memory | 9,432 MiB (9.21 GiB) |
| Controller allocation | 4,590.73 seconds (1.27520 GPU-hours) |
| Process exit | 0 |

The screen did not establish a competent baseline for this interface. Repeated
Search clicks, attempts to fill the read-only date field, and missed city
autocomplete selections occur before successful result comparison. For example,
seed 4023700000 fills the date field, receives an action error, then repeatedly
clicks Search until the turn cap. Seed 4023700001 opens the calendar but then
repeatedly scrolls the date input without selecting the required date.

These are observations of an untrained policy, not effects of reward shaping.
The count of repeated actions is descriptive, not proof of feedback neglect.
There are no successful-response efficiency measurements in this screen.
Zero greedy successes does not establish that stochastic training can never
discover a success, but it does not meet the project's qualification band.

One episode, seed 4023700009, lasted 1,120.94 seconds. This exceeds the installed
MiniWoB default episode clock of 1,000 seconds. Its final assistant turn generated
21,631 tokens without an action; the evaluator logged a trajectory-cap stop, and
no subsequent tool result reported termination. Therefore the logged stop
reason must not be treated as proof that the browser clock was irrelevant.
The other episodes lasted less than 1,000 seconds. This caveat does not turn the
screen into a qualified baseline.

## Native training attempt

Run: `flight-native-check-s4024-c24576`. The user explicitly authorized the
resource test after the screen, including if all base episodes failed. The
original bf16/LoRA recipe was preserved: batch size 4, eight rollouts,
micro-batch 1, three planned updates, vLLM reservation 0.24, no sleep mode or
Liger, fresh base initialization, seed block 4024.

The run failed while initializing vLLM, before a rollout group or optimizer
update. The runtime reported that one 28,672-token request requires **3.06 GiB
of KV cache**, while the configured reservation leaves **1.83 GiB**. It estimated
a maximum model length of **17,104 tokens** for that cache allocation. This is
context length, not an assistant-generation allowance or proven training limit.

| Measurement | Result |
| --- | ---: |
| Completed updates | 0/3 |
| Process exit | 1, ValueError during cache initialization |
| Sampled peak device memory | 7,072 MiB (6.91 GiB) |
| Controller allocation | 36.08 seconds (0.01002 GPU-hours) |
| Combined phase allocation | 1.28522 GPU-hours |

This is a measured cache-reservation failure, not a new backward OOM. Increasing
the vLLM reservation could clear this startup check but would leave less memory
for backward. The earlier 5,120-trajectory-token inbox backward OOM remains
relevant evidence against assuming that this longer training workload fits.
Neither run establishes the exact memory capacity required for flight training.
Inference survival cannot establish that capacity either.

The automatic handoff first stopped before run creation because the launch
manifest and eval stamp spelled the same Python executable differently, with
and without `pipeline/..`. The corrected check resolves executable paths while
preserving strict comparisons of other stack fields. Regression checks accept
that alias and reject package or interpreter changes. Admission then passed;
there was exactly one native training attempt and no retry after its failure.

## Options overview

These are alternatives to qualify, not approved changes or guaranteed fixes.
Some can be combined.

| Route | Examples |
| --- | --- |
| More hardware | Larger-memory GPU; split generation and training across GPUs; distributed training |
| Different policy | Qwen3-4B or 8B; another model lineage; a smaller policy if memory is the priority |
| Lower precision | 4-bit QLoRA, 8-bit weights, or quantized inference KV cache |
| Runtime changes | Rebalance vLLM memory; use another generation backend; CPU offload; revalidate fused loss or sleep mode |
| Shorter observations/context | Compact visible HTML, structured observations, deltas, or an explicit history policy |
| Better task competence | Prompt/tool guidance, examples, thinking-mode changes, SFT, or curriculum before a fresh matched comparison |
| Different interaction protocol | Higher-level tools, shorter stages, revised token/turn limits, or revised rollout geometry |
| Another MiniWoB family | Search-engine, multi-layouts/multi-orderings, choose-date, or login-user-popup after approval |
| Extend retained evidence | Frozen-checkpoint robustness tests, interface shifts, or replication on table/inbox |

LoRA, gradient checkpointing, and micro-batch size 1 are already in the recipe.
They cannot simply be switched on again to create new headroom. Liger and sleep
mode have prior correctness findings in `probe_p2_findings.md`; they require
revalidation. Any change affecting the experiment needs a new declared protocol
and a consistent comparison. No family fallback or full campaign was launched.

Documentation checked: [TRL memory options](https://huggingface.co/docs/trl/reducing_memory_usage),
[separate vLLM serving](https://huggingface.co/docs/trl/vllm_integration),
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B),
[quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/),
and [MiniWoB families](https://miniwob.farama.org/environments/list/).
These document possibilities; compatibility with the pinned stack still needs
to be checked for any selected change.

## Evidence and stop state

All 18 harvested run files match hashes taken after both controllers exited.
Frozen configs, stack stamps, exact seeds, logs, costs, and five-second GPU
samples are retained in the two run directories. The operations bundle is
`flight-budget-s4023-ops/`; `final_resource_check.json` contains artifact hashes
and the final device/process check, and `review.json` records the assessment.
Both GPUs were idle with zero device memory use and no compute processes at
2026-10-06T21:31:21Z. No further phase is queued.

The evidence supports two separate concerns: base-policy competence and the
current training memory allocation. It does not yet support a specific larger
GPU capacity as a sufficient solution. A supervisor request can use these
findings, but its requested size and duration need a selected workload and
explicit uncertainty. The next step is to choose the qualification route.
