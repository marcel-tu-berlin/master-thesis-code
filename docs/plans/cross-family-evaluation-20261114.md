# Cross-family evaluation of final E1/E2 checkpoints

Agreed on 2026-11-14. Finish the current replication campaign first. After
its 40 cells, harvests, scientific reviews and reporting are complete, compare
evaluation concurrency on gpu-l4 physical GPU 1, then use the fastest verified
configuration for this evaluation matrix. Taskwarrior tracks this follow-up
as dependent on replication task UUID 1ddd098f-c221-4754-bbdc-94b8ace95d2f.
Do not change, interrupt or add workers to the running replication queue.

## Scope

Use the first two seeds overall: the original seed and the first additional
replication seed. The second additional seed remains outside this follow-up,
even though completing it is part of the current campaign.

| Training family | Original seed | First replication seed |
|---|---:|---:|
| read-table-2 | 4016 | 4017 |
| email-inbox-noscroll | 4021 | 4240 |
| click-checkboxes-soft | 4232 | 4242 |
| tic-tac-toe | 4230 | 4244 |
| click-tab-2 | 4231 | 4246 |

Use only the final update-300 E1 and E2 checkpoints, with the existing E2
dose 0.1. This gives 20 checkpoints. Each receives the four other target
families: 80 new evaluations of 200 episodes each, or 16000 episodes.
Reuse the 20 existing own-family final evaluations and the 10 matching E0
evaluations after verifying compatibility. No new training, placebo adapters,
intermediate checkpoints, model scales, doses or E3 are included.

For each target family and seed round, reuse that target's existing E0
held-out questions, initial observations, fixed thresholds and evaluation
settings. All source-family adapters in that round must receive the same
target questions. The target seed controls evaluation questions; record the
adapter's training seed separately. Do not accidentally generate target
questions from the source family's seed or evaluate on training questions.

Keep the target family's frozen environment and evaluation semantics,
model/tokenizer revision, greedy decoding, 4096 trajectory-token budget
including tool feedback, 4096 prompt limit, 8192 context and eight turns.
Retain native checkbox success separately from exact completion and native
tic-tac-toe outcomes with the accepted saved-prefix sensitivity.

Before execution, bind the exact adapters, source run IDs, hashes, target
configs, question sets and E0 references in a manifest. Verify reusable
diagonal/E0 compatibility rather than silently rerunning or substituting it.
Check checkpoint availability on L4, especially the original checkbox E2
trained on home hardware. Missing weights or incompatible evidence require
a reported blocker, not contact with the home host or selection of another
seed. Only gpu-l4 GPU 1 is authorized; GPU 0 and the desktop remain excluded.

## Concurrency selection after replication

Compare one worker as the baseline with two, three and four concurrent
evaluation workers on GPU 1. Reuse the existing evaluator and its precision;
do not change inference backends, quantization or experiment budgets to fit
more workers.

Freeze a small, bounded benchmark workload before timing it. Include all
five target families, both E1/E2 and cases that exercise long trajectories.
Use identical total work and question IDs at every concurrency level, with
separate output directories, environment servers and ports. Benchmark outputs
remain diagnostic and never substitute for the declared 200-episode results.

Measure aggregate completed episodes per elapsed hour, including startup,
plus steady-state throughput, peak GPU memory, GPU health and CPU/RAM use.
Compare per-episode outputs, actions, outcomes, token accounting and stop
reasons against the single-worker baseline. Only configurations that preserve
the evaluation behavior and finish without resource failures are eligible.
Retain failed benchmark evidence without overwriting or retrying it.

Select the fastest eligible configuration automatically. If timings are
indistinguishable, prefer fewer workers and more memory headroom; if parallel
execution gives no verified benefit, use one worker. Record measured throughput
and the resulting full-matrix estimate before starting the matrix. The current
approximately 4.6 GiB evaluator snapshot suggests room for parallelism but
does not establish peak memory or a speedup for any concurrency level.

## Results and completion

Use new, exclusive run IDs and output paths; preserve all existing artifacts.
Hash-verify completed outputs and retain full trajectories, environment
stamps and costs. Count shared-GPU elapsed allocation once when reporting
campaign GPU-hours; summed overlapping worker wall times are not distinct
physical GPU-hours. Stop and preserve evidence on an unexpected production
runtime, integrity or audit failure, without automatic repair or restart.

Compare E2 versus E1 within each source family, target family and seed round;
use E0 to describe the target baseline and specialization. Preserve paired
questions, joint-success token coverage and target-specific metric definitions.
Report both seeds separately before any equal-weight summary. Two training
seeds provide limited robustness evidence; episodes are not training-seed
replicates. Keep original training-host/runtime differences explicit even
though the new evaluations use L4. Obtain independent Claude Opus 5.5/xhigh
review of the execution protocol before launch and of the findings before
closing this follow-up. No GPU work is started by recording this plan.
