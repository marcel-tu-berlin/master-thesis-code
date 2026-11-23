# Cross-family transfer evaluation

Agreed on 2026-11-14 and narrowed to one seed per family on 2026-11-22.
On 2026-11-22 the user added the matched placebo, approved staging and authorized launching the
benchmark once the current NAS backup is verified and physical GPU 1 on
gpu-l4 is released. The preceding campaign owns its own closeout, including
harvests, scientific reviews and reporting. The transfer controller is
separate from its queue. Operations and the frozen input manifest are in
`pipeline/runs/cross-family-one-seed-20261122-ops/`.
Taskwarrior tracks the dependency on replication task
1ddd098f-c221-4754-bbdc-94b8ace95d2f.

## Scope

Use the first replication round for all five families. All fifteen E1, E2 and placebo models
in this round trained on L4 GPU 1. This choice keeps training hardware
consistent. The original checkbox E2 and placebo adapters and the
original tic-tac-toe placebo were trained on home hardware. The original seed round is deferred pending a separate decision after
the one-seed results.

| Family | Training and own-family evaluation seed |
|---|---:|
| read-table-2 | 4017 |
| email-inbox-noscroll | 4240 |
| click-checkboxes-soft | 4242 |
| tic-tac-toe | 4244 |
| click-tab-2 | 4246 |

Use only the final update-300 E1, E2 and matched-placebo checkpoints. E2
and its placebo use the same dose of 0.1. Fifteen checkpoints times four
other targets gives 60 transfer evaluations of 200 episodes, or 12000
episodes. Reuse the fifteen own-target finals and five E0 baselines.
Checkpoint 100/200 and the original seed round require a separate decision.

For each target family, reuse its E0 held-out questions, initial observations,
fixed thresholds and evaluation settings. A local check on 2026-11-22
compared all 15 saved E0 and final E1/E2 episode files. Each contains the
expected 200 ordered question seeds and each family's three files match
exactly on seed and initial observation. Check every transfer trajectory
against the same target record. The target seed controls the questions;
record the adapter's source family and training seed separately.

Use the target family's frozen evaluator and environment, including its tools,
model/tokenizer revision, greedy decoding, 4096 trajectory-token budget
including tool feedback, 4096 prompt limit, 8192 context and eight turns.
Retain native checkbox success separately from exact completion and native
tic-tac-toe outcomes with the accepted saved-prefix sensitivity.

Before execution, bind the exact adapters, source run IDs, hashes, target
configs, question sets and E0 references in a manifest. Verify reusable
diagonal/E0 compatibility rather than silently rerunning or substituting it.
Verify all fifteen final adapter files on L4 and bind each to its original
update-300 report. A transfer report may lack checkpoint-step metadata for
an external adapter, so verify the source path and hash against the manifest.
Report missing weights or incompatible evidence before launch. Only gpu-l4
GPU 1 is allocated to this plan.

## Concurrency selection after replication

Compare one worker as the baseline with two, three and four concurrent
evaluation workers on GPU 1. Increase one level at a time, checking measured
peak memory and host headroom before admitting another worker. Stop escalating
when headroom is insufficient or a level has a resource failure. Reuse the
existing evaluator, precision and budgets throughout.

First compare one historical episode for each of the five base and fifteen
own-target final evaluations against its saved trajectory, ignoring elapsed
times alone. Then use twelve four-episode transfer jobs (48 episodes per
worker setting). Each condition has four jobs. Each target has a block
beginning at held-out offset zero and its most token-intensive four-question
E0 block. Two additional blocks begin at offset 96. Group the same targets
together to exercise their concurrent servers. Separate ports, output,
temporary files and mutable kernel caches for each worker. Diagnostic
outputs stay separate from the full cells.

Measure aggregate completed episodes per elapsed hour, including startup,
plus steady-state throughput, peak GPU memory, GPU health and CPU/RAM use.
Compare per-episode outputs, actions, outcomes, token accounting and stop
reasons against the single-worker baseline. Only configurations that preserve
the evaluation behavior and finish without resource failures are eligible.
Retain failed benchmark evidence without overwriting or retrying it.

Keep a 2 GiB GPU memory reserve and at least 4 GiB of available host memory.
Before adding a worker, allow for the measured single-worker peak and a
full-context KV cache per worker. Use measured worker overhead and episode
time to predict 200-episode cell durations. Match target and condition where
available, otherwise use the target average. Simulate the fixed 60-cell queue
at each eligible concurrency, including its final wave. Choose the fewest
workers within 5 percent of the shortest projected time. Publish the raw
throughput and update the ETA using full cells.

The ten E1/E2 own-target final evaluations took 30.46 hours. The first four
placebo finals averaged 3.46 hours each. At these rates the expanded workload
would take about eight days with one worker. Transfer failures can consume
the full budget. The benchmark provides the first transfer-based estimate.

## Results and completion

Use new, exclusive run IDs and output paths; preserve all existing artifacts.
Hash-verify completed outputs and retain full trajectories, environment
stamps and costs. Count shared-GPU elapsed allocation once when reporting
campaign GPU-hours; summed overlapping worker wall times are not distinct
physical GPU-hours. Stop and preserve evidence on an unexpected production
runtime, integrity or audit failure, without automatic repair or restart.

Report three 5-by-5 matrices, one per arm, with source families in rows and targets
in columns. The diagonal is the target-trained specialist. Within each
column, compare each transferred model's success to the same target E0 and
to the corresponding specialist. For each source-target pair, report the
paired E2-minus-E1 and E2-minus-placebo success differences in percentage
points on all 200 questions. Placebo-minus-E1 is secondary. Use descriptive
unadjusted 95 percent intervals by paired question bootstrap (20000 draws,
seed 0) for success and the established paired median interval for tokens.
Report the target-specific pairs individually.
Estimate the paired change in assistant tokens on joint successes, giving
their count out of 200. Report the token comparison as unavailable when that
count is zero. Show early stops and cap hits alongside the target-specific
outcomes. These intervals describe question-level uncertainty for the chosen
models. Review the full matrix before deciding whether another training seed
is worth evaluating.

Obtain independent Claude Opus 5.5/xhigh review of the execution protocol
before launch and of the findings before closing this follow-up.

### Recovery after the first production stop

On November 22 the user confirmed that the goal is completion of all 60
cells, with health checks in this chat through interruptions and recovery.
Diagnose each stop before continuing. Ask for a decision if recovery would
change the scientific protocol, hardware allocation or scope.

On November 23 the user set an hourly check for this evaluation. If SSH
times out, pause checks until the user confirms the TU VPN is back. Keep the
last GPU-verified state in `RUNNING.md` during the pause.

The initial stop and both saved traces are bound in
`pipeline/runs/cross-family-one-seed-20261122-ops/stopped-20261122T191641Z/`.
The first recovery uses a separate directory. It admits the original
200-episode Tables E2 to Inbox cell only after a full offline audit on a
byte-identical copy. Tables E1 to Inbox starts a fresh 200-episode attempt.
The interrupted 44 rows count toward compute cost only. Keep the original
manifest, two-worker setting, evaluator, native scores and 60-cell denominator.
