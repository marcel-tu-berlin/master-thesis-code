# Validate the instrument before the campaign

Date: 2026-09-11. Status: implemented through Gate 3, not a launch authorization.

The next work should establish that the pipeline measures and optimizes the
intended quantities. Then run one E1 pilot, inspect it, and release one shaped
contrast at a time. A nine-cell batch is too expensive to use as a setup test.

No finite test suite establishes that software is 100% correct. The practical
standard is that every listed scientific invariant has a passing check, no known
critical discrepancy remains, and the remaining uncertainty concerns the research
outcome rather than the measuring instrument. A null result must remain possible.

## What the current evidence establishes

The local gate passed: formatting, lint, types, 369 tests passed with 3 skips,
and the setup harness reported 17 passed, 0 failed. The skips cover checks that
need the training stack, including real response parsing and metrics callbacks.
CPU tests do not establish correctness of the installed trainer's execution.

The accepted recipe is decisions 0010-0013: Qwen3-1.7B, mean-only advantages
through `naive_sum` and `scale_rewards: none`, one update per rollout batch,
Liger off, sleep off. Keep it fixed while validating it. The archived P2 runs
used `scale_rewards: group`, so their successful execution does not validate the
new recipe's gradient scale or stability. Their findings already exclude two
unsafe optimization paths; repeating that four-arm comparison buys nothing.

Five gaps matter before the next campaign:

1. **A smoke tests different geometry.** It changes rollouts from 8 to 2,
   caps total context at 2048 and the prompt at 1024, and caps eval generation
   at 256 tokens. It cannot establish memory fit, stopping behavior, or learning
   signal at the intended 8192/4096 context split and 4 x 8 rollouts.
2. **Passing component tests is not trainer integration proof.** Verify the
   rewards, actual advantages, tool-token mask, loss denominator, and policy
   updates together against the pinned stack. Aggregate reward or ISR means can
   hide a wrong assignment or a length-dependent distortion.
3. **The setup documents contain unsafe starting points.** Code BACKLOG's E3
   criterion uses total non-termination; the current reward and vault backlog
   correctly target budget exhaustion. Also, the disabled E2 template has
   `max_len: 256`: enabling it unchanged makes correct lengths 256, 1024, and
   2048 all score 0.5. Neither observation proves an existing run is wrong.
4. **There is no scientific admission gate around batches.** The runner reads
   headers up front, leaves full validation to subprocesses, and proceeds to
   later configs after a failed training phase. It does not stop on bad
   measurement evidence. Resume skips completed phases; `trainer.train()` is
   not currently called with a training-checkpoint resume argument.
5. **The evidence saved during training is insufficient for cheap diagnosis.**
   The P2 runs have no saved training trajectories. `train_log.json` is written
   after training returns. Eval saves assistant turns and action arguments, but
   not the initial task observation and tool feedback needed for a complete
   task-grounded audit. Small, bounded diagnostic records would avoid reruns.

## Gate 1: freeze the contract and check it without training

Use the existing schema, reward implementations, tests, and comparison tools.
Add only checks that close a demonstrated coverage gap. This is a CPU gate;
real tokenizer checks can run on the box without model generation.

| Contract | Required evidence |
|---|---|
| Recipe and geometry | Fully validate every proposed config before launch. Resolve and record effective trainer settings, including DAPO loss, precision, model/tokenizer revision, 4 prompt groups, 8 rollouts, micro-batch 1, 32 accumulation steps, budgets, task list, and scheduler. Reject campaign dose warnings. |
| Reward arithmetic | Hand-labelled examples for correct/incorrect, short/long, environment completion, voluntary stop, turn exhaustion, and token exhaustion. Recompute each raw reward and the weighted sum independently. |
| Dose survives centering | On fixed groups, compare the actual trainer advantages with `R_i - mean_group(R)`. Include constant-task-reward groups with varying costs. Doubling lambda must double the shaped contribution to centered advantages. Test both E2 and E3, plus lambda zero. |
| Placebo | Reuse the uniform within-group shuffle. Preserve each group's multiset and check expected centered contribution over permutations. Do not require every rollout to move. Do not call this matched total-reward variance or matched gradient noise. |
| Tokens and stopping | Reasoning and tool-call arguments belong to assistant content length; tool replies do not. Trajectory budget includes tool replies. Record exact generated tokens separately from the existing content re-encoding approximation. Test budget boundaries, truncated calls, multiple calls per turn, and done precedence. |
| Seeds and artifacts | Assert train/calibration/test disjointness, same question IDs across compared arms within each seed, correct split-to-family mapping, checkpoint labels, report completeness, and overwrite refusal. |

For signed, correctness-coupled E2 rewards, enumerate the configured endpoints
and possible placebo assignments before selecting lambda. Do not borrow E3's
unit-cost bound blindly. The chosen grid must not rank an unsuccessful rollout
above a successful one merely through its cost assignment. Check the actual
formula used by the code, rather than only the generic thesis notation.

The existing E2 ruler is a documented approximation, while eval counts raw
generated tokens. Do not silently redefine either measurement to make them
equal. Quantify their difference on fixed examples and check that missing
reasoning, tool interleaving, or success does not change which ruler is used.

**Pass:** all deterministic contracts hold, including deliberately constructed
edge cases. Existing regression tests must reject their known bad cases. An
unverified field is not a pass. No GPU training begins while this gate fails.

## Gate 2: inspect one real rollout batch before a learning experiment

Run against the exact deployed stack and physical GPU 1. Reuse the real adapter
and trainer path; the old standalone difficulty probe has different historical
budget/decoding assumptions and is not the admission test.

First check the OpenEnv pin, installed package and source hashes, model and
tokenizer snapshot, MiniWoB availability, port ownership, and GPU assignment.
Exercise real resets and actions. Confirm that eight rollout slots for a seed
see the same task, different seed groups stay separate, and the goal survives
observation truncation. Re-run the relevant skipped tests with the installed
tokenizer and training packages.

Capture one full 4 x 8 sampled batch, with prompt/seed/group/slot IDs, initial
observations, assistant messages and token IDs, tool replies, environment
reward/done, masks, raw component rewards, advantages, and selected-token
sampler/trainer log probabilities. Keep this diagnostic capture bounded and
separate from ordinary metrics. Replay the captured inputs across candidate
weights and placebos without generating new trajectories.

Use the installed trainer on fixed inputs to compare its advantages, scalar
loss, and gradients with the small reference calculation. Include different
trajectory lengths and tool masks, which is where the Liger comparison failed.
Replay checks validate arithmetic, not the policies that other weights would
eventually learn. If a needed ending is absent from this batch, use a labelled
scripted trace rather than waiting for random generation to produce it.

**Pass:** reward-to-rollout alignment and loss checks agree within a declared
numerical tolerance; the sampled policy matches the intended synchronized
weights; no unexplained token-mask or length-dependent logprob discrepancy
remains. Ordinary HF/vLLM numerical differences need not be exactly zero.
Check the distribution and its relation to length, not only ISR mean near one.

## Gate 3: short execution checks at full geometry

Run three optimizer updates each for E1, E2, and E3, sequentially. Use separate
diagnostic IDs and a development seed block. Reuse Gate 2's capture in the first
check instead of generating it twice. Test the highest proposed admissible
weight for each shaped condition. This is nine updates, not three 50-step runs.

These are ordinary explicit diagnostic configs, not `--smoke`: preserve model,
precision, 8192 total context, 4096 prompt cap, eight turns, 4 x 8 geometry,
sampling, loss, and reward recipe. Shortening `max_steps` also shortens the
relative warmup schedule; this gate proves execution and finite updates, not
stability over the final run's full warmup. Record that limitation.

Inspect memory, finite loss/gradients/parameters, nonzero updates on known
informative groups, per-group reward variance, and sampler synchronization after
updates and across tool turns. A zero gradient for a uniform-reward group is
expected, not automatically a defect. An agent's invalid action is also distinct
from infrastructure returning invalid actions for every request.

Save and reload the adapter, evaluate a small fixed set at the real trajectory
budget, and check the reports against recorded episodes. Checkpoint save/load
and evaluation resume already have CPU tests and historical live evidence;
repeat the live lifecycle only where code, stack, or protocol changes invalidate
that evidence. Do not demand improved success after three updates.

**Proposed diagnostic ceiling:** two GPU-hours for Gates 2-3, including model
loads and tiny evals. This is a spending limit, not a runtime promise. If reached,
retain the evidence and diagnose the bottleneck before allocating more. Stop at
the first unexplained invariant failure; do not auto-retry or launch the next arm.

**Pass:** the final recipe executes at real geometry, produces the intended
updates, and saves an evaluable policy. This admits the first real E1 run.

## Gate 4: one E1 run decides whether the experiment is informative

Train one E1 from the base model with the accepted recipe. Do not add another
50-step learnability run before it. If its recipe and family survive the pilot,
retain this E1 as the first exploratory campaign cell. If they change, keep its
results as development evidence rather than silently comparing across recipes.

Declare the horizon before launch. Keeping the existing 300-step design means
retaining checkpoints at 100, 200, and 300. Evaluate every planned observation;
never select the best checkpoint. The existing schedule evaluates saved
checkpoints after training, not as an online stop-and-resume controller.

Use calibration instances for family choice, reward-scale selection, and proxy
validation. Start with 100 same-family calibration episodes and a balanced,
smaller sample from each shifted family. These are diagnostic sample sizes,
not a claim of adequate power for the final paired comparison.

Create the E0 calibration reference before launching a training config that
requests scheduled evaluation: the schema requires an existing reference with
samples for every split. Keep calibration reports under dedicated experiment
IDs, using explicit saved checkpoint paths. Final test evaluations get separate
outputs after protocol freeze. Never change the protocol in a completed
scheduled-eval directory; its resume guard deliberately rejects that change.
Retain checkpoint and training-run provenance when using separate eval IDs.

Write numeric feasibility criteria before reading the new E1 results. Suggested
starting rules, to be agreed as design choices rather than treated as facts:

- **E3 opportunity:** at least 10 budget-exhausted episodes out of the first
  100 same-family calibration episodes. Compute this from `max_turns` plus
  `hit_generation_cap`; voluntary `no_tool_call` stops do not qualify.
- **Success headroom:** same-family E1 success below 90%. Report its interval;
  a borderline value is inconclusive. A ceiling is a design concern, not an
  assertion that success degradation is mathematically unmeasurable at 100%.
- **E2 opportunity:** successful trajectories have nonconstant length rewards
  within sampled groups and plausible room for a meaningful reduction, checked
  against valid shorter solutions. Predeclare the meaningful reduction before
  testing shaped policies; 10% is a candidate planning margin, not a result.
- **Readable off-target panel:** blind a small stratified trajectory sample to
  condition, including efficient correct negative controls. Retain harmful-
  behavior labels only where task observations and actions support them.
- **Bounded ambiguity:** allow at most one predeclared extension to 200
  same-family calibration episodes. If still inconclusive, report that and make
  a design decision; do not sample indefinitely until a bar is crossed.

Also inspect sampled training groups near the end of E1: held-out nonzero cost
alone does not prove the cost varies within rollout groups, where GRPO obtains
its signal. Training reward trends alone do not prove held-out learning either.

Reserve untouched final test instances. Freeze the family, recipe, reward
endpoints and lambda grid, metric definitions, success non-inferiority margin,
off-target harm margins, and paired analysis before using them. Calculate the
paired minimum detectable effects from calibration discordance and the planned
seed count. A nonsignificant success loss is not evidence of preserved success.

The current seed-block design pairs arms within each training seed and changes
evaluation instances across training seeds. Keep that explicit in the thesis and
analysis; do not claim one common instance set across seeds. E0 needs coverage
of each distinct evaluated instance set, although it has no training replicas.
Reuse E0 results only for exactly matching instances and protocol. Pin reference
thresholds per split across every arm and scheduled checkpoint.

**Pass:** the chosen family retains the targeted costs, the metrics can express
the thesis questions, and the evaluation can resolve effects worth claiming.
Failure blocks the affected shaped campaign, not the reporting of the E1 result.

## Gate 5: release research work in small units

The first informative contrast is E1 against one predeclared E2 weight on the
same paired instances, followed by one E3 weight if its opportunity gate passes.
Every trained arm starts from the same base, not from E1's trained adapter.
Hold training steps, prompt exposure, sampling, precision, and budgets fixed.

After each arm, harvest the evidence and check task success, both-correct paired
token change, ending transitions, and family-specific off-target outcomes. A
negative, null, or indeterminate result is a research result and is not itself a
reason to debug or replace the setup. Expand only after the measurement checks
pass; do not choose replication solely because an effect is significant.

The complete thesis design still needs three weights per shaped condition,
placebos at the declared top weights, and at least three training seeds per
cell. Complete those in explicit tranches once the first contrasts are readable.
This is a proposed change to the old operational plan to queue all nine stage-1
cells together, not removal of the dose-response or replication requirements.
Run the relevant placebo before making a cost-assignment mechanism claim.

Keep all planned checkpoint observations on the same final test instances and
reference thresholds. They are repeated measurements within
a seed, not extra independent experiments. Report per-seed effects and account
for training-seed variation; episode counts do not replace training replications.
Do not alter the off-target panel until a verdict flip appears. RQ3 can return
that no additional harm was detected within the validated panel.

## Time budget and the smallest implementation

The archived P2 baseline's 50 logged updates average 414.95 seconds each,
recomputed from `train_log.json`. The four-arm batch took 21h27m.

ASSUMPTION: using that baseline speed as a provisional forecast, nine diagnostic
updates take about 62 minutes, a 300-step arm about 34.6 hours, and nine such
arms about 311 hours, before startup and evaluation. The new mean-only recipe
has not been timed and may change trajectory lengths. Re-estimate from Gate 3,
and budget all three checkpoint evaluations separately. The old six-hour-per-
expert estimate is not a safe budget for this recipe and horizon.

If 300 steps do not fit the thesis budget, choose a shorter fixed horizon for
all compared arms before launch, with thirds preserved. Do not turn an early
checkpoint into the preferred endpoint after seeing its results, or assume a
short-run adapter can be resumed by the current batch runner.

The implementation is `python -m probes.readiness`. It validates the explicit
E1/E2/E3 configs under `configs/readiness/`, runs the local project gate, and
checks any matching run directories. Its JSON report contains config, source,
and stack fingerprints; pass/fail/not-tested results; captured GPU time; and the
one next phase it admits. Missing evidence exits 2, failed or stale evidence
exits 1, and a complete Gate 3 exits 0.

Each diagnostic run uses `training.train --readiness-capture`. This selects an
observational trainer subclass only for that invocation. It records one full
4 x 8 batch before TRL shuffles it into microbatches, the first update's 32 loss
shards, raw E1/E2/E3 rewards, exact masks and log probabilities, gradient and
parameter checks, effective trainer settings, and per-step logs. Normal training
still uses TRL's trainer class directly. Evaluation episode records now include
the initial observation and tool feedback.

Run the controller between arms:

```bash
cd pipeline
python -m probes.readiness \
  configs/readiness/g3-e1.yaml \
  configs/readiness/g3-e2.yaml \
  configs/readiness/g3-e3.yaml

# Run only the experiment_id named by Next phase. Re-run this controller in the
# deployed GPU checkout before harvesting so current-stack drift is detectable.
python -m training.train \
  --config configs/readiness/g3-e1.yaml \
  --eval --readiness-capture
```

The command deliberately does not launch jobs. Manual review remains between
arms, so one failed contract cannot fall through into another GPU run.

Persist small step summaries as they occur and harvest logs/reports after each
phase. Keep adapters on the box under the existing retention rule. Maintain
RUNNING with live state only; findings own numbers, decision records own adopted
protocol changes, and BACKLOG/taskwarrior own agreed open work. Repair the E3
criterion wording when adopting this plan. IMPROVEMENTS remains off the critical
path: no larger models, quantization, Liger, sleep, or backend experiments until
the first interpretable 1.7B contrast and its seed plan are secured.

## Evidence and limits of this review

Read the current LAB_NOTES, BACKLOG, IMPROVEMENTS, recipe decisions, historical
P2 evidence, pipeline entry points and tests, and the thesis PDF's research
questions and Chapters 4-6. The writing vault backlog already distinguishes
calibration from final test data and budget exhaustion from total non-termination.
Taskwarrior currently identifies the final-recipe E1 pilot as the critical task;
the 4B probe is optional.

[TRL's GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) describes
the separate reward-scaling and loss-aggregation controls. For version-specific
claims, this review uses the archived installed 1.6.0 source excerpts in
`pipeline/runs/probe_p2_source_audit.json`, rather than assuming current online
documentation describes the locked environment exactly.

This review and implementation did not launch GPU work, revalidate the live
remote installation, or establish that the new recipe meets Gates 2-4. Those
gates remain not tested until the controller accepts the corresponding capture.
