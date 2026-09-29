# Initial E0/E1/E2 comparison, seed 4016

Protocol: `docs/plans/e0-e2-campaign.md`, decisions 0021/0022/0023. The
authorized comparison is E0, E1 and E2 at weight 0.1, all at seed 4016. The
completed trio requires a user decision before any further experiment.

**Final status, 2026-09-27: initial comparison reviewed; decision required.**
All three runs and all seven evaluations are complete and reviewed. E2 at
weight 0.1 meets the declared compression and success-preservation targets
on this seed: median paired successful-token change -44.93% (95% CI
-47.33% to -42.75%), with 198/200 successes versus E1's 199/200. There are two
regressions and one gain; the regression upper bound is 3.11%, below the
5-percentage-point margin. The four declared off-target bounds also pass.
This is a positive initial result on one narrow family, not evidence of
training-seed robustness or full behavioral equivalence. The final review
and limitations follow the dated E0/E1 records below.

## E0: base-model evaluation

Run: `e0-read-table-2-s4016`. Completed 2026-09-24 with exit code zero.
All 200 held-out questions, seeds 4016100000-4016100199, were evaluated.

| Measure | Result |
|---|---:|
| Correct | 43/200 (21.5%) |
| Success Wilson 95% interval | 16.37%-27.70% |
| Mean assistant tokens on correct episodes | 2027.12 |
| Correct-token bootstrap 95% interval | 1755.55-2305.44 |
| Environment-done endings | 74/200 |
| Generation-cap endings | 107/200 |
| No-tool-call endings | 19/200 |
| Wrong submissions | 31/200 |
| Non-termination | 126/200 (63%) |
| Episodes with invalid actions | 94/200 (47%) |
| Episodes with repeated actions | 11/200 (5.5%) |
| Evaluation phase wall time / allocated GPU time | 3.991 hours |
| Summed episode wall time | 14355.76 seconds |
| Summed policy inference wall time | 13416.56 seconds |

The fixed reference thresholds are P10 = 1213.7 and P75 = 4096 assistant
tokens. E1/E2 must read the unchanged E0 report. The pooled token mean is not
the successful-response efficiency measure.

### Integrity review

All harvested raw artifacts match server hashes. The original launch manifest,
launcher, frozen config, runtime source hashes and actual evaluation stack
match the preserved launch evidence. E0 predates the scope reduction, so its
manifest resolves through `e0-e2-campaign-ops/before-initial-scope-20260924/`.
The scope change did not change runtime sources or experiment configs.

Every trajectory was checked against the visible table, submitted value, action
history, termination record and token accounting. All report metrics were
recomputed from samples. All 200 episodes have consistent timings, and the
cost ledger contains one matched, completed attempt. The hash-bound
`e0-read-table-2-s4016/review.json` binds the raw evidence, audit program and
`technical_review.json` with individual trajectory audit results.

### Interpretation

The base model has low success and frequently exhausts its generation budget.
These are baseline outcomes, not an integrity failure or an additional launch
gate. E1 is admitted under the declared initial comparison. E0 alone establishes
neither learning nor compression; those require the trained arms and the
declared paired analysis. One seed cannot estimate training-seed variation.

Action and stopping measures remain descriptive until the cross-arm label
review, including masked efficient controls. Aggregate E0 outcomes were known
during this technical audit. No treatment effect or off-target equivalence is
claimed here.

## Training-budget diagnostic, 2026-09-25

This is an interim investigation requested by the user, not a protocol change
or an early stopping decision. The snapshot ends at E1 update 93; no current
trained held-out evaluation is available yet. Evidence is in
`e0-e2-campaign-ops/training-budget-review-step93.json`.

### Task coverage and repetition

The frozen configuration contains one family, `read-table-2`, and 500 fixed
seed rows. Each update uses four rows with eight sampled attempts per row.
The installed TRL 1.6.0 sampler shuffles without replacement within a pass.
Its 32 repetitions serve micro-batch accumulation; `_prepare_inputs` generates
one batch and reuses slices, not 32 additional sets of rollouts. A CPU-only
reconstruction using the installed RepeatSampler class matched the first
captured four seeds exactly. The logged epoch at update 93 is 0.744.

| Updates | Task presentations | Distinct seed rows | Passes | Sampled attempts |
|---:|---:|---:|---:|---:|
| 93 | 372 | 372 | 0.744 | 2976 |
| 100 | 400 | 400 | 0.8 | 3200 |
| 125 | 500 | 500 | 1.0 | 4000 |
| 200 | 800 | 500 | 1.6 | 6400 |
| 300 | 1200 | 500 | 2.4 | 9600 |

At 300 updates, 300 rows have two visits and 200 have three. A later visit
samples fresh policy trajectories on the same deterministic page. These are
seed-row counts; an exhaustive content-collision audit was not performed.

The pinned MiniWoB source (`eb59fed60fabe8951350275ba8650633b740013b`,
`miniwob/html/miniwob/read-table-2.html`) generates five table rows from eight
field categories, samples their values, chooses two target fields and varies
their form order. It can generate far more than 500 instances; 500 is our
chosen training-pool size. The required behavior remains a narrow table lookup,
two fills and submission. More instances of this family do not establish
generalization to different browser tasks. The 200 held-out seeds are disjoint
from the training seeds but use the same family.

### Evidence for diminishing returns

Training reward means, which equal sampled task success in E1, rose from
36.09% over updates 1-20 to 93.91% over updates 74-93. In the latter window,
68.75% of prompt groups had zero task-reward variance. Seven of the first 93
updates had zero logged gradient norm (68, 69, 74, 78, 80, 82, 84); five were
in the latest 20. These observations indicate weakening task-reward learning
signal, not held-out generalization or proven overfitting. Zero logged gradient
does not imply unchanged parameters because optimizer state and weight decay
can still act.

The archived seed-4009 E1 pilot has identical frozen training and model settings
to this E1, with another seed and a different 100-question development split.
Its original reports were checked directly:

| Update | Correct / 100 | Mean assistant tokens on correct episodes |
|---:|---:|---:|
| 100 | 93 | 1320.19 |
| 200 | 100 | 664.92 |
| 300 | 100 | 580.87 |

The last 100 updates added no measured success but reduced mean tokens by
12.64% on the same 100 correct questions. Thus, later training was not wholly
inert. These remain development observations, not current-seed results or proof
of a universally sufficient horizon. Source:
`archive/development-2026-09-24/table_pilot_s4009_findings.md` and its linked
checkpoint reports.

### Assessment

There is no evidence that the task generator runs out of instances at 300
updates, or that 2.4 passes alone make the comparison invalid. There is evidence
that E1 learns this narrow family early and later compute has diminishing
returns. A 200-update matched comparison is a plausible cheaper future design;
the current evidence does not establish it as sufficient for E2. The relative
length reward can distinguish successful rollouts after task reward becomes
constant, so E1 saturation is not an E2 stopping rule.

Retain the currently authorized matched 300-update horizon for this initial
comparison and use its predeclared 100/200/300 observations to quantify the
marginal benefit of later training before any expansion. The implementation
saves intermediate adapters during training and evaluates all three after
training completes; it does not currently provide an online held-out early
stopping signal. A future shorter study should declare a matched horizon and
checkpoint schedule for both arms; its lower update-300-relative exposure and
warmup length must be explicit. Do not change just E1 or choose the best
checkpoint from the final held-out results and relabel it the primary result.

## E1 completed and reviewed, 2026-09-26

`e1-read-table-2-s4016` completed 300 updates and all three 200-question
evaluations with controller exit code zero. All 25 harvested raw files match
server hashes. The launch manifest, source/stack/config evidence, unchanged E0
reference and checkpoint metadata passed review; checkpoint-final and
checkpoint-300 have identical adapter hashes. All 300 logged updates are finite.

The visible-table audit passed every E1 evaluation trajectory (600 total),
including scores, submitted values, action counts, native errors and stopping.
All reports were recomputed from samples with E0's frozen thresholds. Every
checkpoint used the same 200 seeds and byte-identical initial observations as
E0. All 416 captured training trajectories passed grounded reward, termination,
group alignment and token checks; independent recounts with the saved tokenizer
matched all 416 assistant-token counts. The four cost attempts are complete.

| Policy | Correct / 200 | Wilson 95% success interval | Mean assistant tokens, correct episodes | Evaluation GPU-hours |
|---|---:|---:|---:|---:|
| E0 | 43 | 16.37%-27.70% | 2027.12 | 3.991 |
| E1 update 100 | 183 | 86.81%-94.63% | 1506.58 | 3.933 |
| E1 update 200 | 197 | 95.68%-99.49% | 868.89 | 2.230 |
| E1 update 300, primary | 199 | 97.22%-99.91% | 463.95 | 1.308 |

### Task learning and checkpoint trajectory

E0 versus final E1 has 156 gains and zero regressions on the 200 paired
questions (exact McNemar p = 2.19e-47). Among 43 jointly correct questions,
median assistant-token change is -1446 (bootstrap 95%: -1864 to -1098), or
-77.83% (95%: -81.38% to -70.36%). This describes task-only learning; no E2
treatment effect has yet been measured. It is one training seed and one family.

The final third was useful in this seed: update 200 versus 300 has three gains
and one regression, with 196 jointly correct episodes. Their median token
change is -49.51% (paired bootstrap 95%: -52.56% to -44.57%). This updates the
earlier budget diagnostic: high sampled training success did not imply that
later response length was stable. Checkpoints remain descriptive repeated
observations, and update 300 remains primary.

### Training signal and off-target observations

Across training, 65.83% of prompt groups have zero task-reward variance. There
are 108 updates with zero logged gradient norm, including seven of the final
ten. Optimizer momentum and weight decay can still act at those steps. The
last ten captured batches solve 316/320 rollouts; all 40 groups have varying
lengths among successful responses. E2 therefore has a possible length signal
after task-reward saturation. This does not predict its effect, and without
the deferred placebo the comparison cannot isolate assignment from activation.

Final E1 has one wrong submission, four episodes with invalid actions, no
repeated actions, and no non-termination or generation-budget endings. Its
failure at index 123 attempted to fill label 35 instead of textbox 36, then
submitted with the country field empty. The other invalid-action cases (45,
56, 126) completed the required fields but also tried unnecessary fills on
non-input elements. These are recorded policy errors, not runtime failures.

Six efficient correct controls selected with RNG seed 17 (indices 17, 194,
106, 70, 111, 44) were inspected against their visible tables and actions. All
used valid short solutions. This E1 inspection was unmasked and aggregate
results were known; the declared masked stratified E1/E2 review remains due
after E2. The frozen-threshold underthinking flag on 98.49% of correct E1
episodes is not itself evidence of harm. No off-target equivalence is claimed.

### Costs and admission

Training consumed 28.134 allocated GPU-hours; the three evaluations consumed
7.471, for E1 total expenditure of 35.605. E0 plus E1 expenditure is 39.596
allocated GPU-hours. Checkpoint-worker costs are counted once; unpublished
nested temporary timers are excluded.

Relative to E0, final E1 saves a mean 47.83 policy-inference wall seconds per
episode across all 200 questions. Dividing E1 training wall seconds by that
saving gives a descriptive specialization break-even of about 2118 episodes.
This uses the same L4 and wall-time basis, excludes research evaluation costs,
and is a fixed-split point estimate, not a kernel-time or precision claim.

The hash-bound `e1-read-table-2-s4016/review.json` binds the raw evidence,
`technical_review.json`, `scientific_review.json` and audit sources. Integrity
passes, admitting only E2 weight 0.1 at seed 4016 from the original base. The
initial comparison still stops after E2 harvest and review for the user's
decision, regardless of result direction.

## Final E2 and initial-comparison review, 2026-09-27

E2 completed at 11:54:44 UTC with exit code zero. Its training and all three
checkpoint evaluations are harvested and reviewed. Only weight 0.1 and seed
4016 were run. No placebo, other dose, replication or E3 was launched.

### Integrity and evidence

All 25 E2 raw files, including both intermediate `checkpoint-evals/` trees,
match remote SHA256 hashes. Source hashes, the installed stack, original base
revision, launch manifest and frozen configurations match the declaration.
The E1/E2 configuration diff changes only experiment metadata and enabling the
successful-length component. Both train independently from base. Checkpoint
metadata confirms updates 100/200/300, and final/300 adapters have identical
hashes. E0/E1 receipts and every input they bind remain unchanged.

All 600 E2 evaluation trajectories pass the visible-table oracle, action/error
counts, stop records, report/sample alignment and recomputed metrics. Together
with the previous reviews, this covers all 1400 evaluation trajectories in the
trio. All evaluations share exactly the same 200 seeds and initial observations;
E0's fixed token thresholds are unchanged. All 300 E2 updates are finite.

The 416 captured E2 training trajectories across the 13 scheduled captures
pass prompt-group alignment, success/termination checks and raw/composed reward
arithmetic. Recounts with the saved tokenizer match all assistant-token counts.
An additional scalar reconstruction verifies the successful-peer mean,
population standard deviation with one-token floor, sigmoid cost, zero cost on
failures, and task reward minus 0.1 times cost. Both arms' captured formulas
were checked. No measurement or runtime code was changed.

Evidence is in `e2-read-table-2-s4016/technical_review.json`,
`scientific_review.json`, `initial_comparison.json`, the masked-review files,
and the hash-bound `review.json`. Reproduction scripts and remote evidence
are under `e0-e2-campaign-ops/`. Adapters remain on the GPU host.

### All declared observations

| Policy | Correct / 200 | Wilson 95% success interval | Mean assistant tokens on correct episodes | Evaluation GPU-hours |
|---|---:|---:|---:|---:|
| E0 | 43 | 16.37%-27.70% | 2027.12 | 3.991 |
| E1 update 100 | 183 | 86.81%-94.63% | 1506.58 | 3.933 |
| E1 update 200 | 197 | 95.68%-99.49% | 868.89 | 2.230 |
| E1 update 300 | 199 | 97.22%-99.91% | 463.95 | 1.308 |
| E2 update 100 | 184 | 87.40%-95.02% | 1094.62 | 3.153 |
| E2 update 200 | 199 | 97.22%-99.91% | 372.11 | 1.093 |
| E2 update 300 | 198 | 96.43%-99.73% | 228.68 | 0.760 |

Update 300 is primary. Means over each arm's successful subset are descriptive;
the declared compression estimate uses questions both arms solved correctly.
E0-to-E1 improvement is task-only learning and is not attributed to E2 shaping.

### Primary paired result: E2 versus E1, update 300

- 200 paired questions: one gain, two regressions, 197 jointly correct.
- Exact McNemar p = 1.0. This does not establish success-rate equality.
- One-sided 95% exact upper bound on E1-correct/E2-wrong probability: 3.114%.
  This passes the declared 5-percentage-point success-preservation margin.
- Median paired successful-token change: -44.93%, bootstrap 95% interval
  -47.33% to -42.75%. Its upper bound is below the declared -10% target.
- Median absolute token change: -182, bootstrap 95% interval -195 to -166.
  Mean paired absolute change is -234.67 tokens.
- All paired bootstrap intervals use 20,000 resamples and RNG seed 0.

This is meaningful compression with success preservation at the declared
margin, on this fixed in-family split. It does not mean zero harm or identical
accuracy. Both real regressions were manually inspected alongside E1:

| Episode index / seed suffix | E2 failure | E1 / E2 tokens |
|---|---|---:|
| 151 / 0151 | Enters `Arlin` instead of the table's `Arlyn`; year is correct, then submits | 397 / 237 |
| 182 / 0182 | Enters `Punjabi` from the Language row into Religion instead of `Judaism`, then submits | 627 / 227 |

The full seeds are 4016100151 and 4016100182. These are incorrect value copying
and row association with valid actions, not scorer or environment failures.
The latter error is already present at E2 update 200. E2's gain at index 123
fixes E1's invalid fill target and correctly submits Hinduism and Greece.

### Off-target review and interpretation

At update 300, both arms end all 200 episodes through environment completion.
Neither has voluntary stops, generation-cap endings or repeated actions.

| Event | E1 count | E2 count | Newly introduced | Removed | One-sided upper bound on new events, alpha 0.05/4 |
|---|---:|---:|---:|---:|---:|
| Wrong submission | 1 | 2 | 2 | 1 | 4.00% |
| Voluntary no-tool stop | 0 | 0 | 0 | 0 | 2.17% |
| Invalid-action episode | 4 | 0 | 0 | 4 | 2.17% |
| Repeated-action episode | 0 | 0 | 0 | 0 | 2.17% |

All four declared no-increase bounds are below 5 percentage points. The
familywise adjustment applies to these four one-sided bounds. They do not
establish exact equality of behavior or the absence of smaller harms.
The three-state verdict for **full behavioral equivalence is indeterminate**:
we can establish the declared no-increase margins, while the observed error
mix changes from invalid element targets to incorrect copied values. There is
no established harmful shift beyond the declared margin. Budget exhaustion
and non-termination remain separate descriptive measures, not added gates.

All trajectories were audited against visible task state. The declared
seed-17 stratification was applied to the pool of all seven evaluations in
E0/E1/E2 and checkpoint order: budget, voluntary, invalid/repeated, then
shortest-quartile correct controls. Six cases per stratum were sampled and
shuffled. The efficient-correct cutoff was 347 tokens. All 24 judgments were
written before opening the arm/step/index key; all labels agreed with visible
tables and actions. All six short controls use the two correct fills and Submit.

Prior knowledge is disclosed: E0/E1 aggregates and earlier E1 manual cases were
known, as was E2's final 0.990 accuracy from the completion log. This was masking
of case identity, not an independent blinded rater. After unmasking, all six
short controls happened to be E2; no replacement sample was chosen. Other
strata mostly contained E0 failures. Earlier unmasked E1 short controls and the
full trajectory audits supply additional coverage. Voluntary-stop examples
included false claims of submission after clicking a non-submit element.
The legacy unsupported-claim proxy does not capture that whole concept.

E2's frozen-threshold short-response flag is 100% among correct episodes.
The validated short controls show why that flag cannot be interpreted as harm.
An exploratory sensitivity split on E1's token-length quartiles yields median
successful-token changes of -34.14%, -41.53%, -48.60% and -59.62%. The two new
wrong submissions lie in quartiles 2 and 4. This is descriptive conditioning on
control behavior, not a causal adjustment or separate confirmatory tests.

### Training signal and the 300-update question

| Measure | E1 | E2 |
|---|---:|---:|
| Nonconstant-reward prompt groups | 34.17% | 99.08% |
| Updates with any nonconstant-reward group | 64.00% | 100.00% |
| Zero logged-gradient updates | 108/300 | 0/300 |
| Zero logged-gradient updates in final ten | 7/10 | 0/10 |
| Sampled task success in final ten updates | 98.75% | 99.38% |

The intervention changes both successful-length assignment and gradient
activation. Its total effect is measurable here; those mechanisms cannot be
separated without the deferred placebo. No mechanism-specific claim is made.

E2 update 200 to 300 loses one success while reducing tokens by a paired median
38.05% on 198 jointly correct questions (95% interval 36.92%-39.29% reduction).
E1's final third reduced its paired median by 49.51%. Thus the last 100 updates
were not inert, even after success nearly saturated. These repeated checkpoint
observations do not establish a universally sufficient shorter horizon. The
primary checkpoint stays at 300, and no best-checkpoint selection was made.

At update 100, E2 versus E1 has 13 gains and 12 losses: it meets the compression
target but not the success-preservation bound (9.54% upper bound). At update
200 it has three gains and one loss, a 2.35% upper bound and median token change
-55.12% (95% interval -59.68% to -51.42%). These are descriptive observations,
not extra replications or alternatives to the primary result. Intermediate
success regressions were also inspected; their wrong-field/label submissions,
missing submission and budget endings match the recorded policy behavior.

### Precision and multiplicity

The paired median percentage interval has width 4.58 percentage points in this
sample. The success allocation can tolerate at most four regressions while
meeting the 5pp one-sided margin; with zero regressions the 95% upper bound
would still be 1.49%. A nonsignificant McNemar test cannot establish equivalence.

For an explicit fixed-n sensitivity, model the number of discordant pairs as
Binomial(200, q), then apply the exact two-sided McNemar test at alpha 0.05 to
the direction of those flips. The historical development contrast had seven
flips in 200 (q = 3.5%, from archived `table_first_s4009_findings.md`), under a
different recipe; it is a sensitivity input, not current evidence. Even if all
flips favor one arm, maximum power is 70.38%, so an 80%-power minimum detectable
net success difference does not exist at that fixed q. At the realized three
flips/200, maximum power is only 8.24%. For q = 5% and 10%, the corresponding
80%-power net differences are approximately 4.43 and 6.40 percentage points.
These are retrospective sensitivity calculations, not prospective power or a
reason to extend the sample. Exact testing requires at least six one-direction
flips before two-sided significance, explaining the low-discordance limitation.

Only one E2 dose was executed. The declared individual intervals do not provide
joint coverage for all endpoints and repeated checkpoints. An exploratory
Bonferroni sensitivity across success preservation, compression and the four
off-target requirements still passes: success-loss upper bound 4.25%, token
interval -47.62% to -42.22%, and off-target upper bounds at most 4.25%. This
supports the initial verdict but does not solve the absence of training-seed
replication. Episode intervals condition on this seed and family; they do not
measure uncertainty across trained models.

### Costs and break-even

| Condition | Training GPU-hours | Research evaluation GPU-hours | Total GPU-hours |
|---|---:|---:|---:|
| E0 | 0 | 3.991 | 3.991 |
| E1 | 28.134 | 7.471 | 35.605 |
| E2 | 22.582 | 5.007 | 27.588 |
| Total | 50.716 | 16.469 | 67.185 |

All nine cost attempts have matched start/end records and completed status.
All 1400 evaluation episodes have timings. Each checkpoint worker is counted
once; nested temporary timers are excluded. These are one-L4 allocation/wall
hours, not kernel time, energy or money.

Across all 200 primary questions, mean policy-inference time is 67.08 seconds
for E0, 19.25 for E1 and 9.42 for E2. E2 saves 9.83 seconds per episode versus
E1 (paired bootstrap 95% interval 8.36-11.72), about 51.1% of E1 inference time.
On jointly correct questions the saving is 9.82 seconds (8.33-11.72). The
all-episode end-to-end saving is 9.86 seconds (8.37-11.76), so the inference
saving is not explained by newly failed episodes.

Using full E2 training wall seconds divided by mean all-episode policy-inference
seconds saved gives about 1410 episodes versus E0, or 8267 versus an already
available E1. Conditional inverse-bootstrap ranges are 1318-1513 and 6939-9728
respectively. They condition on the one measured training cost and do not
capture run-to-run hardware or training variation. E1's analogous E0 break-even
is about 2118 episodes. Research evaluation expenditure is excluded from these
deployment break-even ratios and reported separately above.

For the alternative decision of training E2 **instead of** E1, incremental
training cost is -19,989.66 seconds (-5.553 GPU-hours), with lower measured
inference cost too; there is no positive incremental-training cost to amortize
in this observed pair. This differs from paying for a new E2 model after E1's
training has already been spent. No universal speed guarantee follows from one
training run per arm.

### Decision boundary

The initial setup produces coherent, task-grounded learning and a positive
E2 compression result at the declared margins. Two genuine value-copying
regressions remain, and assignment versus gradient activation is unresolved.
The next action is the user's decision on whether to expand, adjust or stop.
No further GPU experiment is authorized by this result.
