# Inbox seed 4240: shorter responses with lower final success

Prepared 2026-11-09 from the completed, hash-verified second inbox seed.
Scientific counterpart review passed on 2026-11-09 with Claude Opus 5.5/xhigh.
The added cost interpretation and supplemental validation record were also
reviewed. Update 300 remains the declared primary endpoint.

## Result

E2 finishes with 168/200 successes, versus E1's 199/200 and matched placebo's
189/200. It compresses successful responses in both paired contrasts, but
fails the declared success-preservation requirement. The original seed's
success-preserving compression does not repeat at this seed's final endpoint.
The full off-target panel also fails. These are scientific outcomes of runs
that passed their technical checks; they do not authorize a restart.

| Final contrast | Gains / regressions | Joint successes | Median paired token change, 95% interval | Net success change, percentage points, 95% interval | Gross-regression upper 95% bound |
|---|---:|---:|---:|---:|---:|
| E2 versus E1 | 0 / 31 | 168 | -74.38% [-76.72%, -72.70%] | -15.5 [-20.5, -10.5] | 20.34% |
| E2 versus placebo | 6 / 27 | 162 | -44.24% [-52.23%, -37.18%] | -10.5 [-16.0, -5.0] | 18.13% |
| Placebo versus E1 | 0 / 10 | 189 | -45.09% [-54.88%, -36.61%] | -5.0 [-8.0, -2.0] | 8.33% |
| E1 versus E0 | 86 / 0 | 113 | -22.95% [-31.00%, -11.44%] | +43.0 [36.0, 50.0] | 1.49% |

The fixed preservation margin is 5%, and compression requires the upper end
of the token-change interval to be below -10%. Both primary contrasts pass
compression and fail preservation. Exact paired McNemar p-values are
9.31e-10 for E2 versus E1 and 0.000324 for E2 versus placebo. All 168 jointly
correct E2/E1 responses shorten; E2/placebo has 155 shorter, one equal and six
longer responses. The compression claim excludes failures and does not imply
that shortening compensates for lost task completion.

Placebo versus E1 also passes compression and fails success preservation:
its 10 regressions have an 8.33% upper bound, with McNemar p=0.00195.
Thus observed success loss is not confined to true length assignment,
although E2 is worse in this seed. This does not identify a causal mechanism.
Final Wilson 95% success intervals are E0 56.5% [49.57%, 63.18%],
E1 99.5% [97.22%, 99.91%], placebo 94.5% [90.42%, 96.90%], and
E2 84.0% [78.29%, 88.43%]. Absolute paired median token changes are
E2 minus E1 -1029 [-1093, -937.5], E2 minus placebo -220 [-392, -123],
placebo minus E1 -487 [-561, -400], and E1 minus E0 -358 [-498, -152].

Intervals use the retained paired analysis, 20000 bootstrap resamples and
RNG seed 0. They quantify held-out question uncertainty within this seeded
experiment. They do not estimate training-seed uncertainty. Checkpoints are
repeated observations of the same training run, not independent replications.
These exploratory comparisons were declared after the original results;
nominal intervals are not simultaneous campaign-wide confirmation.

## Design and verification

The protocol is `docs/plans/seed-replication-20261025.md`, decision 0029, with
the inherited inbox margins and accepted batch/continuation semantics.
Runs are `e0-email-inbox-noscroll-s4240-c4096`,
`e1-email-inbox-noscroll-s4240-c4096`,
`e2-email-inbox-noscroll-s4240-c4096`, and
`e2placebo-l010-email-inbox-noscroll-s4240-c4096`.

All conditions use the same ordered held-out seeds 4240100000 through
4240100199 and identical initial observations. Each trained arm starts
independently from pinned Qwen3-1.7B with fresh LoRA, 300 updates, batch four,
eight rollouts, naive_sum and scale_rewards none. Budgets remain 4096
trajectory tokens including feedback, 4096 prompt, 8192 context and eight
turns. E2 and placebo use dose 0.1. Frozen E2/E1 configs differ only in
identifiers/descriptions and the enabled length reward; E2/placebo differ
only in identifiers/descriptions and shuffled assignment. Fresh E0 supplies
the fixed token thresholds, 688.2 and 2439, used by all checkpoint reports.

The local analysis verifies all four harvest manifests against passing
technical receipts and the bound inbox deployment manifest. It re-audits
all 2000 saved evaluations using the retained native auditor. Each arm's
forward-body lookup pools only its E0 reference and its own saved checkpoints,
matching the approved technical audit scope; conflicting bodies are rejected.
All re-audits equal the saved audits. Ten corrupted-success-label checks are
rejected. Config comparisons, 900 optimizer records, 1248 captured rollouts,
capture seed/prompt/group/slot alignment, reward arithmetic and all 13 cost
attempts pass. Deterministic regeneration reproduces both result artifacts
byte-for-byte; the retained paired-statistic suite passes 30 tests.

Remote technical receipts additionally cover tokenizer recounts of the 416
captured training completions per trained arm, metric recomputation,
source/config/stack integrity and final/update-300 adapter identity.
Held-out assistant token counts are checked for turn-sum consistency and
limits, but are not independently re-tokenized. This is narrower than the
evaluation-trajectory recount coverage described for the original inbox
comparison. Weights remain on the L4; this local analysis does not re-download
or independently re-check weights. Saved training captures cover selected
updates and can omit oversized final feedback through the pinned rollback
behavior. The local checks do not claim exhaustive semantic validation of
all training trajectories. No experiment source or raw artifact was changed.

All new arms ran on L4 GPU 1 with driver 580.178.04, kernel 6.8.0-146 and the same recorded
Python/package/OpenEnv stack. Original seed 4021 E1/E2 used kernel 6.8.0-139,
while its placebo used -146. The original runs' driver versions have not been
established in this analysis. These historical timing and host limitations
remain; adding matched L4 seeds does not erase them. The home hardware
limitations in other families remain separate.

## Retained checkpoint trajectory

E0 solves 113/200 questions. Success-conditioned means below describe each
arm's own solved subset; the paired estimates above use their intersection.

| Update | E1 / placebo / E2 correct | E1 / placebo / E2 mean tokens on successes |
|---|---:|---:|
| 100 | 128 / 167 / 190 | 1343.79 / 630.41 / 848.49 |
| 200 | 182 / 196 / 196 | 1603.73 / 628.90 / 522.76 |
| 300 | 199 / 189 / 168 | 1390.35 / 816.67 / 313.50 |

| Contrast | Update | Gains / regressions | Joint successes | Median paired token change, 95% interval |
|---|---:|---:|---:|---:|
| E2 versus E1 | 100 | 67 / 5 | 123 | -41.67% [-47.26%, -33.15%] |
| E2 versus E1 | 200 | 16 / 2 | 180 | -68.46% [-70.99%, -65.46%] |
| E2 versus placebo | 100 | 29 / 6 | 161 | +26.41% [16.55%, 36.88%] |
| E2 versus placebo | 200 | 3 / 3 | 193 | -12.63% [-16.58%, -7.35%] |
| Placebo versus E1 | 100 | 45 / 6 | 122 | -54.99% [-58.53%, -46.47%] |
| Placebo versus E1 | 200 | 15 / 1 | 181 | -64.15% [-67.07%, -60.89%] |

| Update | E1 Wilson 95% | Placebo Wilson 95% | E2 Wilson 95% |
|---|---:|---:|---:|
| 100 | [57.14%, 70.33%] | [77.73%, 88.00%] | [91.04%, 97.26%] |
| 200 | [86.22%, 94.23%] | [94.97%, 99.22%] | [94.97%, 99.22%] |
| 300 | [97.22%, 99.91%] | [90.42%, 96.90%] | [78.29%, 88.43%] |

E2's success falls from 196 to 168 between updates 200 and 300 as its
successful responses shorten further. Placebo also falls, from 196 to 189,
while E1 improves. Update 200 cannot replace update 300 after observing this
trajectory. E2's update-200 token interval versus placebo also fails the
declared 10% compression threshold despite its negative median.

## Task-grounded behavior and coverage

| Operation | Questions | E1 / placebo / E2 final correct | E2 versus E1 gains / losses | E2 versus placebo gains / losses |
|---|---:|---:|---:|---:|
| Reply | 49 | 48 / 40 / 25 | 0 / 23 | 4 / 19 |
| Forward | 55 | 55 / 53 / 47 | 0 / 8 | 2 / 8 |
| Delete | 48 | 48 / 48 / 48 | 0 / 0 | 0 / 0 |
| Important | 48 | 48 / 48 / 48 | 0 / 0 | 0 / 0 |

All final E2 failures occur in reply or forward tasks. Of its 32 failures,
21 stop without a tool call, eight reach the turn cap, one reaches the token
cap and two submit incorrect forwards. Its 31 regressions against E1 comprise
20 no-tool stops, eight turn-cap stops, one token-cap stop and both incorrect
forwards. Against placebo, 27 regressions comprise 18 no-tool stops, six
turn-cap stops, one token-cap stop and both incorrect forwards.

At indices 49 and 189, E2 forwards from the requested sender to the correct
recipient but replaces the required original body with different text. E1 and
placebo preserve the original body on both questions. Other failures include
fills targeting non-input elements, unsupported `type` calls and repeated
clicks before stopping. These are task-policy errors, not infrastructure faults.

E2 has 65 invalid actions in 52 episodes and 53 exact repeated actions in 19
episodes. All 53 repeats occur across turns, with earlier feedback available;
there are no final E2 multi-call turns. This differs from original seed 4021,
whose final E2 repeats were inside precommitted batches. Availability of
feedback does not establish comprehension, intent or semantic feedback neglect.
Mean actions are E1 2.58, placebo 2.50 and E2 3.985. Non-termination is
respectively 1, 11 and 30 episodes. Placebo's 11 failures are ten token-cap
stops and one no-tool stop; E1's only failure is a token-cap stop.

Complete final stop transitions below run from reference to named arm.
Zero entries are shown explicitly; an env-done transition can still contain
an incorrect terminal action.

| Reference -> arm stop | E2 versus E1 | E2 versus placebo | Placebo versus E1 | E1 versus E0 |
|---|---:|---:|---:|---:|
| env_done -> env_done | 170 | 164 | 189 | 134 |
| env_done -> hit_generation_cap | 1 | 1 | 9 | 0 |
| env_done -> max_turns | 8 | 6 | 0 | 0 |
| env_done -> no_tool_call | 20 | 18 | 1 | 0 |
| hit_generation_cap -> env_done | 0 | 5 | 0 | 58 |
| hit_generation_cap -> hit_generation_cap | 0 | 0 | 1 | 0 |
| hit_generation_cap -> max_turns | 0 | 2 | 0 | 0 |
| hit_generation_cap -> no_tool_call | 1 | 3 | 0 | 0 |
| no_tool_call -> env_done | 0 | 1 | 0 | 7 |
| no_tool_call -> hit_generation_cap | 0 | 0 | 0 | 1 |

The declared event bounds count an event present in the arm and absent in
its paired reference. They are one-sided exact 98.75% bounds, alpha 0.05/4,
against the original 5% margin.

| Contrast / event | Reference -> arm episodes | New / resolved | Upper bound |
|---|---:|---:|---:|
| E2 versus E1: wrong submission | 0 -> 2 | 2 / 0 | 4.00% |
| E2 versus E1: voluntary no tool | 0 -> 21 | 21 / 0 | 16.37% |
| E2 versus E1: invalid action | 3 -> 52 | 51 / 2 | 33.08% |
| E2 versus E1: repeated action | 0 -> 19 | 19 / 0 | 15.18% |
| E2 versus placebo: wrong submission | 0 -> 2 | 2 / 0 | 4.00% |
| E2 versus placebo: voluntary no tool | 1 -> 21 | 21 / 1 | 16.37% |
| E2 versus placebo: invalid action | 15 -> 52 | 49 / 12 | 32.01% |
| E2 versus placebo: repeated action | 2 -> 19 | 18 / 1 | 14.58% |
| Placebo versus E1: wrong submission | 0 -> 0 | 0 / 0 | 2.17% |
| Placebo versus E1: voluntary no tool | 0 -> 1 | 1 / 0 | 3.15% |
| Placebo versus E1: invalid action | 3 -> 15 | 14 / 2 | 12.15% |
| Placebo versus E1: repeated action | 0 -> 2 | 2 / 0 | 4.00% |

Both primary contrasts fail the no-tool, invalid-action and repeated-action
margins. Their wrong-submission bound passes, which does not establish full
behavioral preservation. Placebo versus E1 fails the invalid-action margin.

All saved evaluations have deterministic task-state audit coverage. All 91
success-regression pairs across checkpoints and contrasts were inspected
through their compact goals, action targets, errors, terminal states and
stops. They contain 58 distinct failed arm traces, some shared by contrasts;
the paired references have identical goals and native-audited valid outcomes.
Per-case judgments and their mapping to all 91 pairs are retained. This
inspection used known aggregates and visible native-audit labels; it is not
independent semantic validation.

The original seed-17 stratified sample was also run on the nine new trained
evaluations: six cases each from budget, voluntary, invalid/repeated and
efficient-correct strata. The success-token first quartile is 340.5; available
strata contain 97, 57, 223 and 364 episodes. All 24 compact-view judgments
agree with the task evidence. Arm/checkpoint/index/seed were hidden until
judgments were saved, but outcomes, metrics and native audit summaries were
visible, and aggregates and regression traces were known. Cases 12 and 18
were recognized. This is a consistency review, not an independent blinded
validation. The six efficient cases cover only delete and important tasks:
two final E2 cases and four placebo cases (one at 100, one at 200, two at 300).

The shortest correct final response per operation, with ties broken by index,
provides 12 supplementary descriptive controls across E1/E2/placebo. All preserve the requested
task without invalid or repeated actions. This confirms that short successful
responses exist; selected controls do not estimate an error rate. No masked,
independent semantic label validation is claimed. Unknown broader behavioral
labels remain unknown, and the frozen short-response flag is not a harm label.

## Training and costs

| Measure | E1 | E2 | Placebo |
|---|---:|---:|---:|
| Mean active prompt-group fraction | 33.25% | 97.83% | 97.33% |
| Updates with any live group | 71.67% | 100% | 100% |
| Zero-gradient updates / 300 | 85 | 0 | 0 |
| Correct rollouts in final ten captured updates / 320 | 309 | 306 | 309 |
| Task-variable groups in that tail / 40 | 3 | 8 | 6 |
| Composed-variable groups in that tail / 40 | 3 | 40 | 40 |

The seeded placebo replay changes 332 of 416 captured assignments. It preserves
each group's length-cost multiset, including failure zeros, but does not fix
alignment with success, total reward variance, state visitation or gradient
noise. Increased gradient availability remains part of the E2/E1 contrast;
placebo is not an exact optimization-noise control.

| Arm | Training GPU-hours | Evaluation GPU-hours | Total GPU-hours |
|---|---:|---:|---:|
| E0 | 0 | 2.707 | 2.707 |
| E1 | 35.348 | 11.924 | 47.271 |
| E2 | 29.331 | 4.901 | 34.231 |
| Placebo | 29.057 | 6.066 | 35.122 |
| Total | 93.735 | 25.597 | 119.333 |

All 13 phase attempts are complete and counted once, with one allocated GPU.
Costs exclude deployment, preflight and local analysis/review. These are
wall-time allocations, not measured energy or monetary costs.

Mean final inference / total episode times are E1 58.52 / 62.34 seconds,
E2 15.08 / 20.30, and placebo 39.60 / 43.36. E2 saves paired mean inference
time of 43.44 seconds [40.72, 46.20] versus E1 and 24.52 [20.51, 28.83]
versus placebo over all questions. Corresponding total-time savings are
42.04 [39.33, 44.84] and 23.06 [19.11, 27.32]. On jointly successful questions,
E2's inference savings are 41.46 [38.51, 44.50] versus E1 and
17.47 [14.55, 20.54] versus placebo; total-time savings are
40.04 [37.11, 43.09] and 16.10 [13.24, 19.14], respectively.
The lower joint-success savings, especially against placebo, show why the
all-question view cannot stand in for successful-behavior efficiency.
E2 has more non-inference time
per episode: 5.22 seconds versus E1's 3.82 and placebo's 3.76. That residual
includes tool and other overhead; it is not isolated tool time.

All-question time amortization of all E2 training is about 2431 episodes versus E1
and 4307 versus placebo using inference-only savings, or 2511 and 4579 using
total episode time. E2 costs 6.017 fewer training GPU-hours than E1 and 0.274
more than placebo; the latter incremental time amortizes in about 40 episodes.
These all-question estimates include failed and budget-stopped episodes.
Using joint-success savings as a sensitivity gives about 2547 episodes versus
E1 and 6045 versus placebo for full inference-only amortization, and about
56 episodes for incremental training versus placebo. Full training amortization
using joint-success total-time savings is about 2637 and 6557 episodes,
respectively. These conditional
sensitivities do not predict how many attempted tasks a deployment would need.
These calculations do not value errors, exclude evaluation costs and do not
establish deployment utility when E2 completes fewer tasks. Their timing
interpretation remains conditional on hardware, kernel, load and run date.

In these measured evaluation runs, E1 has no finite inference-time break-even
against E0: paired mean inference
savings are -14.56 seconds [-18.32, -10.78] over all questions and
-16.11 [-21.18, -11.25] over joint successes. E1's positive training cost is
not recovered by measured inference-time savings in this comparison. This
latency observation does not value E1's substantial task-success gain.
Its sign is conditional on hardware, serving configuration, load and run date;
these results do not attribute it to specialization.

## Seed robustness and review status

Original seed 4021 had E1/placebo/E2 success 183/196/199, and E2 passed
success preservation and compression in both primary comparisons while
failing behavioral margins. Seed 4240 has 199/189/168: compression repeats,
success preservation does not, and behavioral margins still fail with a
different error pattern. This is evidence of seed-sensitive outcomes, not a
precise training-seed variance estimate or a causal explanation of instability.
The seed changes both training randomness and generated question sets; this
replicates the full seeded experiment and does not isolate optimizer randomness.
Reply/forward/delete/important counts change from 54/60/41/45 in seed 4021 to
49/55/48/48 here, and E0 success changes from 103 to 113. Question set and
operation mix therefore remain part of the cross-seed comparison.
Seed 4241 remains queued. No final three-seed family conclusion is available;
do not pool episodes as seed replications or select the more favorable seed.

Evidence is in
`pipeline/runs/seed-replication-20261025-ops/inbox-s4240-analysis-20261109/`:
`analyze.py`, `analysis.json`, `behavior-review.json`, `case-summary.json`,
its retained generator `review_evidence.py`, regression-inspection evidence
and judgments, seed-17 sample/compact views/judgments, execution/regeneration
logs, paired tests and versioned validation receipts. validate_evidence.py and
validation-v4.json/log retain the exact successful sample-reproduction check.
The supplemental validator initially assumed table-style judgment dictionaries;
its failed v3 check is preserved, and the corrected validator reads the existing
inbox judgment lists. No judgments or numerical artifacts changed. Existing
family helpers are imported; their old analyses are not executed again.
The scientific review status and any terminal counterpart receipt belong
in this directory. A technical pass is not a scientific review pass.
Task 2 remains open until the full campaign and its reports are verified.
