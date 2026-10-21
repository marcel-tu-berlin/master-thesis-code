# Email-inbox-noscroll seed 4021: matched 0.1 placebo

Scientific analysis prepared 2026-10-22 Europe/Berlin from the completed,
unchanged inbox run. Update 300 remains primary. The audit-only recovery
passed on 2026-10-21 UTC; it did not rerun training or evaluation.

## Result

E2 uses a median 83.93% fewer assistant tokens than the shuffled-cost placebo
on their 195 jointly correct questions (95% interval 82.21% to 85.49% fewer).
All 195 pairs shorten. E2 solves 199/200 questions versus placebo's 196/200:
four gains and one regression, a paired difference of +1.5 percentage points
[95% bootstrap interval -0.5, +4.0], exact McNemar p=0.375. This does not
establish an accuracy advantage. The gross-regression upper bound is 2.35%,
which passes the declared 5% preservation margin, and compression clears 10%.

The behavioral result is unfavorable on two declared endpoints. E2 has an
invalid action in all 200 final episodes versus placebo's 40, and repeated
actions in 66 versus zero. Both new-event bounds exceed the 5% margin.
Thus E2 passes success preservation and compression against placebo, but does
not pass the full declared behavioral panel. High native success does not
make these extra invalid actions disappear.

Placebo itself improves over task-reward-only E1: 196 versus 183 successes,
fifteen gains and two regressions, +6.5 percentage points [2.5, 10.5],
McNemar p=0.00235. Its median paired token reduction is 49.50%
[43.10%, 53.03%] on 181 joint successes, with 168 shorter, one tied and twelve
longer. Its regression bound is 3.11%, but its invalid-action bound also fails.

The length-based assignment is associated with substantially stronger
compression and a different action pattern than shuffled assignment in this
seed. P also shows substantial compression relative to E1 in this seed. These observations do not isolate total reward variance,
gradient noise or an intent to ignore feedback. The comparison is exploratory,
made after the original findings were known. Intervals quantify held-out
question uncertainty within one training seed, not training-seed robustness.
Checkpoint observations are not independent replications, and nominal intervals
are not simultaneous campaign-wide confirmation.

## Design, provenance and recovery

The protocol is `docs/plans/placebo-l010-campaign.md`, decision 0028, and the
original inbox protocol with the inherited fixed margins. Runs are
`e2placebo-l010-email-inbox-noscroll-s4021-c4096` (P),
`e2-email-inbox-noscroll-s4021-c4096` (E2), and
`e1-email-inbox-noscroll-s4021-c4096` (E1).

Frozen E2/P configs differ only in experiment ID, description and
`successful_length.placebo: true`. Both use weight 0.1, fresh LoRA from pinned
Qwen3-1.7B, 300 updates, batch four, eight rollouts, naive_sum,
scale_rewards none, and unchanged optimizer settings and budgets:
4096 trajectory tokens including feedback, 4096 prompt, 8192 context, eight
turns. Retain the implemented batch/continuation semantics; this is not a
new hard per-action transcript cutoff. Original E0 thresholds remain
544.3 and 2475 tokens.

All comparisons use identical ordered seeds 4021100000 through 4021100199
and byte-identical initial observations at every checkpoint. The 416 captured
training seed/prompt/group/slot identities also align. Historical runtime
source hashes, Python 3.12.3, package pins, OpenEnv and episode-boundary stamps
match. All three runs used L4; original E1/E2 record kernel 6.8.0-139 and
placebo 6.8.0-146, both glibc 2.39. The kernel and run-date differences limit
timing attribution. The home-host differences of other families do not apply
to this inbox comparison.

The original audit stopped because its single-checkpoint body lookup lacked
the requested email bodies for two wrong-source forwards at update 100,
indices 79 and 199. The approved supplemental auditor restored the original
E1/E2 reference scope: manifest-bound E0 plus all three saved checkpoint
evaluations, with identical observations and conflicting bodies rejected.
Both outcomes remain wrong. No source, raw result, training run or evaluation
was rewritten. Original failed receipts remain preserved.

Canonical recovery authority is the run's `recovery_harvest_manifest.json`
and `audit-recovery/supplemental-pass.json`, not a fabricated normal
`review.json`. The 24 raw non-weight files and nine supplemental files were
hash-verified. The passing technical recovery covers all 300 finite updates,
416 captured rollouts, tokenizer recounts, seeded-shuffle replay, all 600
native evaluation audits and recomputed metrics, checkpoint/final adapter
identity, source/config/stack and four complete cost attempts. The recovery
and tabs launch gates passed independent Opus 5.5/xhigh review separately.

This scientific analysis rechecks comparator receipt hashes and the recovered
harvest, reruns all 600 new native evaluation audits and the capture arithmetic,
and reconciles action evidence against saved audits. It does not claim an
independent semantic audit of every training trajectory: captures cover only
selected updates, and oversized-turn feedback rollback limits observable
training state. All saved evaluation trajectories have complete native-audit
coverage. Original E1/E2 findings and artifacts are retained unchanged.

## Checkpoint and operation results

Success-conditioned means describe different subsets; compression estimates
below use only jointly correct pairs. Final Wilson 95% intervals are
E1 91.5% [86.81%, 94.63%], P 98.0% [94.97%, 99.22%], and
E2 99.5% [97.22%, 99.91%].

| Update | E1 / P / E2 correct out of 200 | E1 / P / E2 mean tokens on successes |
|---|---:|---:|
| 100 | 128 / 164 / 154 | 1554.57 / 1105.72 / 980.29 |
| 200 | 170 / 190 / 150 | 1329.12 / 1030.71 / 260.28 |
| 300 | 183 / 196 / 199 | 1475.91 / 860.01 / 121.75 |

| Contrast | Update | Gains / regressions | Joint successes | Median paired token change, 95% CI |
|---|---:|---:|---:|---:|
| E2-v-P | 100 | 21 / 31 | 133 | -19.24% [-28.31%, -3.90%] |
| E2-v-P | 200 | 3 / 43 | 147 | -72.50% [-75.02%, -71.39%] |
| E2-v-P | 300 | 4 / 1 | 195 | -83.93% [-85.49%, -82.21%] |
| P-v-E1 | 100 | 51 / 15 | 113 | -49.32% [-55.96%, -38.51%] |
| P-v-E1 | 200 | 26 / 6 | 164 | -29.71% [-33.41%, -21.33%] |
| P-v-E1 | 300 | 15 / 2 | 181 | -49.50% [-53.03%, -43.10%] |

E2's update-200 accuracy deficit versus P is retained: -20 percentage points
[-26, -14], McNemar p=4.62e-10. Its 43 regressions comprise 25 budget stops,
five turn-cap stops and thirteen incorrect terminal actions. At update 100,
31 regressions comprise eighteen budget stops, four voluntary stops and nine
incorrect terminal actions. The final reversal is not a checkpoint-selection
rule. Final E2-minus-P absolute median token change is -703 [-793, -641];
paired means are 856.89 for P and 121.07 for E2. Final P-minus-E1 absolute
median is -644 [-715, -589], with paired means 1468.06 for E1 and 826.49 for P.

| Operation | Questions | E1 / P / E2 final correct | E2-v-P gains / losses | P-v-E1 gains / losses |
|---|---:|---:|---:|---:|
| reply | 54 | 51 / 51 / 53 | 3 / 1 | 2 / 2 |
| forward | 60 | 47 / 59 / 60 | 1 / 0 | 12 / 0 |
| delete | 41 | 41 / 41 / 41 | 0 / 0 | 0 / 0 |
| important | 45 | 44 / 45 / 45 | 0 / 0 | 1 / 0 |

Operation medians for E2-v-P are -83.40% reply, -87.82% forward,
-82.16% delete and -71.13% important. Corresponding P-v-E1 medians are
-39.45%, -27.35%, -59.70% and -73.82%. These small subgroups are descriptive;
the fixed 200-question sample was not sized for separate equivalence claims.
Per-operation paired intervals are retained in `behavior-review.json`.

## Task-grounded behavior

Every new saved evaluation was audited against the requested operation,
sender, exact text or forward body, visible state and executed actions.
The 98 regression pairs across both contrasts and all three checkpoints
were inspected through their task-state/action records; complete records and
paired successes are retained. An incorrect send, a rejected call, voluntary
stopping and budget exhaustion remain separate outcomes. Indices are zero-based.

The four final P failures are concrete:

- Index 67 sends `Nisl, Orci.` instead of the requested `Nisl,. Orci.` to Paule.
- Index 129 forwards Betti's correct body to `Christian` instead of `Christean`.
- Indices 169 and 182 eventually fill the exact requested reply text after
  rejected fills, but exhaust the trajectory budget before sending.

E2 solves all four. Its sole regression against P is index 91, sending
`Feugiat` instead of `Feugiat.` to Iolande. P sends the exact text. The two P
regressions against E1 are the unsent replies at 169 and 182; the two wrong
P sends are shared with E1, not new incorrect-terminal events.

Final P has 42 invalid calls in 40 episodes: 29 non-input fill errors,
eleven click timeouts and two fill timeouts. Thirty-eight of those episodes
still succeed, and two end at the budget. There are no repeated calls or
voluntary stops. Final E2 has 206 invalid calls in all 200 episodes, including
199 successful episodes, and 66 exact repeats. All 66 repeats occur inside
a generated batch, before feedback on the preceding call can inform a new
decision. They do not demonstrate ignoring observed error feedback. E2's
errors are mainly stale or absent bids after an opening click; later calls
usually recover and complete the task. Mean actions are E1 3.155, P 3.120,
E2 4.180. All 200 final E2 episodes contain a multi-call turn, versus 14 for P.

Earlier P checkpoints have 24 budget stops, two voluntary stops and ten wrong
terminal actions at update 100, then four budget stops, one voluntary stop
and five wrong terminal actions at 200. Failures include wrong-source
forwards, wrong or empty recipients, empty replies, text copying errors,
and repeated stale-bid clicks. Among secondary regressions specifically,
update 100 has ten budget stops, one voluntary stop and four wrong terminal
actions; update 200 has four budget stops and two wrong terminal actions.
The primary intermediate regressions include wrong recipients, modified
forward bodies, empty replies, unsupported tool names, non-input fills and
budget stops. These are policy outcomes, not runtime failures.

### Declared final off-target bounds

"New" is present in the named arm and absent in its paired reference.
Bounds are one-sided exact 98.75%, alpha 0.05/4, against the fixed 5% margin.

| Contrast / event | Reference -> arm episodes | New / resolved | Upper bound |
|---|---:|---:|---:|
| E2-v-P: wrong submission | 2 -> 1 | 1 / 2 | 3.15% |
| E2-v-P: voluntary no tool | 0 -> 0 | 0 / 0 | 2.17% |
| E2-v-P: invalid action | 40 -> 200 | 160 / 0 | 85.98% |
| E2-v-P: repeated action | 0 -> 66 | 66 / 0 | 40.97% |
| P-v-E1: wrong submission | 12 -> 2 | 0 / 10 | 2.17% |
| P-v-E1: voluntary no tool | 2 -> 0 | 0 / 2 | 2.17% |
| P-v-E1: invalid action | 10 -> 40 | 38 / 8 | 26.03% |
| P-v-E1: repeated action | 1 -> 0 | 0 / 1 | 2.17% |

E2-v-P fails the invalid and repeated-action margins; P-v-E1 fails the invalid
margin. The other listed bounds pass. Non-termination is separately E1 5/200,
P 2/200, E2 0/200. Final E2/P stop transitions are 198 env-done to env-done
and two P budget stops to E2 env-done. Passing selected bounds does not
establish full behavioral equivalence.

### Controls and coverage

The retained seed-17 procedure was applied to the nine E1/E2/P evaluations,
with the original priority budget, voluntary, invalid/repeated, then efficient
correct. Its correct-response first quartile is 297.5 tokens; available
strata contain 146, 20, 737 and 93 episodes. Six per stratum give 24 cases.
Arm/checkpoint/seed/index were hidden until judgments were saved, but outcome,
reward, stop, length and error fields remained visible, as did deterministic
audit summaries in the compact view. This is a consistency review, not an
independent blinded label validation; aggregates and earlier cases were known.
All 24 judgments agree with the visible labels. The six efficient controls
are valid single-click delete/important actions: three E2 cases all come
from update 100, and the three P cases comprise one at 100 and two at 300.
No final E2 episode is eligible for the error-free efficient stratum, because
all contain invalid actions. These six cases do not cover reply/forward. Eight supplementary unmasked controls,
selected as the shortest correct final response per operation in P and E2,
cover all four operations. P uses the correct direct row click or exact
reply/forward sequence; E2 completes the same goals after an extra rejected
call. The labels are supported by the exhaustive deterministic evaluation
audit, not by agreement in this selected sample.

The frozen-threshold short-response flag covers 2.73% of E1 successes,
34.18% of P and 100% of E2. Valid short controls show that this flag alone is
not evidence of harmful underthinking. No broad intent, reward-hacking or
semantic feedback-neglect rate is claimed. The repeat metric counts exact
consecutive calls, not all unnecessary actions.

## Training and cost

| Measure | E1 | P | E2 |
|---|---:|---:|---:|
| Mean active prompt-group fraction | 49.67% | 95.58% | 92.17% |
| Updates with any live group | 90.33% | 100% | 100% |
| Zero-gradient updates / 300 | 29 | 0 | 0 |
| Correct rollouts in last ten captured updates / 320 | 297 | 317 | 318 |
| Task-variable groups in that tail / 40 | 15 | 3 | 2 |
| Composed-variable groups in that tail / 40 | 15 | 40 | 26 |

348 of 416 captured P rollout assignments change under the replayed shuffle.
The stored raw component values remain unshuffled; the composed reward contains
the actual assignment. Shuffling preserves each group's cost multiset,
including failure zeros, but not its alignment with correctness, total reward
variance, state visitation or gradient noise. Both P and E2 activate every
update, but their group activity and final-tail structure differ. Their
comparison is not an exact optimization-noise control.

Four complete ledger attempts per arm cover training and evaluations at
100/200/300, with one allocated GPU each and no unmatched or failed attempt.
Each end record is counted once, without duplicating temporary checkpoint timers.

| Arm | Training GPU-hours | Evaluation GPU-hours | Total GPU-hours |
|---|---:|---:|---:|
| E1 | 38.235 | 11.759 | 49.994 |
| P | 33.649 | 8.551 | 42.201 |
| E2 | 33.026 | 4.690 | 37.717 |

P took about 42 hours 12 minutes in total, including 33 hours 39 minutes of
training. These are wall-time allocations, not energy or monetary cost.
E0, deployment, preflight and audit/review work are excluded from this table.

Final E2 saves 31.86 inference seconds per question versus P [29.02, 34.82],
or 31.00 on joint successes [28.20, 33.87]. P saves 27.12 seconds per question
versus E1 [23.83, 30.40], or 26.33 on joint successes [23.44, 29.26].
Both all-question and successful-pair views are retained so failed responses
do not supply the successful-compression claim.

The total-episode view includes tool execution and other non-inference overhead.
Mean total times are E1 69.22, P 42.20 and E2 11.73 seconds per question.
Non-inference time is respectively 4.40, 4.51 and 5.90 seconds. E2 therefore
uses about 1.39 additional non-inference seconds per question versus P,
consistent with its extra actions and timeouts; the residual is not an isolated
measurement of tool time alone. Paired total-time savings are E2-v-P
30.46 seconds [27.59, 33.49] over all questions and 29.60 [26.75, 32.52] over
joint successes. P-v-E1 total savings are 27.02 [23.69, 30.33] and
26.27 [23.33, 29.24], respectively.

Charging full arm training against all-question inference-only savings gives
about 3732 episodes for E2 versus P and 4466 for P versus E1. Using total
episode-time savings instead gives about 3903 and 4483 episodes.
E2 training costs 0.623 fewer allocated GPU-hours than P; P costs 4.586 fewer
than E1. Neither contrast has a positive incremental training cost to repay;
a zero in the incremental calculation does not mean training is free.
Research evaluation costs stay separate. These are descriptive same-L4 timing
calculations conditional on kernel, load and run-date differences, not a
causal speed or deployment guarantee.

## Evidence and review status

Evidence is in
`pipeline/runs/placebo-l010-20261018-ops/inbox-analysis-20261022/`:
`analyze.py`, `analysis.json`, `review_cases.py`, `behavior-review.json`,
seed-17 selection and judgments, supplementary operation controls and
`validate.py` / `validation.json`. Analysis uses the retained paired-statistic
implementation, 20000 bootstrap resamples and seed 0. Execution, negative
checks, deterministic reproduction and paired-statistic test results are
retained with exit codes and hashes in `checks.json` and its logs.
The original analysis directory `inbox-analysis-20261021/` preserves the
first report as `findings-reviewed-v1.md`, all original outputs and validation.
The first counterpart invocation failed without a verdict (wrapper code 2,
underlying CLI exit 1); its cause was not established. The next completed
Opus 5.5/xhigh review returned four findings (code 1), accepted in the
correction directory's `review-disposition.md`: total latency coverage,
one causal sentence, review-status wording and efficient-control checkpoints.
These are scientific reporting corrections, not changed experimental outcomes.
The corrected analysis is versioned separately and has new validation.

`review-status.json` in the correction directory is the current scientific
review authority: it distinguishes pending, findings and accepted states and
points to the applicable prompt, output, start-time input hashes and terminal
receipt. A pending or failed invocation is not a pass. The earlier technical
recovery pass is not a substitute for scientific review.

Task 9 remains open until tabs findings and all five counterpart reviews are
verified and reported. Tabs is the remaining authorized cell; no further
experiment follows from these results.
