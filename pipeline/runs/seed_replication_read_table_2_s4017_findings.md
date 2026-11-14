# Read-table-2 seed 4017 replication

Analyzed 2026-10-31. The four-condition seed group completed on 2026-10-31
at 03:09 UTC. All four runs passed technical admission and their non-weight
artifacts were harvested and hash-verified. Update 300 remains primary.
Independent counterpart review completed on 2026-11-09 with Claude Opus
5.5/xhigh after reporting and provenance corrections. The final correction
passed review; all earlier verdicts, the original authentication failure and
exact reviewed versions remain preserved. The technical queue remains active.

## Result

The direction of successful-response compression and clearance of the declared
10% criterion repeat, but the full original endpoint result does not.
E2 solves 192/200 questions versus E1's 197/200 and
placebo's 191/200. On joint successes, E2 uses a median 92.01% fewer assistant
tokens than E1 and 89.32% fewer than placebo. Every joint-success pair is
shorter in both contrasts.

Against E1, six gross regressions give a one-sided 95% upper bound of 5.83%,
above the declared 5% preservation margin. Preservation is therefore not
established. This is not proof that the population loss exceeds 5%. Against
placebo, four regressions give a 4.52% upper bound, passing that margin.
Both contrasts fail to establish the declared wrong-submission no-increase
margin: six newly introduced wrong submissions give a 6.96% upper bound.
The other three declared behavior bounds pass. Full behavioral equivalence
remains indeterminate; a smaller observed net loss does not erase real errors.

The original seed 4016 passed the declared compression, preservation and
behavior criteria. This second seed supports compression across both observed
table seeds, while qualifying the original preservation/behavior result.
Seed 4018 is still queued. No three-seed robustness claim or cross-family
pooled verdict is available. Question-level intervals below condition on one
trained seed and do not estimate variation across trained models. These are
exploratory replications declared after the original outcomes, not confirmation.
Checkpoints are repeated observations of the same training run, not independent
replications. Apart from the four adjusted off-target bounds, the nominal
intervals do not provide simultaneous coverage across endpoints, contrasts or
checkpoints. The seed changes training randomness and the train/eval question
block together; this is not optimizer-seed uncertainty alone.

Compression magnitude is not stable across the two observed seeds. E2 versus
E1 changes from -44.93% at seed 4016 to -92.01% here. Unpaired E1 mean success
tokens change from 463.95 to 1772.18 (about 3.8 times), while E2 changes from
228.68 to 138.51. E1 versus E0 paired compression changes from -77.83%
[-81.38%, -70.36%] to -11.98% [-19.14%, +2.94%]. Placebo versus E1 changes
from +156.76% to -22.41%; here placebo clears the compression criterion but
fails preservation. These are descriptive control-arm and placebo shifts,
not evidence that any one factor caused the cross-seed difference.

## Design and verification

The declared protocol is `docs/plans/seed-replication-20261025.md`, decision
0029. Runs are `e0-read-table-2-s4017`, `e1-read-table-2-s4017`,
`e2-read-table-2-s4017` and `e2placebo-l010-read-table-2-s4017`.
All trained arms start independently from the pinned Qwen3-1.7B with fresh
LoRA. They retain 300 updates, batch four, eight rollouts, `naive_sum`,
`scale_rewards: none`, weight 0.1, and 4096 trajectory / 4096 prompt / 8192
context / eight-turn limits. Native batch/continuation semantics are retained.

Frozen E1/E2 configs differ only in ID, description and length-cost enablement;
E2/placebo differ only in ID, description and shuffled assignment. All ten
evaluations use the same 200 questions, seeds 4017100000-4017100199, and
identical initial observations. All trained evaluations use fresh E0's fixed
thresholds: 1164.8 and 4096 tokens. These legacy percentile flags are not
validated harmful-underthinking labels.

The local analysis rechecks all harvested hashes against the remote technical
receipts, reruns the retained native table audit on all 2000 new episodes,
checks all 900 optimizer records and replays the reward arithmetic for all
1248 captured training rollouts. It checks paired training question/group/slot
identities and complete cost ledgers. The remote technical receipts additionally
cover tokenizer recounts of captured training assistant completions, checkpoint
adapter identity, metric recomputation, frozen runtime/stack and native resets.
Held-out token checks verify recorded counts and limits, without an independent
re-tokenization of held-out traces. Weights remain on L4; no local weight
recount is claimed. The placebo changed 342 of 416 captured assignments.

All new conditions use L4 GPU 1 and the same recorded Python/package/OpenEnv
stack, Linux 6.8.0-146 and glibc 2.39. This is the host-consistent replication
subset. The current recorded driver is 580.178.04; historical driver equality
is not established here. The original E1/E2 kernel was 6.8.0-139 and original
placebo used 6.8.0-146; run-date, possible driver and kernel differences remain cross-seed timing
and reproducibility limitations. Other families' original home-host confounds
are unchanged. Neither fresh seeds nor matched current hosts erase them.

## All checkpoints

E0 solves 69/200 (34.5%, Wilson 95% CI [28.26%, 41.32%]), with 1987.43
mean tokens on successful questions.

| Update | E1 correct | E2 correct | Placebo correct | E1 / E2 / placebo mean success tokens |
|---|---:|---:|---:|---:|
| 100 | 169 | 192 | 149 | 1249.28 / 728.98 / 1710.17 |
| 200 | 198 | 196 | 192 | 1258.52 / 400.78 / 1173.85 |
| 300 | 197 | 192 | 191 | 1772.18 / 138.51 / 1383.96 |

All counts are out of 200. These means describe different successful subsets;
the paired compression estimates use only joint successes.

| Update | E1 Wilson 95% CI | E2 Wilson 95% CI | Placebo Wilson 95% CI |
|---|---:|---:|---:|
| 100 | [78.84%, 88.86%] | [92.31%, 97.96%] | [68.04%, 80.04%] |
| 200 | [96.43%, 99.73%] | [94.97%, 99.22%] | [92.31%, 97.96%] |
| 300 | [95.68%, 99.49%] | [92.31%, 97.96%] | [91.67%, 97.61%] |

| Contrast | Update | Gains / regressions | Joint successes | Median paired token change, 95% CI |
|---|---:|---:|---:|---:|
| E2 minus E1 | 100 | 28 / 5 | 164 | -37.39% [-42.11%, -31.26%] |
| E2 minus E1 | 200 | 1 / 3 | 195 | -66.38% [-67.61%, -65.01%] |
| E2 minus E1 | 300 | 1 / 6 | 191 | -92.01% [-92.40%, -91.38%] |
| E2 minus placebo | 100 | 45 / 2 | 147 | -55.05% [-58.41%, -51.79%] |
| E2 minus placebo | 200 | 8 / 4 | 188 | -64.50% [-65.75%, -62.45%] |
| E2 minus placebo | 300 | 5 / 4 | 187 | -89.32% [-90.16%, -88.79%] |
| Placebo minus E1 | 100 | 21 / 41 | 128 | +38.09% [+28.82%, +54.72%] |
| Placebo minus E1 | 200 | 2 / 8 | 190 | -4.47% [-10.55%, -1.43%] |
| Placebo minus E1 | 300 | 2 / 8 | 189 | -22.41% [-28.62%, -12.38%] |

The primary E2-minus-E1 net success difference is -2.5 percentage points
[-5.0, 0.0], exact McNemar p=0.125. E2-minus-placebo is +0.5 points
[-2.5, +3.5], p=1.0. Nonsignificance does not establish equality. Both primary
compression intervals clear the declared 10% target. Absolute paired median
changes are -1566 tokens [-1709, -1463] versus E1 and -1192 [-1273, -1102]
versus placebo. Bootstrap settings remain 20000 resamples, seed 0.

E1 versus E0 has 129 gains and one regression, demonstrating task learning.
Its successful-pair median token change is -11.98% [-19.14%, +2.94%] on 68
joint successes. Placebo versus E1 has eight regressions (7.10% upper bound),
so its modest final compression does not establish preserved success.
E2's 200-update result is retained, but cannot replace the declared final
checkpoint when success falls from 196 to 192 during the last third.

## Task-grounded behavior

E2's eight final failures are wrong submissions with valid actions; none is
a generation-budget stop. Its six regressions relative to E1 occur at indices
17, 45, 64, 101, 152 and 161 (zero-based). They submit `Malee` for `Emalee`,
`Language` for `Arabic`, a color for the requested language or birth year,
or the labels `Language`/`Color` and `Language`/`Religion` instead of values.
Indices 131 and 165 fail in both E1 and E2. E2's one gain, index 149, resolves
E1's no-action budget stop. These are concrete copying/association errors;
the data do not identify whether shorter reasoning caused them.

Against placebo, E2 gains indices 25, 39, 105, 114 and 134 and loses 17, 64,
101 and 165. Both policies fail indices 45, 131, 152 and 161. In the latter
two cases placebo hits the budget while E2 submits wrong values. Thus E2 has
six new wrong submissions despite only four new incorrect outcomes: changes
in the type of failure also count in the behavior panel.

| Final event | E1 | E2 | Placebo |
|---|---:|---:|---:|
| Wrong submission | 2 | 8 | 4 |
| Voluntary no-tool stop | 0 | 0 | 0 |
| Invalid-action episode | 0 | 0 | 6 |
| Repeated-action episode | 0 | 0 | 32 |
| Budget stop / non-termination | 1 | 0 | 5 |

Off-target bounds use newly introduced paired events, one-sided exact
98.75% intervals (alpha 0.05/4), and the unchanged 5% margin. E2 introduces
six wrong submissions against each comparator: upper bound 6.96%, failing
the bound. It introduces zero voluntary stops, invalid-action episodes or
repeated-action episodes: each upper bound 2.17%, passing. Against placebo,
six invalid-action and 32 repeated-action episodes resolve. Placebo versus E1
fails the invalid-action and repeated-action bounds (6.96% and 22.69%).

The retained native auditor covers every new episode. On 2026-11-09, inspection
of all 75 distinct failed-arm compact traces covered all 82 paired regressions
across the listed contrasts and checkpoints, including E1 versus E0. Expected
and submitted fields, action arguments/errors and stop reasons were visible;
the extractor separately asserts that each paired reference submits both
expected values on the same seed. The saved per-case judgments distinguish:

| Contrast/update | Inspected regression causes |
|---|---|
| E2/E1 100 | Four label-as-value submissions; one correct fill without submission before a no-tool stop. |
| E2/E1 200 | One wrong value; two invalid fills then empty submission (143); field 36 overwritten while 33 stays empty (153). |
| E2/E1 300 | Three label-as-value and three other wrong-value submissions, as detailed above. |
| E2/placebo 100 | One label-as-value and one other wrong-value submission. |
| E2/placebo 200 | Two wrong-value submissions plus the empty/missing-field cases 143 and 153. |
| E2/placebo 300 | Four wrong-value submissions, as detailed above. |
| Placebo/E1 100 | 29 label-as-value submissions, two incomplete submissions after invalid fills, nine budget stops and one no-tool stop. |
| Placebo/E1 200 | Three wrong values, one missing field, two budget stops and two no-tool stops. |
| Placebo/E1 300 | Three wrong submissions and five budget stops; three of these stops include an invalid extra fill after correct fields. |
| E1/E0 300 | One budget stop without an executed action. |

The original seed-17 stratified sampling logic was also applied to this new
pool: six budget cases, all four voluntary stops, six invalid/repeated-action
cases and six efficient correct cases (success-token lower quartile 563.5).
All 22 compact traces were inspected before opening the arm/step/index key.
The six efficient cases show two correct fills and submission without invalid
or repeated actions. All four voluntary cases filled the correct values but
did not submit. The invalid/repeated sample shows later-turn repeated fills
and occasional rejected fills into the submit button, followed by correct
submission. Budget cases include no-action exhaustion and correct fills without
submission. The judgments and revealed sample key are retained.

This is task-grounded inspection, not independent semantic label validation.
Outcomes, token counts and native audit fields were visible; aggregates and
regression cases were already known, and several sampled cases were recognized.
No independent blinded agreement estimate is claimed.
Unknown or unmeasured behavioral categories remain outside coverage; task 8
remains separate. Passing any selected bound would not prove full equivalence.

Transitions run from comparator to treatment. All omitted transition cells are zero.

| Contrast | env_done to env_done | env_done to budget stop | budget stop to env_done | no-tool stop to env_done |
|---|---:|---:|---:|---:|
| E2 minus E1 | 199 | 0 | 1 | 0 |
| E2 minus placebo | 195 | 0 | 5 | 0 |
| Placebo minus E1 | 194 | 5 | 1 | 0 |
| E1 minus E0 | 96 | 1 | 91 | 12 |

## Training activity and costs

| Measure | E1 | E2 | Placebo |
|---|---:|---:|---:|
| Mean active prompt-group fraction | 29.58% | 96.50% | 98.50% |
| Updates with any live group | 57.67% | 100% | 100% |
| Zero logged-gradient updates | 127 | 0 | 0 |
| Tail correct rollouts / 320 | 311 | 318 | 315 |
| Tail task-variable groups / 40 | 9 | 2 | 4 |
| Tail composed-reward-variable groups / 40 | 9 | 31 | 40 |

Both added-reward arms remain much more active than E1, yet E2 compresses
much more. Assignment matters in these observed outcomes beyond a simple
binary account of whether extra groups are active. Placebo does not equate
total reward variance, gradient magnitude, state visitation or optimization
noise, so this does not isolate a unique causal mechanism.

| Arm | Training GPU-hours | Evaluation GPU-hours | Total GPU-hours |
|---|---:|---:|---:|
| E0 | 0 | 3.768 | 3.768 |
| E1 | 28.391 | 11.128 | 39.518 |
| E2 | 23.017 | 3.670 | 26.687 |
| Placebo | 29.719 | 11.015 | 40.734 |

The quartet consumed 110.708 recorded GPU allocation/wall hours, about
4.6 days, with 13 complete cost attempts. Each attempt is counted once;
nested evaluation timers are not added again. These are not energy or
monetary costs. Preflight, local review and other campaign cells are excluded.

E2 saves mean inference time of 67.76 seconds per question [64.65, 70.89]
against E1 and 53.24 [50.25, 56.36] against placebo. On joint successes the
savings are 66.17 [63.21, 69.21] and 51.54 [48.78, 54.40] seconds. These are
descriptive timing differences on the measured hosts/runs, not universal
deployment guarantees; error tradeoffs remain part of the result.

All-question mean inference / total episode times are E0 63.04 / 67.76 seconds,
E1 73.50 / 77.67, E2 5.75 / 10.05 and placebo 58.98 / 63.38. The paired
inference savings and their joint-success sensitivities are reported above.
Full E2 training cost amortizes in about 1223 questions versus E1 or 1556 versus
placebo using all-question mean inference savings. Joint-success sensitivities
are about 1252 and 1608, respectively. E2 training costs 5.374 fewer GPU-hours
than E1 and 6.702 fewer than placebo, so there is no positive incremental
training-cost deficit to recover in either comparison. These descriptive
ratios exclude evaluation costs and do not value errors. All-question figures
include failed and budget-stopped episodes; joint-success ratios are conditional
sensitivities, not predictions of required attempted deployments.
Their timing interpretation remains conditional on hardware, kernel, load and
run date. Against E0, full E2 training amortizes in about 1446 questions using
the measured all-question mean inference saving of 57.30 seconds, with the same
run-conditional and error-valuation limitations.

In these measured evaluation runs, E1 has no finite inference-time break-even
against E0: its
paired mean inference saving is negative, -10.46 seconds [-15.13, -5.74] over
all questions and -23.41 [-29.48, -17.39] over joint successes. Higher success
and a lower point-estimate joint-success token count (median paired change
-11.98% [-19.14%, +2.94%], interval includes zero) do not imply lower measured
latency. The sign is conditional on hardware, serving configuration, load and
run date; these results do not attribute it to specialization. The original
seed-4016 report estimated about 2118 episodes for E1 versus E0; that contrast
does not establish why the timing sign changed.

## Evidence

`pipeline/runs/seed-replication-20261025-ops/table-s4017-analysis-20261031/`
contains `analyze.py`, `analysis.json`, `behavior-review.json`, and execution
validation. Regeneration reproduced both outputs byte-for-byte and all 30
paired-statistic tests passed; actual logs and exit codes are retained.
`counterpart-review-output.txt` and `counterpart-review-receipt.json` preserve
the authentication failure and the explicit reviewer/model settings. The
review prompt and pre-review report are saved alongside them. The substantive
2026-11-09 review and its input hashes are preserved there too. Supplemental
case extraction, seed-17 sampling, inspection judgments, reporting corrections,
validation and re-review evidence are in
`pipeline/runs/seed-replication-20261025-ops/table-s4017-review-20261109/`.
The original numerical analysis was not rerun or changed during these corrections.
The supplemental validator, exact command, actual execution log and exit code
are retained as validate_evidence.py and validation-v2.json/log. The earlier
validation.json lacked this retained execution record and is preserved.
`analyze.py` imports the existing table paired-statistic and
case-extraction helpers without executing the old analysis. New outputs are
exclusive and refuse changed regeneration. Historical findings and raw runs
remain unchanged. The ongoing campaign still requires the remaining seeds,
family comparisons and independent review before task 2 can close.
