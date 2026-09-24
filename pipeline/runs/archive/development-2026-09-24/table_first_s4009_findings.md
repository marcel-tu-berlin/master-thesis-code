# First read-table-2 contrast, seed 4009

E0 and E1 reviewed on 2026-09-21; E2 reviewed on 2026-09-23. The experiment and
analysis were declared in `docs/plans/read-table-first-contrast.md` before these
evaluations. The subsequent [reward audit](table_first_s4009_reward_audit_findings.md)
checks application of the length signal and investigates the adverse result.

## Fresh held-out baseline

All observations use the same 200 questions at seeds 4009100000-4009100199,
disjoint from training and the earlier development sample. These are new
instances of read-table-2, not a shifted family. E1 reuses the original pilot's
adapters; it was not retrained. Step 300 remains the primary endpoint.

| Observation | Success | Wilson 95% interval | Mean correct tokens | Wrong submissions | Budget endings |
|---|---:|---:|---:|---:|---:|
| E0 base | 55/200 (27.5%) | 21.78-34.07% | 2002.58 | 39/200 | 91/200 |
| E1 step 100 | 181/200 (90.5%) | 85.64-93.83% | 1331.75 | 11/200 | 8/200 |
| E1 step 200 | 197/200 (98.5%) | 95.68-99.49% | 700.89 | 1/200 | 2/200 |
| E1 step 300 | 196/200 (98.0%) | 94.97-99.22% | 633.30 | 3/200 | 1/200 |

Correct-token means condition on each observation's own successes. Their
difference is not a paired efficiency estimate. None of these changes is an E2
effect. E1's high development accuracy carries over to fresh same-family
questions, with four final failures rather than the development sample's zero.
This does not establish transfer across task structures or training seeds.

All E1 observations have zero invalid-action episodes, repeated-action episodes,
and voluntary no-tool stops. The final three wrong submissions are table-value
selection or transcription mistakes (indices 8, 21 and 118). Index 111 repeats
its analysis until the generation cap without acting. These are model outcomes;
the recorded environment rewards agree with the visible form contents.

The frozen short-response flag marks 95.92% of final successful episodes. It
remains descriptive: valid short solutions are not evidence of harmful
underthinking. Budget endings remain the separate E3 warning panel.

## Technical review

The existing oracle audited all 600 E1 trajectories, including visible table
values, actions, rewards, stops and absence of generation after completion.
Every aggregate metric was recomputed. All checkpoints used identical initial
observations, matching E0; all expected seeds and episode indices were present.
Both evaluation exits succeeded, and the log contains no fatal runtime errors.
The three checkpoint evaluations took 7.31 hours in total.

Remote hashes match all harvested files. Every copied adapter matches the
admitted original pilot checkpoint, including its step. The frozen recipe,
metric reference, source and stack match the admission. The original pilot's
completed training audit is reused, with identical training geometry and reward.
Its historical failure under the old success ceiling is preserved; decision
0018 supplies the current admission.

The reviewer inspected failure tool calls at every checkpoint, the four final
failures in full, and final correct controls 137, 108 and 79 selected with seed
17. Aggregate outcomes were already known; this was not a blinded review. The
declared masked paired review remains due after E2.

Evidence: `table-first-e1-s4009/technical_review.json`, `review.json`, and
`table-first-s4009-ops/e1-review/remote_evidence.json`. The runnable audit is
`table-first-s4009-ops/review_run.py table-first-e1-s4009` from the repository
root with `.venv-test/bin/python`.

E1 passes technical review. The admitted next stage is E2 at cosine weight 0.4,
300 updates from the same original base, followed by all three scheduled
evaluations and the declared paired comparison. No weight, sample size,
checkpoint-selection rule, family or runtime setting changes follow from these
E1 results.

## E2 final review

E2 completed its 300-update run from the declared original base and all three
held-out observations. The frozen oracle re-audited every 600 E2 trajectories:
visible table values, rewards, stops and aggregate metrics all recomputed. The
source and stack match the admission, all harvested main and scheduled-evaluation
files match their remote SHA256 values, and train, eval and controller exits are
zero. The run recorded all 300 finite update rows. Its 13 declared group captures
(1, 100, 200 and 291-300) have four groups of eight records each, finite raw/composed rewards
and within-group cosine variation.

The primary step is 300. E1 solved 196/200 and E2 195/200 fresh same-family
questions; E2 has four E1-to-E2 regressions and three gains. The declared
one-sided exact upper bound is 4.52%, so success preservation is established at
the predeclared 5pp margin. On the 192 jointly correct questions, however, E2
used a median 47.04% more tokens (95% percentile-bootstrap CI 39.80% to 56.38%;
median absolute difference +282 tokens, 95% CI +241.5 to +313). The 10% reduction
target is not supported.

The step-100 sensitivity observation had a -16.88% median change (CI -21.48 to
-8.94) but six regressions, so did not establish preservation or the 10% target.
Step 200 established the success margin with four regressions but had +27.61%
median token change (CI +23.21 to +33.48). These observations are sensitivity
only; none replaces step 300.

At step 300, E2 introduced four wrong submissions and five invalid-action
episodes relative to E1. Their simultaneous one-sided upper bounds are 5.54% and
6.26%, so neither excludes a 5pp increase. It introduced no voluntary stops or
repeated actions (2.17% upper bound each), and removed E1's one budget ending.
The four success regressions (seeds 4009100020, 4009100054, 4009100183 and
4009100196) were audited
against their visible tables; three involved invalid actions. Seed-17 masked
correct controls 167, 023 and 146 were also inspected. This review was not
blinded because aggregate outcomes were already known.

This is a technically valid but scientifically adverse/inconclusive one-seed
first contrast on new read-table-2 instances. It does not estimate training-seed
robustness, other task structures, placebo cost assignment or general safety.
Decision 0018 prohibits a weight, sample or family extension before the user's
result review. Evidence: `table-first-e2-l040-s4009/technical_review.json`,
`review.json`, and the declared `table-first-s4009-ops/analyze.py` output.

Reporting correction, 2026-09-23: the original hash-bound `review.json` reason
says four introduced invalid-action episodes. The data and analysis show five,
as stated above; the 6.26% bound is correct. Its receipt is preserved unchanged.
The capture-count prose above was corrected from eight records per batch to
32 (four groups of eight). Neither correction changes the measurements.
