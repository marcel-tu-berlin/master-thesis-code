# Tic-tac-toe seed 4230: completed E0 and reviewer correction

E0 completed all 200 held-out episodes with exit code zero. The original
controller stopped during technical review, before E1 or E2 existed. The failed
assertion compared executable strings containing `pipeline/../.venv` and
`.venv`. They name the same file; every other stack field matched.

## E0 results

| Measure | Result |
|---|---:|
| Wins | 8/200 (4.0%; Wilson 95% CI 2.0-7.7%) |
| Losses / draws / unfinished | 15 / 0 / 177 |
| Stop: environment done / generation cap / no tool call | 23 / 174 / 3 |
| Mean assistant tokens, successful episodes | 2,089.25 |
| Allocated GPU-hours | 3.933 |

The state-grounded first-action audit contains 157 eligible decisions: 31
known-occupied clicks across 28 episodes, one missed immediate win among nine
winning opportunities, and 18 unblocked single threats among 36 opportunities.
These are descriptive observations of the base policy. Precommitted later calls
do not enter this decision-boundary panel. The threat label is not proof of
suboptimal expected-win play against the partly random opponent.

This greedy evaluation and the earlier sampled training qualification use
different decoding and questions. Their success percentages are not a matched
comparison. Low E0 success does not change the approved advancement rules.
E1/E2 results and the scientific shaping comparison remain pending.

## Technical review and continuation

`tic-tac-toe-campaign-s4230-ops/review_v2.py` normalizes only lexical executable
path aliases. Regression checks accept the equivalent paths and reject a
different interpreter, changed package version, or changed episode boundary.
The complete E0 review passes, including all 200 native outcomes, seeds, metrics,
cost records, configuration and source integrity. All 12 harvested files were
verified against remote hashes; the original nine data files are unchanged.
The original reviewer, controller, manifest and failure log remain intact.

`continuation-amendment.json` binds the original proof, corrected reviewer,
completed E0 receipt and `controller_v2.py`. This controller admits only the
unstarted E1 and E2 arms, retaining the original preflight and failure stops.
Its checks verify ordering, exclusion of E0 and stopping on process/review
failure. The existing authorization covers these remaining arms. E0 is not
rerun, and no scientific setting or production pipeline source changes.

Project validation passed: 509 tests, seven skips, and 17 setup checks. Detailed
receipts and the exact reviewer diff are in the operations directory. Live
progress is recorded separately in `RUNNING.md`.
