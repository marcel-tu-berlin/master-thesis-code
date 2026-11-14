# Checkbox seed 4242: compression with fewer exact completions

Prepared 2026-11-09 from the completed, hash-verified second checkbox seed.
Independent scientific counterpart review passed after reporting corrections on
2026-11-09 with Claude Opus 5.5/xhigh. Update 300 is primary.
Review evidence and exact final report binding are retained in the analysis
directory as `counterpart-corrections-output.json` and `report-finalization.json`.

## Result

E2 succeeds on 184/200 questions under the native score, versus E1's 193/200
and matched placebo's 186/200. Exact completion is lower: 96/200 for E2,
134/200 for E1 and 142/200 for placebo. E2 passes the declared compression
criterion against both references, including among jointly exact completions,
but fails the success-preservation criterion. Similar native totals do not
establish preservation: gains cannot cancel the gross regressions in that rule.

Native success is the primary endpoint. The frozen task's majority-based
checkbox score can accept missing required selections or extra irrelevant
selections. Exact completion requires the full correct set and an executed
Submit; it is a separate diagnostic, not a replacement training reward.

## Fixed final comparisons

All rows pair the same 200 questions. Negative token differences mean shorter
responses in the named arm. Token estimates use only the stated joint-success
intersection; they exclude failures and cannot compensate for lost completion.

### Native endpoint

| Contrast | Gains / losses | Joint / 200 | Median token change %, 95% CI | Median absolute tokens, 95% CI | Net success pp, 95% CI | Gross-loss upper 95% | Exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2-v-E1 | 7 / 16 | 177 | -39.73 [-46.40, -32.65] | -351.00 [-401.00, -294.00] | -4.50 [-9.00, 0.00] | 11.90% | 0.09314 |
| E2-v-P | 12 / 14 | 172 | -25.54 [-32.79, -15.19] | -166.50 [-270.50, -76.00] | -1.00 [-6.00, 4.00] | 10.73% | 0.84502 |
| P-v-E1 | 7 / 14 | 179 | -22.44 [-28.25, -14.02] | -173.00 [-251.00, -128.00] | -3.50 [-8.00, 1.00] | 10.73% | 0.18925 |
| E1-v-E0 | 70 / 4 | 123 | -15.25 [-21.28, -6.17] | -197.00 [-256.00, -72.00] | 33.00 [26.00, 40.00] | 4.52% | 1.2898e-16 |

### Exact endpoint

| Contrast | Gains / losses | Joint / 200 | Median token change %, 95% CI | Median absolute tokens, 95% CI | Net success pp, 95% CI | Gross-loss upper 95% | Exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2-v-E1 | 19 / 57 | 77 | -52.66 [-57.74, -45.37] | -401.00 [-470.00, -348.00] | -19.00 [-27.00, -11.00] | 34.22% | 1.4837e-05 |
| E2-v-P | 14 / 60 | 82 | -29.07 [-44.40, -19.12] | -165.50 [-271.00, -76.00] | -23.00 [-31.00, -15.00] | 35.78% | 6.2218e-08 |
| P-v-E1 | 34 / 26 | 108 | -27.89 [-35.78, -20.17] | -236.00 [-321.00, -162.00] | 4.00 [-3.50, 12.00] | 17.58% | 0.36629 |
| E1-v-E0 | 97 / 10 | 37 | 0.41 [-21.39, 8.43] | 6.00 [-224.00, 49.00] | 43.50 [35.00, 51.50] | 8.33% | 4.8166e-19 |

Preservation requires the one-sided exact 95% upper bound on reference-success /
arm-failure probability to be at most 5%. Compression requires the upper 95%
endpoint of the median per-question percentage token change to be below -10%.
Both E2 contrasts pass compression and fail preservation. Their net native
success intervals include zero; this is not evidence of equivalent policies.
E2/E1's gross-regression lower bound is 5.08%, barely above 5%; E2/placebo's
is 4.28%, so only the former supports gross harm beyond the margin at that
one-sided level. Neither establishes a net native loss beyond five points.
The exact-success deficits are much larger and their paired intervals exclude zero.

Placebo/E1 also compresses while failing native preservation. Its exact net
gain is unresolved. E1/E0 improves native and exact success, but its native
compression interval does not meet the 10% target; the exact token interval
includes zero. Applying the same preservation diagnostic, E1/E0 passes for
native success and fails for exact success despite its large net improvement.

Bootstrap intervals use 20000 paired question resamples, RNG seed 0, through
the retained family helper. They estimate held-out question uncertainty for
these fitted policies, not training-seed uncertainty. Marginal Wilson intervals
below are not intervals for a paired difference. Secondary endpoints and
checkpoint intervals are descriptive, with no simultaneous coverage across
contrasts, endpoints or the campaign. The four-event adjustment below applies
only within that event panel. These are exploratory replications declared after
the original family outcomes.

| Final policy | Native / 200, Wilson 95% | Exact / 200, Wilson 95% | Native-success non-exact |
| --- | --- | --- | --- |
| E0 | 127 [56.63%, 69.86%] | 47 [18.16%, 29.84%] | 80 |
| E1 | 193 [92.95%, 98.29%] | 134 [60.22%, 73.14%] | 59 |
| P | 186 [88.59%, 95.78%] | 142 [64.36%, 76.85%] | 44 |
| E2 | 184 [87.40%, 95.02%] | 96 [41.18%, 54.90%] | 88 |

| Final contrast | Native shorter / equal / longer | Exact shorter / equal / longer |
| --- | --- | --- |
| E2-v-E1 | 140 / 0 / 37 | 72 / 0 / 5 |
| E2-v-P | 112 / 0 / 60 | 59 / 0 / 23 |
| P-v-E1 | 115 / 0 / 64 | 79 / 0 / 29 |
| E1-v-E0 | 78 / 0 / 45 | 18 / 0 / 19 |

## Design and integrity

Protocol: `docs/plans/seed-replication-20261025.md` and decision 0029.
The four run IDs are `e0-click-checkboxes-soft-s4242-c4096`,
`e1-click-checkboxes-soft-s4242-c4096`, `e2-click-checkboxes-soft-s4242-c4096`,
and `e2placebo-l010-click-checkboxes-soft-s4242-c4096` (P).
All held-out seeds are 4242100000 through 4242100199, with identical initial
observations across all ten evaluations. Each trained arm starts independently
from pinned Qwen3-1.7B with fresh LoRA, 300 updates, batch four, eight rollouts,
naive_sum and scale_rewards none. Dose is 0.1 for E2 and placebo. Budgets remain
4096 trajectory tokens including feedback, 4096 prompt, 8192 context and eight
turns, with the accepted native batch/continuation semantics rather than a new
hard per-action transcript cutoff. The fresh E0 fixes thresholds at 592.8 and
1684.75 tokens. Assistant tokens count content, reasoning and serialized
arguments, excluding feedback; the trajectory budget also charges feedback.

Frozen E2/E1 configs differ only in identifiers/descriptions and enabling the
length reward. E2/placebo differ only in identifiers/descriptions and shuffled
assignment. All four receipts pass and bind the declared deployment manifest.
Local checks verify every harvested non-weight file, all 2000 evaluation
records and report samples, all 900 optimizer records, 1248 captured rollouts,
matched capture identities, reward replay and all 13 cost attempts. The retained
native auditor reproduces every saved audit. Its synthetic synonym, exact,
partial, extra, empty-goal and decision-boundary checks pass; deliberately
corrupting a success label is rejected in each of the ten evaluations.

`validate.py` regenerates only this new seed's analysis byte-for-byte, verifies
113 input hashes, and reconstructs the supplementary case selections. No old
completed analysis was rerun. The initial local analysis attempt failed before
analysis because BeautifulSoup was absent. The project-pinned beautifulsoup4
4.15.0 and soupsieve 2.9.1 were installed in an isolated temporary folder;
`.venv-test`, the remote runtime and experiment artifacts were not modified.
The failed execution and source snapshot remain as evidence.

Remote receipts cover source/config/stack integrity, checkpoint presence and
final/update-300 adapter identity, metric recomputation and tokenizer recounts
of the 416 captured training completions per arm. Held-out token counts are
checked against recorded turn sums and limits, not independently re-tokenized.
Weights remain remote and were not downloaded for this analysis. Captures are
selected training updates and can omit oversized final feedback through the
pinned rollback behavior; this is not exhaustive training-trajectory validation.

All four new arms ran on L4 physical GPU 1, driver 580.178.04, kernel
6.8.0-146-generic, glibc 2.39 and the same pinned Python/package/OpenEnv stack.
Original seed 4232 E0/E1 ran on L4 GPU 0; E2 and placebo ran on the home
RTX 4090, kernel 7.2.5-3-omarchy, glibc 2.44, driver 610.57.04. Those original
hardware/OS/driver confounds affect outcome and timing interpretation. This new
host-consistent quartet does not erase them. Neither excluded host nor GPU 0
was contacted for this work.

## All declared checkpoints

Counts are out of 200. Successful-only token means use each policy's own
successful subset, so the paired tables remain the compression evidence.

| Policy | Native (Wilson 95%) | Exact (Wilson 95%) | Non-exact native | Wrong submit / unfinished | Mean successful tokens |
| --- | --- | --- | --- | --- | --- |
| E0-0 | 127 [56.63, 69.86] | 47 [18.16, 29.84] | 80 | 59 / 14 | 1271.41 |
| E1-100 | 184 [87.40, 95.02] | 127 [56.63, 69.86] | 57 | 15 / 1 | 1119.07 |
| P-100 | 177 [83.34, 92.21] | 144 [65.41, 77.76] | 33 | 1 / 22 | 2076.94 |
| E2-100 | 193 [92.95, 98.29] | 134 [60.22, 73.14] | 59 | 7 / 0 | 810.84 |
| E1-200 | 188 [89.81, 96.53] | 129 [57.65, 70.80] | 59 | 8 / 4 | 1257.32 |
| P-200 | 177 [83.34, 92.21] | 132 [59.19, 72.21] | 45 | 4 / 19 | 1604.22 |
| E2-200 | 172 [80.51, 90.13] | 84 [35.37, 48.93] | 88 | 18 / 10 | 303.34 |
| E1-300 | 193 [92.95, 98.29] | 134 [60.22, 73.14] | 59 | 6 / 1 | 1114.55 |
| P-300 | 186 [88.59, 95.78] | 142 [64.36, 76.85] | 44 | 13 / 1 | 1003.20 |
| E2-300 | 184 [87.40, 95.02] | 96 [41.18, 54.90] | 88 | 15 / 1 | 803.92 |

| Contrast | Update | Endpoint | Gains / losses | Joint | Median tokens %, 95% CI | Net success pp, 95% CI | Gross-loss upper95% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2-v-E1 | 100 | native | 15 / 6 | 178 | -28.04 [-36.79, -24.01] | 4.50 [0.00, 9.00] | 5.83 |
| E2-v-E1 | 100 | exact | 41 / 34 | 93 | -31.44 [-39.98, -19.41] | 3.50 [-5.00, 12.00] | 21.98 |
| E2-v-E1 | 200 | native | 9 / 25 | 163 | -75.21 [-77.21, -72.06] | -8.00 [-13.50, -2.50] | 17.02 |
| E2-v-E1 | 200 | exact | 15 / 60 | 69 | -71.73 [-76.91, -67.67] | -22.50 [-30.50, -14.50] | 35.78 |
| E2-v-P | 100 | native | 22 / 6 | 171 | -63.58 [-66.83, -60.15] | 8.00 [3.00, 13.00] | 5.83 |
| E2-v-P | 100 | exact | 28 / 38 | 106 | -64.52 [-67.79, -59.79] | -5.00 [-13.00, 3.00] | 24.15 |
| E2-v-P | 200 | native | 18 / 23 | 154 | -79.61 [-81.64, -77.60] | -2.50 [-8.50, 3.50] | 15.90 |
| E2-v-P | 200 | exact | 23 / 71 | 61 | -75.42 [-79.89, -69.50] | -24.00 [-33.00, -15.00] | 41.45 |
| P-v-E1 | 100 | native | 11 / 18 | 166 | 79.05 [60.25, 99.20] | -3.50 [-8.50, 1.51] | 13.05 |
| P-v-E1 | 100 | exact | 43 / 26 | 101 | 78.89 [59.83, 103.52] | 8.50 [0.50, 16.50] | 17.58 |
| P-v-E1 | 200 | native | 8 / 19 | 169 | 23.64 [14.52, 34.52] | -5.50 [-10.50, -0.50] | 13.63 |
| P-v-E1 | 200 | exact | 38 / 35 | 94 | 15.75 [4.99, 31.66] | 1.50 [-7.00, 10.00] | 22.52 |

E2 falls from 193 native / 134 exact successes at update 100 to 172 / 84 at
200, then recovers partly to 184 / 96 at 300 while its successful-response mean
increases. Placebo has 22 and 19 budget stops at 100 and 200, then one at 300.
No earlier checkpoint replaces the declared final endpoint. All checkpoints
belong to the same three training runs, not additional seeds.

These descriptive length flags use the same fixed E0 ruler for every policy.
Their denominators are native successes, including non-exact successes. They
are not evidence of harmful underthinking or overthinking. Counts are checked
against saved report rates by `report_supplement.py`.

| Policy | Native-success denominator | Short <=592.8 tokens | Long >1684.75 tokens |
| --- | --- | --- | --- |
| E0-0 | 127 | 14/127 (11.02%) | 23/127 (18.11%) |
| E1-100 | 184 | 18/184 (9.78%) | 24/184 (13.04%) |
| P-100 | 177 | 1/177 (0.56%) | 112/177 (63.28%) |
| E2-100 | 193 | 61/193 (31.61%) | 8/193 (4.15%) |
| E1-200 | 188 | 24/188 (12.77%) | 39/188 (20.74%) |
| P-200 | 177 | 17/177 (9.60%) | 67/177 (37.85%) |
| E2-200 | 172 | 165/172 (95.93%) | 0/172 (0.00%) |
| E1-300 | 193 | 20/193 (10.36%) | 18/193 (9.33%) |
| P-300 | 186 | 57/186 (30.65%) | 22/186 (11.83%) |
| E2-300 | 184 | 89/184 (48.37%) | 17/184 (9.24%) |

## Task-grounded behavior

Non-exact native success includes omissions, extras, or both. Among final native
successes the non-exact fractions are E1 59/193, P 44/186 and E2 88/184.
These conditional fractions differ from incidence out of all 200 questions.

| Final policy | Omissions only | Extras only | Both |
| --- | --- | --- | --- |
| E1 | 30 | 24 | 5 |
| P | 35 | 2 | 7 |
| E2 | 86 | 0 | 2 |

| Exposure / event | E1 | P | E2 |
| --- | --- | --- | --- |
| Submitted episodes | 199 | 199 | 199 |
| Episodes with omitted required boxes | 39 | 55 | 103 |
| Missing required boxes | 42 | 77 | 149 |
| Required-box opportunities in submitted episodes | 609 | 606 | 605 |
| Episodes with extra selections | 35 | 15 | 2 |
| Extra selected boxes | 45 | 16 | 2 |
| Irrelevant-box opportunities in submitted episodes | 426 | 430 | 430 |
| Complete decision states | 5 | 0 | 0 |
| Incomplete decision states | 200 | 200 | 200 |
| Precommitted Submit from complete state | 1 | 0 | 0 |
| Precommitted Submit from incomplete state | 195 | 199 | 191 |
| Unknown terminal actions | 0 | 0 | 0 |
| Direct complete Submit | 3 | 0 | 0 |
| Direct incomplete Submit | 0 | 0 | 8 |

Required/irrelevant opportunities are checkbox counts within submitted episodes;
episode counts and decision-state counts have different denominators and correlated
observations. They are not independent binomial samples. All 200 episodes have
known task-grounded audit outcomes, including unfinished cases; unknown terminal
actions are zero here. This does not imply coverage of every semantic behavior.

A direct Submit is the first executed action in its decision batch. A later
Submit follows precommitted actions whose feedback was unavailable when that
batch was chosen. Final E2's eight direct incomplete submissions occur at
indices 4, 47, 48, 90, 144, 159, 168 and 191; four receive native success despite
no selected boxes. For example, index 48 submits before a proposed selection of
loathe, and index 90 submits before four proposed selections. Proposed actions
after termination do not execute and must not be counted as state changes.
At index 0, E2 selects camouflage and submits, leaving four required items;
E1 and placebo complete the exact set. No intent or conscious exploitation is
established by these traces.

The inspection ledger attests to reading all 126 distinct failed-arm state/action
summaries covering final native and exact regressions across the four contrasts,
plus both sides of six seed-17 joint-exact controls per contrast (24 pair controls).
No per-case judgment notes were retained for those 126 cases or the controls;
their selection and packet content are reproducible, but reading is an attestation.
The machine-audited reference states and complete compact pairs remain in
`final-case-packet.json`; this inspection claim concerns displayed state/action
summaries, not a fresh manual reading of every full reasoning trace.
The controls include real exact compression, such as E2/E1 index 74 with
1926 to 425 tokens on the same correct three-word selection; E2 is longer
than placebo on control 106 (707 versus 538), so compression is not universal.

| Final contrast | Native regression causes | Exact regression causes |
| --- | --- | --- |
| E2-v-E1 | {'no_submit_hit_generation_cap': 1, 'omissions_only': 15} | {'both': 2, 'omissions_only': 55} |
| E2-v-P | {'omissions_only': 14} | {'both': 2, 'omissions_only': 58} |
| P-v-E1 | {'both': 6, 'no_submit_hit_generation_cap': 1, 'omissions_only': 7} | {'both': 7, 'extra_only': 1, 'no_submit_hit_generation_cap': 1, 'omissions_only': 17} |
| E1-v-E0 | {'both': 3, 'extra_only': 1} | {'both': 1, 'extra_only': 5, 'omissions_only': 4} |

| Final contrast / incidence | New / resolved | Net pp, paired95% |
| --- | --- | --- |
| E2-v-E1 / direct_incomplete | 8 / 0 | 4.00 [1.50, 7.00] |
| E2-v-E1 / native_success_incomplete | 49 / 20 | 14.50 [6.50, 22.50] |
| E2-v-P / direct_incomplete | 8 / 0 | 4.00 [1.50, 7.00] |
| E2-v-P / native_success_incomplete | 59 / 15 | 22.00 [14.00, 30.00] |
| P-v-E1 / direct_incomplete | 0 / 0 | 0.00 [0.00, 0.00] |
| P-v-E1 / native_success_incomplete | 23 / 38 | -7.50 [-15.00, 0.00] |
| E1-v-E0 / direct_incomplete | 0 / 25 | -12.50 [-17.50, -8.00] |
| E1-v-E0 / native_success_incomplete | 38 / 59 | -10.50 [-20.00, -1.00] |

The supplementary masked review used all ten evaluations with the retained
stratum precedence: budget, voluntary stop, invalid/repeated, efficient native
success. Seed 17 selected six of each (24 total), with pooled successful-token
lower quartile 572. Available counts were 67, 6, 37 and 443. Goals, initial states,
ordered calls and feedback were reportedly read before the key; this ordering
is an attestation, not independently verified. The generator wrote the key
alongside the cases before inspection. Outcomes and aggregates were already
visible, and model reasoning text was not inspected, so this is not independent
blind semantic validation. Per-case notes and the key are retained separately.
Two sampled efficient
native successes omitted a required word; other sampled cases include invalid
HTML-id calls, repeated label toggles and exact selections that never Submit.
No unseen behavior prevalence is inferred from this stratified sample.

Final trained policies have zero recorded invalid-action or adjacent verbatim
repeat events, despite omissions and toggle errors. The repeat detector counts
only a dispatched call with the same name and arguments as the immediately
preceding dispatched call. It misses both non-adjacent same-bid repeats and
label/input aliases that toggle the same checkbox. Masked case 24 (E2-200,
index 133) repeats bids 29, 32 and 26 non-adjacently across turns, unticking boxes,
while its recorded repeat count is zero. All repeat counts and bounds below
refer only to adjacent verbatim repeats. Broad claims about feedback neglect, intent or reasoning quality
remain outside these deterministic indicators. Independent label validation
(task 8) remains separate.

## Stop reasons and event bounds

| Policy | env_done | token cap | no_tool_call | max_turns | Invalid episodes | Adjacent-repeat episodes |
| --- | --- | --- | --- | --- | --- | --- |
| E0-0 | 186 | 13 | 1 | 0 | 40 | 3 |
| E1-100 | 199 | 1 | 0 | 0 | 1 | 1 |
| E1-200 | 196 | 4 | 0 | 0 | 0 | 1 |
| E1-300 | 199 | 1 | 0 | 0 | 0 | 0 |
| E2-100 | 200 | 0 | 0 | 0 | 1 | 2 |
| E2-200 | 190 | 5 | 5 | 0 | 1 | 0 |
| E2-300 | 199 | 1 | 0 | 0 | 0 | 0 |
| P-100 | 178 | 22 | 0 | 0 | 1 | 0 |
| P-200 | 181 | 19 | 0 | 0 | 0 | 0 |
| P-300 | 199 | 1 | 0 | 0 | 0 | 0 |

The following table is reproduced by the retained `report_supplement.py`,
using `eval.paired.stop_reason_transitions` on seed-keyed records from
`behavior-review.json`; `report-supplement.json` binds sources and counts.

| Reference -> arm stop | E2-v-E1 | E2-v-P | P-v-E1 | E1-v-E0 |
| --- | --- | --- | --- | --- |
| env_done -> env_done | 198 | 198 | 198 | 186 |
| env_done -> hit_generation_cap | 1 | 1 | 1 | 0 |
| hit_generation_cap -> env_done | 1 | 1 | 1 | 12 |
| hit_generation_cap -> hit_generation_cap | 0 | 0 | 0 | 1 |
| no_tool_call -> env_done | 0 | 0 | 0 | 1 |

Final E1, E2 and placebo each have one non-terminating episode, all token-cap
stops. Equal totals conceal different questions and do not establish paired
preservation. The following panel counts an event newly present in the arm and
absent in its paired reference; bounds are one-sided exact 98.75% (alpha .05/4),
with the inherited 5% margin. Cap/turn stops are reported separately, not silently
folded into voluntary no-tool stopping.

| Contrast | Event | Reference / arm episodes | New / resolved | Upper98.75% |
| --- | --- | --- | --- | --- |
| E2-v-E1 | invalid_action | 0 / 0 | 0 / 0 | 2.17 |
| E2-v-E1 | adjacent_verbatim_repeat | 0 / 0 | 0 / 0 | 2.17 |
| E2-v-E1 | voluntary_no_tool | 0 / 0 | 0 / 0 | 2.17 |
| E2-v-E1 | wrong_submission | 6 / 15 | 15 / 6 | 12.77 |
| E2-v-P | invalid_action | 0 / 0 | 0 / 0 | 2.17 |
| E2-v-P | adjacent_verbatim_repeat | 0 / 0 | 0 / 0 | 2.17 |
| E2-v-P | voluntary_no_tool | 0 / 0 | 0 / 0 | 2.17 |
| E2-v-P | wrong_submission | 13 / 15 | 14 / 12 | 12.15 |
| P-v-E1 | invalid_action | 0 / 0 | 0 / 0 | 2.17 |
| P-v-E1 | adjacent_verbatim_repeat | 0 / 0 | 0 / 0 | 2.17 |
| P-v-E1 | voluntary_no_tool | 0 / 0 | 0 / 0 | 2.17 |
| P-v-E1 | wrong_submission | 6 / 13 | 13 / 6 | 11.53 |
| E1-v-E0 | invalid_action | 40 / 0 | 0 / 40 | 2.17 |
| E1-v-E0 | adjacent_verbatim_repeat | 3 / 0 | 0 / 3 | 2.17 |
| E1-v-E0 | voluntary_no_tool | 1 / 0 | 0 / 1 | 2.17 |
| E1-v-E0 | wrong_submission | 59 / 6 | 5 / 58 | 6.26 |

Both primary panels fail because their new-wrong-submission upper bounds
exceed the 5% margin. Zero new no-tool, invalid and adjacent-repeat events produce a
2.17% bound for each; those individual bounds passing do not establish absence of
omissions, premature Submit or all off-target behavior. The same wrong-event
bound fails for placebo/E1 and, despite strong net improvement, E1/E0.

## Training activity and costs

| Arm | Mean live-group fraction | Updates with live signal | Zero-gradient updates | Tail successes / 320 | Tail variable task / composed groups, of40 | Changed placebo assignments /416 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 30.42% | 211/300 | 89 | 308 | 9 / 9 | 0 |
| E2 | 99.75% | 300/300 | 0 | 304 | 5 / 39 | 0 |
| P | 99.92% | 300/300 | 0 | 310 | 9 / 40 | 351 |

The last-ten-update capture is diagnostic, not a new evaluation set. Both E2
and placebo keep almost every group active, while E1 has many constant-reward
groups. Extra gradient activity alone does not reproduce E2's exact-success
outcome at this seed; it does not identify a unique causal mechanism.

Costs below sum each complete attempt exactly once on one allocated GPU.
Evaluation hours include all three checkpoint evaluations per trained arm.
They are measured allocated hours, not normalized device throughput or an invoice.

| Arm | Training GPU-h | Evaluation GPU-h | Total GPU-h |
| --- | --- | --- | --- |
| E0 | 0.0000 | 2.0327 | 2.0327 |
| E1 | 26.6271 | 9.1937 | 35.8208 |
| E2 | 22.0425 | 5.4764 | 27.5189 |
| P | 31.9888 | 12.8804 | 44.8693 |

New quartet total: 110.2417 allocated GPU-hours. Historical original-run costs remain in their reports, including the discarded L4 E2 attempt's 3.3742 hours; they are not charged again to this quartet.

| Final policy | Mean inference seconds / question | Mean total episode seconds / question |
| --- | --- | --- |
| E0 | 31.352 | 36.529 |
| E1 | 48.124 | 53.423 |
| P | 41.947 | 47.098 |
| E2 | 35.252 | 39.953 |

| Contrast | Population | Inference seconds saved, 95% CI | Total seconds saved, 95% CI |
| --- | --- | --- | --- |
| E2-v-E1 | All 200 | 12.87 [8.22, 17.68] | 13.47 [8.84, 18.26] |
| E2-v-E1 | Joint native | 12.04 [7.84, 16.13] | 12.40 [8.20, 16.49] |
| E2-v-E1 | Joint exact | 18.71 [14.96, 22.52] | 18.52 [14.77, 22.33] |
| E2-v-P | All 200 | 6.69 [2.56, 10.74] | 7.15 [3.04, 11.16] |
| E2-v-P | Joint native | 8.60 [4.32, 12.87] | 8.92 [4.65, 13.18] |
| E2-v-P | Joint exact | 11.10 [6.63, 15.59] | 11.09 [6.61, 15.58] |
| P-v-E1 | All 200 | 6.18 [2.02, 10.33] | 6.32 [2.17, 10.48] |
| P-v-E1 | Joint native | 4.82 [1.16, 8.41] | 4.87 [1.20, 8.48] |
| P-v-E1 | Joint exact | 7.21 [3.16, 11.12] | 7.04 [2.99, 10.96] |
| E1-v-E0 | All 200 | -16.77 [-20.13, -13.49] | -16.89 [-20.35, -13.55] |
| E1-v-E0 | Joint native | -15.37 [-18.70, -12.29] | -15.16 [-18.48, -12.05] |
| E1-v-E0 | Joint exact | -13.73 [-17.92, -9.63] | -13.71 [-17.90, -9.63] |

Timing is reference minus named arm; positive means time saved. All-question
savings include changed failures; joint-success savings condition on survivorship.
Neither substitutes for success-preserving deployment. E1 is slower than E0
here despite its shorter joint-native median token count: measured latency is
not determined by that token statistic alone, and base versus adapter execution
also differs. These sequential runs are not a randomized systems benchmark.

Illustrative amortization divides full arm training hours, or positive incremental
training hours versus the reference, by the measured mean time saved. Evaluation
costs are reported above but excluded from this training-only calculation. No
positive saving means no finite break-even. A zero incremental figure means the
observed arm training was already cheaper, not free training. Point estimates
are not uncertainty intervals or a quality-preserving recommendation.

| Contrast | Extra training h | Timing basis | Full training / incremental episodes |
| --- | --- | --- | --- |
| E2-v-E1 | -4.585 | inference_wall_seconds_saved_ci | 6165 / 0 |
| E2-v-E1 | -4.585 | episode_wall_seconds_saved_ci | 5891 / 0 |
| E2-v-E1 | -4.585 | joint_inference_wall_seconds_saved_ci | 6593 / 0 |
| E2-v-E1 | -4.585 | joint_episode_wall_seconds_saved_ci | 6401 / 0 |
| E2-v-P | -9.946 | inference_wall_seconds_saved_ci | 11853 / 0 |
| E2-v-P | -9.946 | episode_wall_seconds_saved_ci | 11106 / 0 |
| E2-v-P | -9.946 | joint_inference_wall_seconds_saved_ci | 9223 / 0 |
| E2-v-P | -9.946 | joint_episode_wall_seconds_saved_ci | 8894 / 0 |
| P-v-E1 | 5.362 | inference_wall_seconds_saved_ci | 18642 / 3125 |
| P-v-E1 | 5.362 | episode_wall_seconds_saved_ci | 18209 / 3052 |
| P-v-E1 | 5.362 | joint_inference_wall_seconds_saved_ci | 23886 / 4004 |
| P-v-E1 | 5.362 | joint_episode_wall_seconds_saved_ci | 23654 / 3965 |
| E1-v-E0 | 26.627 | inference_wall_seconds_saved_ci | none |
| E1-v-E0 | 26.627 | episode_wall_seconds_saved_ci | none |
| E1-v-E0 | 26.627 | joint_inference_wall_seconds_saved_ci | none |
| E1-v-E0 | 26.627 | joint_episode_wall_seconds_saved_ci | none |

The joint amortization rows use native joint success. E2's observed training
is 4.58 hours cheaper than E1 and 9.95 hours cheaper than placebo, so incremental
training amortization is zero in those two measured comparisons. Full E2 training
still costs 22.04 hours. E1/E0 has no finite measured latency break-even under
these means. Host-consistent new timings do not resolve the original L4/home
cross-host break-even limitation.

## Partial evidence across seeds

The original reviewed reports are reused, not recomputed. Seed 4232 had E1/P/E2
native counts 191/183/162 and exact counts 139/156/62. Seed 4242 has
193/186/184 native and 134/142/96 exact. In both seeds, E2 compresses versus
both E1 and placebo and fails native preservation, while exact completion is
lower than both references. E2/E1 median native compression is -43.53% in the
original and -39.73% here; E2/placebo is -81.01% versus -25.54%. The direction
repeats, but placebo-relative magnitude and native loss vary substantially.
The original E2/placebo native deficit was -10.5 pp; here it is -1.0 pp with an
interval spanning zero. The original exact deficit was -47 pp; here it is -23 pp.

The P/E1 contrast reverses: joint-native responses lengthen by 171.25% in the
original seed but shorten by 22.44% here. Both the reference and placebo moved:
own-success mean tokens change from 622.81 to 1114.55 for E1, 1695.61 to 1003.20
for P, and 319.41 to 803.92 for E2. Similar relative E2/E1 compression coexists
with large absolute length changes in both arms. These means use different
successful question subsets. Questions and hardware also differ across seeds,
and the original P/E1 comparison crosses L4 and home hardware; two seeds cannot
isolate placebo-specific instability.
The original partly confounded E2/E1 comparison therefore cannot supply a
universal mechanism or cost claim. The new L4-only seed is one host-consistent
replication; seed 4243 is still queued. Seed changes affect both training
randomness and generated questions. No equal-weight three-seed summary or
training-seed interval is available yet, and no pooled cross-family conclusion
is drawn. Earlier table and inbox analyses are unchanged.

## Evidence and remaining work

New artifacts are under
`pipeline/runs/seed-replication-20261025-ops/checkbox-s4242-analysis-20261109/`:
`analyze.py`, `analysis.json`, `behavior-review.json`, `final-case-packet.json`,
`prepare_masked_review.py`, the masked cases/key/selection/inspection notes,
`validate.py`, `report_supplement.py`, `report-supplement.json`,
`inspection-ledger.json`, execution and validation logs/receipts. The helper imports reuse
`placebo-l010-20261018-ops/checkbox-analysis-20261019/analyze_v2.py` and
`review_cases_v2.py`; their old mains were not executed. All interpretation
limits remain attached to the results. This seed group does not complete the
40-cell campaign or thesis task 2.

Reproduce only this analysis with `.venv-test/bin/python` and
`PYTHONPATH=/private/tmp/checkbox-s4242-analysis-deps`, after placing the two
project-pinned HTML-parser packages in that isolated directory. Run the new
`validate.py` to check deterministic regeneration, hashes and case selections.
The retained execution receipts record the exact commands and dependencies.
