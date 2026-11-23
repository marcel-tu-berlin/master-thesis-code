# Tabs, seed 4246

The final E2 policy solved 181 of 200 held-out tasks, compared with 180 for E1 and 164 for the matched 0.1 placebo (P). On jointly solved questions, the median per-question increase in E2 assistant tokens was **41.9% against E1 and 75.4% against P**. Neither primary contrast meets the compression or gross-success-preservation criterion. The near-identical E1 and E2 success totals conceal 17 gains and 16 losses. The severe final success collapse seen at original seed 4231 did not recur.

## Paired step-300 endpoints

Each contrast uses the same 200 task seeds (4246100000 through 4246100199) and identical initial observations. Net success is a percentage-point difference over all questions. Each token result is the median across jointly solved questions of their individual percentage changes. Successful-response tokens count assistant content, reasoning and tool-call arguments. The 4,096-token trajectory budget also charges tool feedback.

| Arm minus reference | Gains / losses | Joint / 200 | Net success, pp [95% CI] | Gross-loss upper 95%, % | Token change, % [95% CI] |
|---|---:|---:|---:|---:|---:|
| E2 minus E1 | 17 / 16 | 164 | +0.5 [-5.0, +6.0] | 11.90 | +41.89 [+32.40, +51.33] |
| E2 minus P | 29 / 12 | 152 | +8.5 [+2.5, +14.5] | 9.54 | +75.45 [+58.53, +92.75] |
| P minus E1 | 10 / 26 | 154 | -8.0 [-13.5, -2.5] | 17.58 | -15.96 [-23.88, -9.84] |
| E1 minus E0 | 129 / 1 | 51 | +64.0 [+57.0, +70.5] | 2.35 | -51.13 [-59.53, -44.89] |

Preservation requires the one-sided exact 95% upper bound on reference successes lost, divided by all 200 questions, to be at most 5%. Compression requires the upper 95% endpoint on the paired median token change to be below -10%. The small net success advantage over P does not cancel the 12 lost P successes. P versus E1 fails preservation with 26 losses and also misses compression at the upper interval endpoint. E1 versus E0 meets both criteria. The intervals resample paired questions 20,000 times with RNG seed 0 and describe these fitted policies. Checkpoints reuse the questions and are descriptive. The paired exact McNemar p-values for E2 versus E1 and E2 versus P are 1.000 and 0.0115.

## All checkpoints

E0 solved 52/200 with a mean of 1,094 assistant tokens on its own successes. All evaluations used its fixed short/long thresholds of 679.3 and 2,946.25. In the last two columns, each entry is joint successes followed by median paired token change.

| Update | E1 / E2 / P successes | Mean tokens on each policy's successes, E1 / E2 / P | E2 minus E1 | E2 minus P |
|---|---|---|---|---|
| 100 | 161 / 87 / 83 | 616 / 651 / 762 | 78, +5.17% | 56, +21.38% |
| 200 | 179 / 152 / 98 | 1,368 / 838 / 669 | 144, -33.24% | 74, +33.87% |
| **300 (primary)** | **180 / 181 / 164** | **837 / 1,173 / 750** | **164, +41.89%** | **152, +75.45%** |

At step 200 E2 was shorter than E1 but lost 35 E1 successes and gained eight. At step 300 both policies solved more tasks and the token contrast reversed. The prespecified endpoint remains step 300. Original seed 4231 had just five joint successes against each reference, versus 164 and 152 here.

## Off-target behavior and coverage

The four prespecified generic event comparisons count newly affected questions, with a one-sided exact 98.75% upper bound for each of four endpoints. The margin is 5% of all questions.

| Final event | New / resolved vs E1 (upper %) | New / resolved vs P (upper %) |
|---|---:|---:|
| Wrong termination | 6 / 8 (6.96) | 5 / 23 (6.26) |
| Voluntary stop without tool call | 5 / 4 (6.26) | 5 / 2 (6.26) |
| Invalid action | 1 / 1 (3.15) | 1 / 5 (3.15) |
| Repeated action | 5 / 1 (6.26) | 5 / 3 (6.26) |

Only the invalid-action bound passes for both references. At step 300, the native audit attributed all terminal actions. E1, E2 and P chose the wrong visible link on 11/191, 9/190 and 27/191 attributable terminations. The target was initially visible in 63 questions, of which E1/E2/P solved 61/57/61. Six, two and one episodes respectively ended unfinished with the target visible. These exposure denominators differ from incidence over all 200 questions. Early unknown terminal attributions were retained (one E0, one E2-100, two P-100 and one P-200). Observable errors and decision-time opportunities do not establish intent or semantic harm. The full opportunity counts, including zero exposure, are in the analysis file.

## Training, time and verification

| Arm | Training allocation-hours | Evaluation allocation-hours |
|---|---:|---:|
| E0 | 0 | 3.021 |
| E1 | 30.389 | 7.592 |
| E2 | 33.850 | 7.921 |
| P | 35.462 | 8.044 |

At the final checkpoint E2 took 12.97 more inference seconds per question than E1 [10.24, 15.78] and 13.18 more than P [8.98, 17.09]. Training reward groups had mean zero-SD fractions of 0.580, 0.145 and 0.209 for E1, E2 and P. The technical receipt recorded 272 changed shuffle assignments among 416 captured P rollouts. These measurements do not isolate a training mechanism. Allocation-hours measure wall time with an allocated GPU. The four arms used the same L4 and frozen family recipe. System load and run dates can affect timing.

The four technical receipts passed. The local analysis rechecked the 102 harvested file hashes, all 2,000 episode/sample pairs, all ten saved native audits and within-question pairing. It calls the original tabs paired analysis, grounded behavior classifier and native auditor on the new quartet. Detailed checkpoint comparisons, timing intervals, opportunities, costs and source bindings are retained in [analysis.json](seed-replication-20261025-ops/tabs-s4246-analysis-20261122/analysis.json). The [analysis script](seed-replication-20261025-ops/tabs-s4246-analysis-20261122/analyze.py) uses the pinned Beautiful Soup 4.15.0 and Soup Sieve 2.9.1 with the repository test environment. See the [original trio](click_tab_2_s4231_findings.md) and [original placebo comparison](placebo_click_tab_2_s4231_findings.md) for the first seed.
