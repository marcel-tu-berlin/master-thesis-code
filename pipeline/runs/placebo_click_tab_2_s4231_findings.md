# Click-tab-2 seed 4231: matched 0.1 placebo

The placebo completed at 2026-10-24 01:40 UTC (03:40 Europe/Berlin).
All 29 non-weight artifacts, including checkpoint evaluations, are harvested
and hash-verified. Update 300 is primary. This report adds the two authorized
placebo contrasts; the completed original E0/E1/E2 analysis is unchanged.

## Result

The shuffled-cost placebo solves 124/200 questions, between E1's 189/200 and
E2's 5/200. E2 loses 119 placebo successes and gains none: -59.5 percentage
points, paired bootstrap 95% interval [-66.0, -52.5], exact McNemar
p=3.01e-36. Its gross-regression probability has a one-sided 95% interval
bound of 65.32% above and 53.46% below, far beyond the declared 5% margin.

Only five questions are jointly correct. E2's median per-question token
change is +85.09% [-55.53%, +478.09%], with one shorter and four longer.
The 10% compression target is not established. This tiny, survivor-selected
subset cannot rescue an efficiency claim after the success collapse.

Placebo itself substantially underperforms task-only E1: seven gains and
72 losses, -32.5 points [-40.0, -25.0], McNemar p=1.06e-14. Its gross-loss
upper bound is 41.96% and lower bound 30.35%, also beyond 5%. On 117 joint
successes, median token change is -7.00% [-32.51%, +15.04%]; 62 shorten and
55 lengthen. Neither preservation nor 10% compression is established.

The extreme final E2 collapse is absent under shuffled assignment, while
shuffled training still performs much worse than task-only training in this
seed. The placebo is therefore not an inert control. These outcomes distinguish
the fitted policies but do not isolate total reward variance, gradient noise,
or a general causal mechanism. This is an exploratory comparison made after
the original outcomes were known, with one training seed per family.
At update 100 both cost-bearing arms already trail E1 (E2 73, P 76, versus
E1 148 successes); E2 minus P is -1.5 points [-9.0, +6.0], McNemar p=0.80.
The early deficit is shared, with the E2-specific collapse appearing later.

## Design and verification

Protocol: `docs/plans/placebo-l010-campaign.md`, decision 0028 and
`docs/plans/click-tab-2-family-extension.md`. Runs are
`e2placebo-l010-click-tab-2-s4231-c4096` (P),
`e2-click-tab-2-s4231-c4096` (E2) and `e1-click-tab-2-s4231-c4096` (E1).
Frozen E2/P configs differ only in experiment metadata and enabling the
within-group placebo shuffle. Weight stays 0.1; both start with fresh LoRA
from pinned Qwen3-1.7B. The 300 updates, batch four, eight rollouts, naive_sum,
scale_rewards none, optimizer settings and 4096 trajectory / 4096 prompt /
8192 context / eight-turn budgets are unchanged.

All arms use L4 and matching Python, packages, OpenEnv, episode-boundary and
Linux 6.8.0-146/glibc 2.39 stamps. The executable path differs because placebo
uses its isolated historical snapshot. Historical runtime sources match, with
no GPU-routing patch. Run dates and system load remain timing limitations;
matching stamps do not establish identical stochastic training trajectories.
The home-host confound does not apply to this family.

Every comparison pairs the same ordered seeds 4231100000-4231100199 and
byte-identical initial observations. The technical receipt passes 300 finite
optimizer records, all 600 evaluation episodes, tokenizer recounts, unchanged
frozen source/config/stack, final/update-300 adapter equality and four complete
cost attempts. Local checks reproduce the 600 new native audits, all three
metric reports and the shuffled reward replay: 193 changed assignments among
416 captured rollouts. Raw observer component values are unshuffled; composed
reward records the actual assignment.

The old controller was never restarted. The original direct tabs wrapper and
worker exited zero; GPU 1 was idle with no compute process at the completion
check. A local harvest exceeded the monitor's 60-second transfer limit; only
missing files were subsequently copied, preserving and verifying all existing
files. This did not rerun any experiment. The normal `review.json` and
`harvest_manifest.json` bind the tabs evidence. Inbox retains its separate
approved recovery authority.

Paired bootstrap intervals use 20,000 question resamples, RNG seed 0.
Compression is the median of per-question percentage changes on joint
successes, not a ratio of unpaired means. Gross success losses use a one-sided
95% exact binomial bound, without cancellation by gains. Four generic
off-target bounds use alpha=0.05/4. Checkpoints share the same questions and
are not independent replications. Intervals describe question sampling within
these policies, not training-seed variation or campaign-wide confirmation.

## All checkpoints

| Update | E1 successes | P successes | E2 successes | P failure / unfinished |
|---|---:|---:|---:|---:|
| 100 | 148 | 76 | 73 | 13 / 111 |
| 200 | 179 | 89 | 56 | 23 / 88 |
| 300, primary | 189 | 124 | 5 | 10 / 66 |

All denominators are 200. Earlier directions are retained, without selecting
a favorable checkpoint.

| Contrast / update | Gains / losses | Joint successes | Median token change, 95% CI |
|---|---:|---:|---:|
| E2 minus P, 100 | 29 / 32 | 44 | +8.02% [-17.91, +35.71] |
| E2 minus P, 200 | 41 / 74 | 15 | +25.38% [+16.25, +107.84] |
| E2 minus P, 300 | 0 / 119 | 5 | +85.09% [-55.53, +478.09] |
| P minus E1, 100 | 13 / 85 | 63 | +28.67% [-13.75, +73.14] |
| P minus E1, 200 | 6 / 96 | 83 | +26.11% [+12.58, +46.98] |
| P minus E1, 300 | 7 / 72 | 117 | -7.00% [-32.51, +15.04] |

At update 300, initially visible targets yield E1/P/E2 successes of 59/56/5
out of 61, and initially hidden targets yield 130/68/0 out of 139. E2 loses
51 visible and 68 hidden placebo successes, without gains. All five shared
successes are initially visible. Their index and P/E2 tokens are
15: 1437/639; 41: 503/931; 137: 857/1545; 154: 249/1406; 191: 178/1029.

P versus E1 has a strong descriptive visibility split: 55 joint visible
successes shorten by median 69.72% [61.28%, 80.44% fewer], while 62 joint
hidden successes lengthen by 35.80% [23.16%, 58.51%]. P loses four visible
and 68 hidden E1 successes, gaining one and six respectively. These are
exploratory strata, not extra independent trials or multiplicity-adjusted claims.

## Task-grounded behavior and coverage

| Final event / opportunity | E1 | P | E2 |
|---|---:|---:|---:|
| Wrong visible-link choice / attributable first-call link selections | 5/194 | 9/133 | 15/20 |
| Tab change with target visible / successful first-call tab selections while visible | 61/61 | 2/4 | 0/2 |
| Same tab-change events / all first-call decisions with visible target | 61/250 | 2/130 | 0/246 |
| Invalid actions / answered calls | 0/552 | 0/338 | 804/834 |
| Adjacent repeated dispatches / dispatched calls | 1/552 | 79/338 | 502/793 |
| Exact repeat after observed error / next-turn error-following decisions | 0/0 | 0/0 | 501/630 |
| Unfinished with target visible / unfinished episodes | 2/6 | 4/66 | 55/180 |

Zero opportunities do not establish a zero conditional rate. Dispatch counts
include the first dispatch, which cannot repeat its predecessor. Falling
successful tab-change counts do not establish better decisions; E2 frequently
names tab controls with rejected identifiers, as established in the retained
original report. All final arms have single-call turns, so final next-turn
error repetitions are not precommitted calls from the same batch.

The decision-state ratios above cluster multiple decisions within episodes.
Every wrong-link denominator is selected by outcome: attributable native
terminations are E1 194, P 133 of 134 and E2 20. The ratios exclude unfinished
episodes to different degrees and cannot establish an unconditional cross-arm
behavior difference. Use paired episode incidence for that separate view:

| Event / contrast | New / resolved | Net incidence change, paired 95% CI |
|---|---:|---:|
| Wrong visible-link choice, E2 minus P | 15 / 9 | +3.0 pp [-1.5, +8.0] |
| Wrong visible-link choice, P minus E1 | 7 / 3 | +2.0 pp [-1.0, +5.0] |
| Tab change with target visible, E2 minus P | 0 / 2 | -1.0 pp [-2.5, 0.0] |
| Tab change with target visible, P minus E1 | 0 / 59 | -29.5 pp [-36.0, -23.0] |
| Unfinished with target visible, E2 minus P | 53 / 2 | +25.5 pp [+19.0, +32.0] |
| Unfinished with target visible, P minus E1 | 4 / 2 | +1.0 pp [-1.5, +3.5] |

The wrong-visible-link incidence differences are not established. These paired
episode intervals are exploratory and do not adjust for all reported labels;
they also do not equal a comparison conditional on identical opportunities.

All 66 P unfinished episodes hit the generation allowance; P has no final
voluntary stop or turn-cap stop. E2 has 174 generation caps, two turn caps,
four voluntary stops and 20 native terminations, versus P's 134 terminations.
Of the 119 E2 losses relative to P, 112 hit generation limits, two hit turn
limits and five terminate wrongly; 114 have errors and 105 have repeats.
The five without an answered call are indices 0, 27, 72, 101 and 156.
Of P's 72 E1-success regressions, 64 hit generation limits and eight terminate
wrongly; none has an action error, 38 have repeats and 20 have no answered call.

| Generic episode event | P / E2 | New / resolved in E2 | Upper new-event bound |
|---|---:|---:|---:|
| Wrong native termination | 10 / 15 | 15 / 10 | 12.77% |
| Voluntary no-tool stop | 0 / 4 | 4 / 0 | 5.54% |
| Invalid action | 0 / 195 | 195 / 0 | 99.32% |
| Repeated action | 39 / 179 | 144 / 4 | 78.89% |

None establishes the 5% new-event bound. The voluntary-stop endpoint is
inconclusive relative to that margin, not evidence of an increase above it.
For P versus E1, wrong-termination new/resolved counts are 8/3 (upper 8.32%)
and repeat counts 39/1 (26.58%): those bounds fail. Invalid actions and
voluntary stops each have 0/0 transitions (upper 2.17%), passing the bound.

P's native terminal action is unattributed in four update-100 episodes
(45, 102, 115, 166), one update-200 episode (18) and one final episode (185).
All six are native failures. The clicked bid was not an identified visible
link in the saved state; these cases retain unknown target attribution and
are not silently labelled visible wrong-link choices. E2-200 retains its two
previously documented unknown terminals. All feedback formats are recognized.
One P-100 episode has a multi-call turn; final P has none. Decision labels
use only observations available when the first call was selected. The accepted
continuation/batch boundary remains unchanged, with no new strict per-action
transcript-cap sensitivity claimed.

Compact evidence covers every new paired regression, gain and shared success.
A supplemental unblinded seed-17 sample contains four final P budget failures,
four wrong terminations and four efficient-correct controls (successful-token
first quartile <=389). It confirms no-action caps (166, 144), navigation then
cap (124), repeated current-tab selection (120), wrong exact text/case choices
(58, 121, 12), unknown terminal attribution (185), and direct exact visible
target selections (195, 16, 63, 91). All six unknown P terminals were inspected.
This is a task-state consistency review with known outcomes, not independent
blinded semantic validation; raw reasoning was not exhaustively labelled.
Intent, neglected reasoning and reward-bias substitution remain unestablished.

E0-fixed short/long thresholds remain 726.4/3036.75 assistant tokens.
Short-success flags are 51/189, 47/124 and 1/5 for E1/P/E2; long-success
flags are 0/189, 1/124 and 0/5. These flags alone are not harm labels.

## Training and costs

| Diagnostic | E1 | P | E2 |
|---|---:|---:|---:|
| Mean fraction of zero-SD reward groups | 0.5150 | 0.3133 | 0.3400 |
| Zero-gradient updates / 300 | 36 | 6 | 16 |
| Last-ten-update captured successes / 320 | 290 | 179 | 5 |
| Last-ten task-variable groups / 40 | 8 | 6 | 5 |
| Last-ten composed-variable groups / 40 | 8 | 25 | 5 |
| Training allocated GPU-hours | 31.219 | 37.887 | 39.275 |
| Three evaluations, allocated GPU-hours | 7.217 | 14.758 | 10.940 |
| Total allocated GPU-hours | 38.435 | 52.645 | 50.214 |

Shuffling preserves the within-group component multiset, not its relationship
to task reward, total reward variance, gradient noise or future state visitation.
More active updates do not by themselves imply more successful learning.

Final mean inference seconds per episode are 45.775/72.043/72.067 for E1/P/E2;
total episode times are 49.806/74.995/76.745 seconds. Relative to P, E2 saves
-0.024 inference seconds [-8.060, +8.263] and -1.750 total seconds
[-9.759, +6.489]. P versus E1 saves -26.268 inference seconds
[-34.707, -18.134] and -25.189 total seconds [-33.646, -17.002]. Negative
savings mean slower. Neither all-question comparison demonstrates a latency
saving.

Joint-success timing differs. On five shared successes, E2 minus P saves
-19.85 inference seconds [-40.08, +7.97] and -21.77 total seconds
[-42.37, +6.43]. On 117 shared successes, P minus E1 saves +7.38 inference
seconds [+0.97, +13.77] and +8.98 total seconds [+2.50, +15.44]. Thus P is
faster on that selected successful subset, despite being slower overall and
losing 72 E1 successes. No break-even is computed from this subset: its savings
cannot be extrapolated to the full workload, whose success mix changed and
whose mean latency worsened. P also costs 6.669 more training allocation-hours
than E1. Timing was not randomized across machine load. No monetary or energy
cost is inferred from allocated GPU-hours.

## Evidence

Reproducible analysis, raw-input hashes, task-state evidence, direct checks and
the independent Opus 5.5/xhigh review record are under
`placebo-l010-20261018-ops/tabs-analysis-20261024/`. The review authority is
`review-status.json`; technical admission alone is not scientific approval.
The current direct validation is `validation-v4.json`, produced by
`validate_v4.py` with execution retained in `validation-v4-execution.log`.
`review-corrections.json` and `review-corrections-final.json` record accepted
findings. Earlier drafts, validations and unsuccessful review attempts remain
preserved under their original names; their hashes refer to those earlier
versions, not this report.
Raw results and the original campaign findings are unchanged. No further
experiment follows from this report.
