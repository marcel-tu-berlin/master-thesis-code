# Click-checkboxes-soft seed 4232: matched 0.1 placebo

Reviewed 2026-10-19; corrected after counterpart review on 2026-10-20. The new placebo is
`e2placebo-l010-click-checkboxes-soft-s4232-c4096-home` (P). Its 29 non-weight
artifacts were harvested and hash-verified. All 300 training updates and the
three 200-question evaluations passed technical review. The controller then
started the authorized tic-tac-toe placebo; the five-family task remains open.

The fixed final comparison shows that E2's response-specific length cost has a
very different outcome from the shuffled-cost placebo on the same home GPU.
E2 is substantially shorter but has fewer native successes and many fewer exact
completions. Both arms activate almost every training group, so extra gradient
activity alone did not reproduce the final E2 outcome in this seed. The placebo is
not an inert baseline: it develops long responses and 14 budget stops.
These are exploratory, single-seed mechanism results, not replicated causal
estimates or evidence of intent.

## Primary endpoint: E2 versus placebo

P and E2 both ran on the RTX 4090. All comparisons pair the same 200 seeds and
byte-identical initial observations. Positive success differences below favor
E2; negative token changes mean shorter E2 responses. Paired bootstrap intervals
use 20,000 question resamples, seed 0, and describe these fitted policies only.
Update 300 remains primary; earlier checkpoints are descriptive.

| Final outcome | E1 reference | Placebo P | E2 |
|---|---:|---:|---:|
| Native success | 191/200 (95.5%) | 183/200 (91.5%) | 162/200 (81.0%) |
| Exact-goal success | 139/200 (69.5%) | 156/200 (78.0%) | 62/200 (31.0%) |
| Native success with non-exact selections (omissions and/or extras) | 52 | 27 | 100 |
| Wrong native submission | 8 | 3 | 33 |
| Unfinished | 1 | 14 | 5 |
| Successful-only unpaired mean assistant tokens | 622.81 | 1695.61 | 319.41 |

Exact completion requires the full correct selection and an executed Submit.
The successful-only means use different question sets; the paired token
statistics below are the compression evidence.

| E2 minus P, final | Estimate | Paired 95% interval |
|---|---:|---:|
| Native success difference | -10.5 pp | [-16.5, -4.5] |
| Exact success difference | -47.0 pp | [-55.0, -38.5] |
| Non-exact native-success incidence | +36.5 pp | [28.5, 44.5] |
| Median token change, 152 joint native successes | -81.01% | [-81.96, -79.48] |
| Median token change, 52 joint exact successes | -79.10% | [-81.02, -75.67] |

Native transitions comprise 10 gains and 31 regressions, exact McNemar
p=0.00145049. E2 is shorter on 151 joint native successes and longer on one.
The paired median absolute change is -1241.5 tokens [-1404.5, -1110.0]; joint
means are 1649.22 for P and 314.93 for E2. On joint exact successes, all 52
responses are shorter, with median -1023 tokens [-1206, -840]. Exact-success
transitions have 10 gains and 104 regressions (McNemar p=7.24e-21).

E2 passes the inherited 10% compression criterion against P. It fails the
success-preservation rule: the one-sided exact 95% upper bound on
the probability that P succeeds and E2 fails is 20.34%, above the 5% margin. Its lower bound is
11.45%, also above that margin. This is evidence of gross regression beyond
5%, without cancelling losses against gains. The two-sided 95% bootstrap net-success interval reaches
-4.5 pp, so it does not separately establish a net loss greater than 5 pp at
that confidence level. The exact-regression upper bound is 58.01%.

## Secondary reference: placebo versus E1

P minus E1 native success is -4.0 pp [-8.5, 0.5], with seven gains and 15
regressions (McNemar p=0.1338). A non-significant net difference does not prove
preservation: the gross-regression upper bound is 11.31%, above 5%.

P has 17 more exact completions: +8.5 pp [0.5, 16.5], with 43 gains and 26
regressions. The approximate bootstrap interval narrowly excludes zero while
the exact McNemar test gives p=0.0533; treat this secondary contrast as
borderline, not a confirmed exact-success improvement. Native non-exact
success falls by 12.5 pp [-20.0, -5.5].

The placebo lengthens responses substantially at the final checkpoint. On 176
joint native successes, median per-question change is +171.25%
[144.08, 194.30], and median absolute change is +967 tokens [834.5, 1060].
On 113 joint exact successes, the median percentage change is +172.19%
[145.01, 206.37]. The inherited compression target fails. E1 ran on L4,
so this secondary comparison retains the original hardware/OS/driver confound.

## Checkpoints and behavior

| Policy | Native success | Exact success | Non-exact native success | Wrong submission | Unfinished | Mean successful tokens |
|---|---:|---:|---:|---:|---:|---:|
| E1-100 | 180 | 107 | 73 | 20 | 0 | 1447.77 |
| E1-200 | 195 | 149 | 46 | 5 | 0 | 749.63 |
| E1-300 | 191 | 139 | 52 | 8 | 1 | 622.81 |
| P-100 | 196 | 133 | 63 | 4 | 0 | 881.38 |
| P-200 | 189 | 137 | 52 | 10 | 1 | 658.36 |
| P-300 | 183 | 156 | 27 | 3 | 14 | 1695.61 |
| E2-100 | 185 | 112 | 73 | 14 | 1 | 793.82 |
| E2-200 | 183 | 123 | 60 | 17 | 0 | 535.52 |
| E2-300 | 162 | 62 | 100 | 33 | 5 | 319.41 |

At updates 100/200/300, E2 minus P native success is -5.5/-3.0/-10.5 pp;
paired median token change on joint native successes is -11.71/-21.37/-81.01%.
The large final length contrast includes a late rise in placebo verbosity as
well as E2 shortening. These repeated checkpoints are not independent replications,
and no earlier checkpoint replaces the declared final endpoint.

| P minus E1, descriptive checkpoint | Native change, pp (95% CI) | Native gains / regressions | Joint native successes | Median paired token change (95% CI) |
|---|---:|---:|---:|---:|
| 100 | +8.0 [3.5, 13.0] | 20 / 4 | 176 | -39.51% [-44.31, -32.30] |
| 200 | -3.0 [-7.0, 1.0] | 5 / 11 | 184 | -12.96% [-18.37, -5.62] |
| 300 | -4.0 [-8.5, 0.5] | 7 / 15 | 176 | +171.25% [144.08, 194.30] |

The placebo also shortened responses substantially early in training relative
to E1. At update 100 its native-success McNemar p is 0.00154. Exact-success
changes at updates 100/200 are +13.0 pp [4.5, 21.5] and -6.0 pp [-14.0, 2.0];
joint-exact median token changes are -29.04% [-38.44, -1.18] (80 pairs) and
-10.25% [-17.46, -0.28] (110 pairs). The final mechanism observation must not
erase this earlier shared shortening or the E1/home host confound.

The original E0 length thresholds remain P10=572.5 and P75=1632.75 tokens.
Final short-success flags are E1 58.64%, P 3.83% and E2 95.68%; long-success
flags are E1 1.05%, P 47.54% (87/183) and E2 0%. These are descriptive length
flags against one fixed ruler, not evidence of harmful underthinking or
overthinking. They make the placebo's late length increase visible.

All 600 new placebo trajectories were mechanically checked against the retained
native synonym/state/action auditor; every terminal action was attributed.
The native scorer counts correctly unselected irrelevant boxes as well as
correctly selected required boxes. This known gap permits both omitted required
boxes and extra selections to pass. The inherited audit key
`native_success_incomplete` means native success without an exact selection,
including extra-only cases. Final P has 27/183=14.75% non-exact native successes, versus
100/162=61.73% in E2. This contrast does not establish intentional exploitation.

| Non-exact native successes | E1 | P | E2 |
|---|---:|---:|---:|
| Omissions only | 45 | 12 | 49 |
| Extras only | 2 | 13 | 29 |
| Both | 5 | 2 | 22 |

| Final submission diagnostic | P | E2 |
|---|---:|---:|
| Attributed Submit episodes | 186/200 | 195/200 |
| Submitted episodes with omissions | 17/186 | 101/195 |
| Missing required boxes | 19/566 | 156/601 |
| Submitted episodes with extras | 18/186 | 73/195 |
| Selected irrelevant boxes | 18/385 | 83/401 |
| Incomplete / complete decision states | 201 / 4 | 210 / 3 |
| Direct Submit from incomplete state | 0/201 | 10/210 |
| Direct Submit from complete state | 1/4 | 2/3 |
| Submit after precommitted actions | 185/186 | 183/195 |
| Unknown terminal-action attribution | 0 | 0 |

These rates condition on different submitted populations. Missing and extra
selections can coexist; checkbox opportunities and decisions within episodes
are correlated. All 185 P and 183 E2 precommitted submissions were chosen from
an incomplete pre-batch state. Later feedback could not inform those already
emitted actions, so these counts are not evidence of ignored feedback. Direct
incomplete submissions rise by 5.0 pp [2.0, 8.0] across paired episodes.

P's final stops are 186 env-done and 14 generation-cap; E2 has 195 env-done
and five no-tool stops. No arm has a max-turn stop. Ten P budget cases emit no
tool call and consume 4096 assistant tokens; four stop after prior actions,
including three with an exact checkbox set but no Submit. Assistant-token
counts below 4096 in those four cases are compatible with a trajectory budget
that also charges tool feedback. All 14 are unfinished, not successful goals.

The 31 native E2 regressions against P comprise 27 wrong submissions and four
no-tool stops. The 104 exact regressions comprise 46 omission-only submissions,
23 extra-only submissions, 31 with both errors, and four no-tool stops.
Against E1, P's 15 native regressions comprise three wrong submissions and
12 budget stops. Longer P responses therefore introduce a different failure mode.

Both P and E2 have zero final invalid/repeated-action episodes. For the inherited
four-event panel (wrong submission, voluntary no-tool stop, invalid action,
repeated action), E2 introduces respectively 31, five, zero and zero events
relative to P. Their one-sided exact upper bounds at alpha=0.05/4 are
22.12%, 6.26%, 2.17% and 2.17%. The first two preservation bounds fail;
the latter two pass. The no-tool failure reflects imprecision: five new cases
do not establish a probability above 5%. These four bounds do not provide simultaneous coverage
for all report diagnostics. Budget stops are reported separately above.

The secondary P-versus-E1 panel has new/resolved counts of 3/8 wrong
submissions, 0/0 voluntary no-tool stops, 0/12 invalid-action episodes and
0/1 repeated-action episodes. Its upper bounds are 4.79%, 2.17%, 2.17% and
2.17%, so all four pass. The 12 new budget-stop regressions remain outside
that panel; passing it does not imply overall preservation.

Saved action-level summaries cover all final regressions and all 14 P budget
cases. They retain executed calls and state changes, but not the assistant
text of the ten no-action budget responses. No manual semantic assessment of
those responses is claimed. The illustrative action-level review includes
indices 0, 1, 88 and 189 for E2 versus P; 2, 41 and 148 for P versus E1;
and six joint-exact controls per contrast.
For example, at index 88, P selects petite and furious then submits (1174 tokens),
while E2 omits petite and still passes natively (181 tokens). At index 148, P
selects the exact set but reaches the budget before Submit; E1 completes it.

Control indices were selected with `sorted(random.Random(17).sample(joint, 6))`
from ascending joint-exact indices: 72, 93, 95, 109, 126, 154 for E2/P, and
73, 74, 84, 95, 111, 183 for P/E1. All controls select the exact set and submit
in one batch in both policies. This review was unblinded and followed aggregate
analysis; it is a diagnostic inspection, not an independent masked validation.

## Integrity, training and cost

P and E2 frozen configs differ only in experiment ID, description and
`successful_length.placebo: true`. Weight is 0.1 in both. Their runtime source
hashes, pinned Python packages, OpenEnv revision and episode boundary match.
Both use the RTX 4090 and the same recorded Linux 7.2.5-3-omarchy/glibc 2.44
platform. E1's platform remains Ubuntu/L4. The retained native outcome and
batch/continuation semantics were not changed; no new hard per-action cap or
saved-prefix sensitivity is claimed.

The placebo replay validated all 416 captured rollouts, including the RNG
advances for uncaptured updates. It changed 355 recorded cost assignments.
Raw successful-length observer fields remain unshuffled; the actual composed
reward follows the shuffled assignments. The technical receipt also verified
checkpoint 100/200/300 files, final/300 adapter equality, all evaluation metrics,
initial observations, source/config/stack hashes and complete cost attempts.

Mean active-group fraction is 99.75% for P, 99.58% for E2 and 27.67% for E1.
P and E2 have no zero-gradient updates; E1 has 115. In the captured final ten
updates, P succeeds on 302/320 rollouts, versus E2's 264/320 and E1's 300/320.
Task reward varies in 9/40 P groups and 22/40 E2 groups; composed reward varies
in all 40 groups in both. The placebo matches the cost multiset within its own
sampled groups, not the evolving total-reward variance or gradient-noise
trajectory of E2. Similar gradient activity is not full distribution matching.

| Arm | Training allocation-hours | Evaluation allocation-hours | Total |
|---|---:|---:|---:|
| E1, L4 reference | 24.1348 | 7.6236 | 31.7583 |
| E2, 4090 reference | 13.2307 | 1.8752 | 15.1058 |
| New placebo, 4090 | 14.6714 | 3.1470 | 17.8184 |

The placebo took about 17 hours 49 minutes across the four recorded phases.
Evaluation includes all three checkpoints. Final all-episode mean latency is
28.02 seconds for P and 8.04 seconds for E2; mean policy-call time is 24.25
versus 4.24 seconds. These include failures. Same hardware removes the original
L4/4090 difference from this primary contrast, but sequential runs, desktop
resource sharing and runtime variability prevent treating it as a controlled
speed benchmark. Shorter latency does not compensate for lost task success.
No compute/energy equivalence across L4 and 4090 or deployment break-even is claimed.

This is one training seed, one placebo assignment stream and an exploratory
comparison authorized after the original E2 outcomes were known. Bootstrap
intervals do not include training-seed uncertainty. Secondary panels and
checkpoints have no overall multiplicity correction. The result supports
studying response-specific cost assignment beyond simple gradient activation;
it does not establish a general causal effect or justify an unapproved new run.

Analysis and behavior evidence are in
`placebo-l010-20261018-ops/checkbox-analysis-20261019/`: `analyze_v2.py`,
`analysis-v2.json`, `review_cases_v2.py`, and `behavior-review-v2.json`.
The corrected analysis asserts audit seed/index alignment before merging and
binds the auditor hash to both original manifests. All numerical results
match the preserved first analysis. The original scripts, outputs, validation
and pre-review report remain available; original E1/E2 reports and raw
artifacts remain unchanged. `correction-validation-20261020.json` retains
the native non-exact subtype counts, alignment checks, and actual test and
regeneration logs.

Authentication is restored. Independent Opus 5.5 at xhigh returned six
findings, addressed here by reporting the missing checkpoint/threshold/panel
results, correcting the non-exact label and uncertainty wording, checking
audit alignment/provenance, and narrowing the manual-review claim.
Corrective Opus 5.5/xhigh review passed on 2026-10-20; its verdict, input
hashes and execution receipt are retained as
`counterpart-correction-{output,receipt}-20261020`. This does not alter the
passing GPU-side technical review or authorize stopping other runs.
