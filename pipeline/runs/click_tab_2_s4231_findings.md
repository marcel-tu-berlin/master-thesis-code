# Click-tab-2 seed 4231: E0/E1/E2 findings

Scientific review completed 2026-10-18 from saved, hash-verified results.

E2 at the declared 0.1 relative successful-length cost severely regressed in
this seed: 5/200 native successes versus E1's 189/200. There were 184 paired
losses and no gains. The five shared successes do not establish the declared
10% compression target. Task-only specialization worked; by update 300 the
shaped policy predominantly produced rejected identifiers and repeated failed
actions. The update-200 decline instead involved wrong terminations and repeats
that mostly dispatched without an action error.
This is a completed negative result for the fixed comparison, not a reason to
replace the checkpoint or restart an arm.

![Checkpoint success and five shared successes](click-tab-2-campaign-s4231-ops/final-analysis-20261017/comparison.png)

## Admission, repair and provenance

All three arms exited successfully. E1/E2 each completed 300 updates and three
200-episode checkpoint evaluations. The controller stopped after E2 because its
post-run auditor misclassified an argument-binding rejection as an executed
terminal click. At checkpoint 200, index 128 (seed 4231100128),
`click(bid="35", selector=".alink")` was rejected before dispatch. A later
`click(bid="35")` actually executed and selected a wrong link. The original
predicate excluded only `Action error:` feedback and missed the evaluator's
`{'error': ...}` binding-error form.

The user explicitly authorized repair and continuation. The corrected predicate
attributes a click only when its feedback begins `Page now:\n`. The original
auditor, reviewer, manifest, training/evaluation source and E0/E1 receipts remain
unchanged. The repaired auditor/reviewer are separate, hashed files under
`click-tab-2-campaign-s4231-ops/audit-repair-20261017/`. Its reviewer refuses to
replace an existing artifact with different bytes. It also verifies all frozen
operation hashes and both earlier admission receipts before admitting E2.

The regression test failed on the original auditor and passes on the corrected
copy, including adapter-error, binding-error and unknown-feedback cases.
Genuine post-terminal actions and corrupt native scores still fail. All 1,400
saved evaluation episodes now pass. Labels for the other 1,399 episodes are
identical. All 47 files in the preserved failure snapshot retained their hashes
after review. No policy, score, token count or experiment recipe was changed.
The earlier blocker record and failure snapshot remain historical evidence.

The full frozen remote review passed source/stack/config checks, episode and
metric recomputation, observed reward arithmetic, cost-ledger reconciliation,
and checkpoint checks, including final-adapter equality with update 300.
Non-weight harvests contain 12 E0, 30 E1 and 30 E2 files, all hash-verified,
including checkpoint-evals. Adapter weights remain on the GPU host. Direct
verification also passed the CPU suite: 561 passed, 7 skipped.

Independent counterpart review with Claude Opus 5.5 at xhigh completed on
2026-10-18 after authentication was restored. It confirmed the auditor repair
and saved statistics, and required three reporting corrections: distinguish
the update-200 pattern from the final collapse, identify rejected tab-anchor
HTML IDs and their relation to the tool's bid argument, and correct the original
source locations. Those corrections and a reproducible identifier tally are
recorded in `counterpart-disposition-20261018.json`. A follow-up review confirmed
that no required fix remained. Primary results and frozen artifacts are
unchanged. The verdict and follow-up are in
`counterpart-review-20261018-native.json` and
`counterpart-corrections-review-20261018.txt` under `final-analysis-20261017/`;
`validation-counterpart-20261018.json` records final hashes and review status.
Earlier connection and authentication failures remain in their original logs.

## Frozen experiment and statistical methods

Protocol: `docs/plans/click-tab-2-family-extension.md`, carrying forward the
fixed analysis margins in `docs/plans/e0-e2-campaign.md`. Runs are
`e{0,1,2}-click-tab-2-s4231-c4096`. Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, independent fresh base starts,
batch four, eight rollouts, 300 updates, `naive_sum`, `scale_rewards: none`,
4096 prompt / 4096 trajectory / 8192 context tokens and eight turns were retained.
Frozen E1/E2 configs differ only in experiment ID, description and enabling
relative successful-length cost at weight 0.1. All phase stack stamps match the
manifest after lexical normalization of the executable path.

The same 200 seeds, 4231100000-4231100199, and byte-identical initial
observations occur in all seven evaluations. Training uses the disjoint
4231000000-4231000499 block. Update 300 is primary; earlier checkpoints are
repeated observations of the same question set, not independent replications.

Native success is primary. Wilson intervals describe individual success rates.
Paired net success and token intervals use 20,000 resamples, RNG seed 0.
Compression uses the median per-question percentage change on joint successes,
not the ratio of aggregate means. The 5-percentage-point preservation criterion
uses a one-sided 95% exact binomial upper bound on the unconditional probability
of an E1 success becoming an E2 failure; gains cannot offset losses. The 10%
compression criterion requires the upper 95% bootstrap bound to be at most -10%.
These intervals cover evaluation-question sampling within this single training
seed, not training-seed variation. With only five joint successes the token
bootstrap is particularly unstable and survivor-selected.

## Primary comparison and checkpoint trends

| Policy | Native success / 200 | Wilson 95% CI | Failure | Unfinished | Mean assistant tokens on that policy's successes |
|---|---:|---:|---:|---:|---:|
| E0 base | 47 (23.5%) | 18.2%-29.8% | 46 | 107 | 987.1 |
| E1 update 100 | 148 (74.0%) | 67.5%-79.6% | 28 | 24 | 879.0 |
| E1 update 200 | 179 (89.5%) | 84.5%-93.0% | 11 | 10 | 725.7 |
| E1 update 300 | 189 (94.5%) | 90.4%-96.9% | 5 | 6 | 1048.6 |
| E2 update 100 | 73 (36.5%) | 30.1%-43.4% | 58 | 69 | 994.2 |
| E2 update 200 | 56 (28.0%) | 22.2%-34.6% | 83 | 61 | 1011.2 |
| E2 update 300 | 5 (2.5%) | 1.1%-5.7% | 15 | 180 | 1110.0 |

The per-policy success-token means above have different success composition;
they are descriptive, not the efficiency comparison.

At update 300 the paired success change is -92.0 percentage points, with a 95%
bootstrap interval of [-95.5, -88.0]. Exact McNemar p = 8.16e-56. The loss
probability is 184/200 = 92.0%; its one-sided exact upper 95% bound is 94.92%,
and its lower 95% bound is 88.10%. The 5-point preservation requirement fails
with clear evidence of harm in this comparison, not merely imprecision.

On the five shared successes, median token change is -192 tokens (95% CI
[-319, +1206]), or -15.72% (95% CI [-27.80%, +355.75%]). Three pairs are
shorter and two longer; the sign-test p is 1.0. Mean paired tokens increase
from 994.6 to 1110.0, a difference of +115.4. The compression target is not
established, and the median point estimate alone cannot support an efficiency
claim after losing 184 successes.

| Checkpoint | Paired E2 gains / losses | Joint successes | Median paired token change, 95% CI |
|---|---:|---:|---:|
| 100 | 15 / 90 | 58 | +18.22% [-0.66%, +39.31%] |
| 200 | 5 / 128 | 51 | +38.97% [+6.08%, +66.97%] |
| 300, primary | 0 / 184 | 5 | -15.72% [-27.80%, +355.75%] |

Task-only specialization, E1-300 versus E0, yields 142 gains and no losses:
+71.0 points (95% CI [+64.5, +77.0]), McNemar p = 3.59e-43. On their 47
joint successes E1 uses a median +386 tokens [156, 552], or +54.99%
[25.55%, 79.87%]. This is improved task success with longer shared successful
responses, not an E2 shaping result.

## Visibility and task-grounded behavior

| Policy | Initially visible target, successes / 61 | Initially hidden target, successes / 139 |
|---|---:|---:|
| E0 | 39 | 8 |
| E1-100 | 50 | 98 |
| E1-200 | 61 | 118 |
| E1-300 | 59 | 130 |
| E2-100 | 43 | 30 |
| E2-200 | 7 | 49 |
| E2-300 | 5 | 0 |

The final paired losses are 54/61 in the initially visible stratum and 130/139
in the hidden stratum, with no gains in either. All five shared successes are
initially visible. Their index and E1/E2 tokens are: 15: 885/639;
41: 1250/931; 137: 339/1545; 154: 1278/1406; 191: 1221/1029.
Every one of these five E2 successes first names a visible tab anchor's HTML
`id` as a `bid`; these calls are rejected. Index 137 additionally uses `tabs-2`,
a tab-panel identifier. Each has one to three rejected calls before the correct
visible-link click. Four corresponding E1 traces successfully switch to tab 2,
return to tab 1 and then succeed; the fifth succeeds directly. Thus E2 also
emits tab-naming calls, but those calls fail to navigate. This comparison
describes emitted arguments and executed actions, not intent.

Decision labels apply only to the first answered call of a model turn, where
the displayed observation was available when the action was chosen. Later
calls in the same batch remain precommitted. Denominators below distinguish
observed decisions from successful selections. Fewer successful selections can
make an event disappear without improving decision quality.

| Final-endpoint event / opportunity | E1-300 | E2-300 |
|---|---:|---:|
| Wrong visible-link choice / successful first-call visible-link selections | 5 / 194 | 15 / 20 |
| Tab change while target visible / successful first-call tab selections while visible | 61 / 61 | 0 / 2 |
| Same tab-change events / all first-call decisions with visible target | 61 / 250 | 0 / 246 |
| Invalid actions / answered calls | 0 / 552 | 804 / 834 |
| Adjacent repeated dispatches / dispatched calls | 1 / 552 | 502 / 793 |
| Exact repeated call following an observed prior error / next-turn error-following decisions | 0 / 0 | 501 / 630 |
| Unfinished with target visible / unfinished episodes | 2 / 6 | 55 / 180 |
| Voluntary no-tool stop with target visible / voluntary stops | 0 / 0 | 0 / 4 |

A zero denominator is no opportunity, not evidence of a zero conditional rate.
The dispatch denominator includes first dispatches, which cannot be duplicates.
E1 has 552 first-call decision states and E2 has 834. Invalid actions occur in
0/200 versus 195/200 episodes; repeated dispatches in 1/200 versus 179/200.
The final evaluations contain no multi-call turns, so final error-following
repetitions cannot be mistaken for precommitted same-batch actions.

E2's common invalid argument is `bid="ui-id-2"` (553 failures), followed by
`ui-id-3` (68) and `ui-id-1` (54). These are the HTML `id` attributes of the
three tab anchors in the displayed page, not their `bid` attributes. Thus
675 of the 804 errors are rejected click calls naming observed tab controls
with the wrong identifier field. Other calls use CSS-like expressions or
unavailable tool names. The retained
`verify_report_corrections_20261018.py` recounts these failed calls and verifies
each identifier against the observation preceding it; its hashed output is
`report-corrections-evidence-20261018.json`. E2 error episodes across updates
100/200/300 are 13/3/195 (27/7/804 events), while repeated-dispatch episodes are
40/71/179. Update 200 has 83 wrong native terminations, with repeats mostly
occurring without errors. This is not a monotonic rise in invalid-ID behavior.
Among all 184
success regressions, 179 have errors and 164 have repeated dispatches. The
remaining five error-free regressions (indices 0, 27, 72, 101, 156) exhaust
generation before producing an answered action. Lost-success stop reasons are
164 generation caps, 15 wrong native terminations, four voluntary stops and
one turn cap. The evidence supports a shift toward invalid/repeated tool use
and budget exhaustion. It does not identify a cognitive cause or establish
reward-bias substitution.

For wrong visible-link episode incidence, paired new/resolved counts are
15/5, net +5.0 points (bootstrap 95% CI [+1.0, +9.5]). For unfinished-visible
incidence they are 55/2, net +26.5 points [+20.0, +33.0]. Tab-change-with-visible
incidence falls by 61 episodes, but E2 still emits rejected tab-naming calls.
The 0/2 and 0/246 rows count successful tab changes, so they do not establish
an absence of calls naming tabs while the target is visible. The decrease is
not an improvement claim.

Task-grounded auditing covered all 1,400 episodes. Native terminal attribution
remains unknown for E0 index 170 and E2-200 indices 1 and 126: the clicked bid
was not an identified visible link in the saved state. All three have native
failure outcomes. Final E1/E2 have zero unattributed terminal actions. No tool
feedback format is unknown, and action/dispatch/error/repetition counters
reconcile exactly with saved traces. Unknown clicked targets are not relabelled
as a visible wrong-link selection.

The seed-17 masked sample contains 24 cases, six from each precedence-ordered
stratum: budget, voluntary stop, invalid/repeated, efficient correct. Efficient
correct means at most the pooled successful-token first quartile, 562 tokens.
Available stratum counts are 376, 81, 54 and 175. Seed and episode index were
removed from the case records; arm and step were not episode fields, and their
mapping was kept in a separate key. Timing fields and outcomes stayed visible;
aggregate results were already known. Labels were saved before revealing the
key. The six efficient controls select
the exact visible target, directly or after valid navigation/current-tab click.
The sample also confirms invalid-ID recovery, repeated tab selections, visible
target abandonment and one genuinely unattributed terminal. This is a
supplemental manual check, not independent blinded validation. Every regression
and shared success has a saved call/state review; raw assistant reasoning was
not exhaustively manually labeled.

## Stops, generic margins and accepted execution boundary

| Policy | env_done | Generation cap | Turn cap | No tool call |
|---|---:|---:|---:|---:|
| E0 | 93 | 88 | 0 | 19 |
| E1-100 | 176 | 10 | 0 | 14 |
| E1-200 | 190 | 1 | 0 | 9 |
| E1-300 | 194 | 6 | 0 | 0 |
| E2-100 | 131 | 48 | 0 | 21 |
| E2-200 | 139 | 46 | 1 | 14 |
| E2-300 | 20 | 174 | 2 | 4 |

Final non-termination rises from 6/200 to 180/200. Paired transitions are:
169 env_done -> generation cap; 20 env_done -> env_done; four env_done ->
no-tool; one env_done -> turn cap; five generation cap -> generation cap;
one generation cap -> turn cap. The 174 generation caps are distinct from
two turn caps and four voluntary stops.

| Generic episode event | E1 / E2 | New / resolved | One-sided 98.75% upper bound on new-event probability |
|---|---:|---:|---:|
| Wrong native termination | 5 / 15 | 15 / 5 | 12.77% |
| Voluntary no-tool stop | 0 / 4 | 4 / 0 | 5.54% |
| Invalid action | 0 / 195 | 195 / 0 | 99.32% |
| Repeated action | 1 / 179 | 178 / 0 | 93.44% |

None establishes the 5-point bound, using alpha 0.05/4 for the four endpoints.
The no-tool result alone is inconclusive relative to that margin, not evidence
of a greater-than-5-point increase. The task-grounded panel is exploratory;
its intervals are not an additional multiplicity-adjusted equivalence test.

E0-fixed short/long reference thresholds are 726.4 and 3036.75 assistant tokens.
Final successful short-response flags are 51/189 (E1) and 1/5 (E2); neither
has a successful long-response flag. These percentile flags do not establish
harmful underthinking or beneficial compression.

Accepted execution semantics are unchanged: whole-trajectory allowance governs
continuation, precommitted calls execute until native completion, and retained
terminal feedback is not a hard per-action transcript cutoff. E0 has 60
multi-call turns in 59 episodes; E2-100 has 16 in 16; other evaluations have
none. The grounded detector separates precommitted calls from observed feedback.
No new tokenizer-based saved-prefix sensitivity was run for this family, and no
strict per-action-cap result is claimed. Error feedback consumes continuation
budget, so a generation-cap label need not mean 4096 assistant tokens.

## Training and measured costs

Both arms have 300 finite optimizer records and 416 observed rollouts across
updates 1, 100, 200 and 291-300. Group size eight and all recorded task,
relative-length and composed rewards recompute correctly.

| Training diagnostic | E1 | E2 |
|---|---:|---:|
| Mean fraction of groups with variable composed reward | 48.5% | 66.0% |
| Updates with a live reward group | 88.0% | 94.67% |
| Zero-gradient updates | 36 | 16 |
| Native successes in recorded final 320 rollouts | 290 | 5 |
| Task-variable groups in those final 40 groups | 8 | 5 |
| Composed-variable groups in those final 40 groups | 8 | 5 |

The final training sample also deteriorated, consistent with the held-out
collapse. It is not another held-out sample. Greater gradient activation over
the full run is part of the shaping intervention; this comparison cannot isolate
length assignment from that activation or training stochasticity.

| Allocated L4 GPU-hours | Training | Evaluation |
|---|---:|---:|
| E0 | 0 | 3.013 |
| E1 | 31.219 | 7.217 |
| E2 | 39.275 | 10.940 |
| Total | 70.493 | 21.169 |

All nine phase attempts have matched start/end records and complete status;
no unmatched cost start is silently dropped. Total recorded campaign allocation
is 91.662 GPU-hours. These are allocation/wall-time measurements, not kernel
utilization, energy or money. Native prelaunch probes and local postprocessing
are outside this phase ledger.

Mean final inference time across all questions is 50.97 s (E0), 45.78 s (E1)
and 72.07 s (E2). Paired E1-minus-E2 savings are -26.29 s [-30.70, -21.99]
across all 200, and -4.78 s [-28.93, +11.35] on the five joint successes.
E2 has no finite inference-cost break-even under either point estimate. Its
training cost is 39.275 hours, 8.056 hours above E1, before research evaluations.

For E1 versus E0, all-question savings are 5.19 s [0.14, 10.22], giving a
purely arithmetic break-even of about 21,637 evaluations against E1's full
31.219-hour training cost. On their joint successes E1 is slower by 30.65 s
[22.99, 38.47], so that subset has no finite break-even. This contrast illustrates
why all-question deployment timing and successful-pair timing must stay separate.

All tab arms used physical GPU 1 on the same L4 host and frozen software stack.
Earlier checkbox work shared that host until its user-requested stop on
2026-10-15; CPU, memory and I/O contention can affect timings even on separate
GPUs. There are no matched contention measurements, so timing changes cannot
establish a shaping mechanism. Checkbox's discarded run cost and home-host
continuation belong to its already completed findings and are not counted twice
in this tab ledger.

## Reproduction and conclusion limits

Detailed statistics, event opportunities, provenance hashes and trace evidence
are in `click-tab-2-campaign-s4231-ops/final-analysis-20261017/`:
`analysis.json`, `behavior_records.json`, `trace_review.json`,
`masked_cases.json`, `masked_labels.json`, `masked_key.json` and
`masked_selection.json`. `analyze.py` rehashes the harvests, verifies aligned
instances/configs, recomputes the statistics and refuses a changed existing
analysis. `comparison.png` and `comparison.pdf` are derived figures.

Original reviewer/auditor sources remain in the operations root and the
preserved `review-blocker-20261017/snapshot/`. The repaired copies, regression
test, local red/green result and 47-file preservation proof are retained in
`audit-repair-20261017/`.
E2's new `review.json` explicitly binds the repair source hashes as well as
original operations and run inputs. The original E0/E1 receipts are unchanged.

This seed provides clear evidence against success preservation at dose 0.1,
no established 10% compression, and a substantial shift toward invalid/repeated
actions and budget exhaustion. It does not estimate training-seed variance,
prove an internal intent, or isolate a causal mechanism of the collapse.
Independent counterpart review passed after the reporting corrections above;
the single-seed, causal and supplemental-masking limitations remain.
No additional dose, family, seed, warm start or E3 run is implied or launched.

The tab Taskwarrior task is completed and the shared heartbeat is absent.
The successful tab harvest/live snapshot was obtained on 2026-10-17 at
20:00:54 UTC. A final status refresh on 2026-10-18 could not reach gpu-l4;
RUNNING.md retains that explicitly marked last-verified snapshot. This does not
leave an unharvested result or a pending experiment phase.
