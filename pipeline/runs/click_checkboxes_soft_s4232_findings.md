# Click-checkboxes-soft seed 4232: completed comparison

Reviewed 2026-10-17. E0 and E1 are the original L4 runs. E2 is the fresh
RTX 4090 run `e2-click-checkboxes-soft-s4232-c4096-home`. All seven evaluations
are complete and the 72 harvested non-weight files pass SHA-256 verification.
Update 300 is the primary endpoint.

E2 produces shorter successful responses, but loses native task success and
substantially more exact-goal success. The declared 10% compression target
passes; the five-percentage-point success-preservation criterion fails, with
evidence of harm beyond that margin for this observed pair. The host change
and one training seed prevent identifying this difference as a causal effect
of length shaping alone.

## Fixed endpoint and uncertainty

Both final policies answer the same 200 questions with byte-identical initial
observations. Positive differences below favor E2 for success; negative token
differences mean shorter E2 responses. Intervals resample paired questions
20,000 times, RNG seed 0. They measure evaluation-question uncertainty for
these fitted policies, not training-seed or host uncertainty.

| Final metric | E1 | E2 | Paired E2 minus E1, 95% interval |
|---|---:|---:|---:|
| Native success | 191/200 (95.5%) | 162/200 (81.0%) | -14.5 pp [-20.5, -8.5] |
| Exact-goal success | 139/200 (69.5%) | 62/200 (31.0%) | -38.5 pp [-46.5, -30.0] |
| Native success with incomplete selections | 52/200 (26.0%) | 100/200 (50.0%) | +24.0 pp [15.49, 32.50] |
| Mean tokens on 155 joint native successes | 614.01 | 318.45 | Median change -220 [-244, -201] tokens |
| Mean tokens on 51 joint exact successes | 503.55 | 281.41 | Median change -174 [-209, -148] tokens |

The native comparison has seven gains and 36 regressions, exact McNemar
p=0.000008963. Marginal native-success Wilson intervals are [91.67%, 97.61%]
for E1 and [75.00%, 85.83%] for E2. These marginal intervals are not the paired
interval for the difference.

On joint native successes, median per-question percentage token change is
-43.53% [-48.66%, -39.85%]. E2 is shorter on 152 pairs, equal on none and
longer on three. The upper interval endpoint is below -10%, meeting the fixed
compression criterion. The paired mean absolute difference is -295.57 tokens.
Successful-only unpaired means, 622.81 and 319.41, use different question sets
and are descriptive; they are not the primary compression estimate.

Preservation requires the one-sided 95% exact binomial upper bound on
P(E1 succeeds and E2 fails) to be at most 5%, without offsetting losses with
gains. Here 36/200=18% regress; the upper bound is 23.07% and the lower bound
is 13.66%. This is evidence of regression beyond the margin, rather than
merely insufficient precision. The net-success interval also lies below -5 pp.

Exact completion is a prespecified diagnostic, not a replacement reward. It
has 11 gains, 88 regressions and 51 joint exact successes; McNemar
p=4.530e-16. The exact-regression upper bound is 50.06%. On joint exact
successes, median percentage token change is -39.26% [-44.74%, -35.85%]; all
51 pairs are shorter. Thus compression persists among fully completed goals,
while the population of fully completed goals shrinks sharply.

All tests concern the fixed final comparison. Checkpoints and secondary
diagnostics are descriptive; their intervals do not have simultaneous coverage.
No best checkpoint or new dose was selected after seeing the data.

## Checkpoint trends and outcomes

| Policy | Native success | Exact success | Native-success incomplete | Native failure | Unfinished | Mean tokens, native successes |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 125 | 48 | 77 | 61 | 14 | 1224.98 |
| E1-100 | 180 | 107 | 73 | 20 | 0 | 1447.77 |
| E2-100 | 185 | 112 | 73 | 14 | 1 | 793.82 |
| E1-200 | 195 | 149 | 46 | 5 | 0 | 749.63 |
| E2-200 | 183 | 123 | 60 | 17 | 0 | 535.52 |
| E1-300 | 191 | 139 | 52 | 8 | 1 | 622.81 |
| E2-300 | 162 | 62 | 100 | 33 | 5 | 319.41 |

Every row has 200 episodes. Exact success plus incomplete native success equals
native success; native success plus native failure plus unfinished equals 200.

| Update | Paired native change, pp | Joint native token change, % | Paired exact change, pp | Joint exact token change, % |
|---|---:|---:|---:|---:|
| 100 | +2.5 [-2.0, 7.0] | -48.20 [-51.06, -41.72] | +2.5 [-6.0, 11.0] | -34.19 [-48.03, -15.59] |
| 200 | -6.0 [-10.5, -2.0] | -29.82 [-35.55, -22.57] | -13.0 [-20.5, -5.5] | -24.30 [-32.61, -17.36] |
| 300 | -14.5 [-20.5, -8.5] | -43.53 [-48.66, -39.85] | -38.5 [-46.5, -30.0] | -39.26 [-44.74, -35.85] |

These are repeated observations of two training runs. At update 100 the
success differences are unresolved; later checkpoints show increasing deficits,
especially in exact completion. They are not independent replications.

E0-to-E1 specialization is separate: native success rises by 33 pp
[26.0, 40.5], with 71 gains and five regressions; exact success rises by
45.5 pp [37.5, 53.0], with 98 gains and seven regressions. Median token change
on the 120 joint native successes is -45.42% [-51.92%, -41.70%]. These changes
cannot be attributed to E2 shaping.

E0 ends 186 episodes with `env_done` and 14 at the generation cap. E1-100
and E1-200 end all 200 with `env_done`; E1-300 has 199 `env_done` and one
generation-cap stop. E2-100 has 199 `env_done` and one generation-cap stop;
E2-200 has 200 `env_done`; E2-300 has 195 `env_done` and five `no_tool_call`
stops. There are no `max_turns` stops. All five final E2 no-tool stops are
native-success regressions. The final transition counts are 194 env-done to
env-done, five env-done to no-tool, and one generation-cap to env-done.

## Native checker and task-grounded behavior

The frozen task awards a positive native result when more checkbox states are
correct than incorrect, counting correctly unselected irrelevant boxes too.
Exact completion requires every requested synonym group selected, no irrelevant
box selected, and an executed Submit. An exact checkbox state without Submit
is unfinished. This gap predates E2: E0 already has 77 incomplete native
successes and final E1 has 52.

The paired E2-versus-E1 comparison adds 69 incomplete-native-success cases and
resolves 21, yielding the +24 pp change above. They represent 52/191=27.23%
of E1's native successes versus 100/162=61.73% of E2's. This is an observed
shift toward incomplete native success in this pair. It does not establish
intentional checker exploitation or isolate shaping from the host change.

| Final submission diagnostic | E1 | E2 |
|---|---:|---:|
| Attributed Submit episodes | 199/200 | 195/200 |
| Submitted episodes with omissions | 58/199 (29.15%) | 101/195 (51.79%) |
| Missing required boxes | 71/620 (11.45%) | 156/601 (25.96%) |
| Submitted episodes with extra selections | 15/199 (7.54%) | 73/195 (37.44%) |
| Selected irrelevant boxes | 15/406 (3.69%) | 83/401 (20.70%) |
| Incomplete states at model decisions | 200/205 | 210/213 |
| Complete states at model decisions | 5/205 | 3/213 |
| Direct Submit from incomplete state | 1/200 incomplete decisions | 10/210 incomplete decisions |
| Direct Submit from complete state | 0/5 complete decisions | 2/3 complete decisions |
| Submit after earlier precommitted actions | 198/199 submissions | 183/195 submissions |
| Unknown terminal-action attribution | 0/199 terminals | 0/195 terminals |

Required/irrelevant denominators count checkbox opportunities at attributed
submissions. Unfinished episodes are excluded from these conditional rates,
which explains the differing exposure. Omissions and extras can coexist.
Decision denominators count policy-visited states, including decisions producing
no tool call; they are not independent trials. The five complete E1 decisions
all come from one unfinished episode. Checkbox events within one episode are
correlated; no independent-binomial precision claim is made for those rates.

Direct Submit means Submit is the first executed action in that assistant
turn, chosen from the displayed state. E2 has four such incomplete submissions
in the initial empty state and six after earlier turns; eight receive native
success and two native failure. E1 has one initial direct incomplete submission,
which receives native success. Per 200 paired episodes, E2 introduces ten and
removes one: +4.5 pp [1.5, 8.0].

All 198 E1 and 183 E2 precommitted submissions were selected while the
pre-batch state was incomplete. Later checkbox feedback could not inform
actions already emitted in that batch. These counts therefore do not establish
ignored feedback. E1 batches actions in 200 episodes/200 turns; E2 in 198
episodes/198 turns. Action counts and preceding-action depth do not measure
deliberation or verification quality here.

All 1,400 saved evaluation trajectories were checked mechanically against the
frozen synonym table, native scoring rule, selected states and executed actions;
all terminal targets are attributed. Final state summaries were inspected for
all 36 native regressions and all 88 exact regressions, with action traces
examined for the examples and controls below. The 36
native regressions contain 31 wrong submissions and five no-tool stops.
Among the 88 exact regressions, 36 submitted with omissions alone, 23 with
extras alone, 25 with both, and four stopped without Submit (one already had
the exact checkbox set). The fifth native no-tool regression was not an E1
exact success.

Six joint exact-success controls were selected with
`sorted(random.Random(17).sample(joint_exact_indices, 6))`, where
`joint_exact_indices` is the ascending list of indices 0 through 199 whose
`exact_goal_success` is true in both final frozen `checkbox_audit.json` files.
The population contains 51 indices; the selected indices are
53, 80, 83, 96, 109 and 143. Both arms select precisely the required
sets and submit in one batch in all six; E1/E2 token counts are respectively
314/233, 797/255, 483/426, 368/220, 377/229 and 467/323. This control review
is unblinded and was conducted after aggregate results were known. It is a
diagnostic sample, not an independent masked validation or a prospective
stratified estimate. Full state/action evidence is in `behavior-review.json`.

The inherited seed-17 stratified review was also reproduced over all seven
evaluations, with its original precedence: budget stops, voluntary stops,
invalid/repeated actions, then efficient native successes at or below the
pooled native-success token quartile (396 tokens). It selected six of 16 budget
cases, all five voluntary stops, six of 47 invalid/repeated cases and six of
303 efficient-native-success cases. Only seed and index were removed from the
23 episode records before labeling; arm and checkpoint were not episode fields.
The case-to-arm/checkpoint mapping was kept in a separate key, opened afterward.
Episode and inference timing fields remained visible and could reveal the host
and therefore the arm, given the large L4/4090 timing difference. Earlier
aggregate results and final regression summaries were already known; these
timing fields, recognizable cases and outcomes prevent claiming independent
blinding.

All six budget cases remain unfinished. All five voluntary cases lack Submit,
including one with an otherwise exact set. All six invalid/repeated cases
submit incomplete sets; five still receive native success and one fails.
Five efficient controls complete the exact goal; the sixth omits one required
box but passes natively. This validates the mechanical distinction on the
selected cases without estimating population label accuracy or causal intent.
The preparation script, cases, selection metadata, labels and key are retained
as `prepare_masked_review.py` and `masked_*.json`/`masked-review-labels.json`.

Illustrative zero-based indices (seed = 4232100000 + index):

| Index | Observed E1 to E2 difference | Interpretation limit |
|---|---|---|
| 31 | E1 selects joyful plus humorous; E2 submits empty, omitting joyful. Both receive native success; 382 to 182 tokens. | Neither is exact; a checker-gap example, not a new exact regression. |
| 41 | Both omit delectable while selecting scared and dumb. E1 submits in the first batch; E2 submits directly after observing that incomplete state on a later turn. | Supports the decision-boundary label, without inferring intent. |
| 88 | E1 selects petite and furious; E2 selects furious only and submits in the same batch. Both receive native success; 397 to 181 tokens. | Exact regression through omission, not demonstrated feedback disregard. |
| 65 | Both select all five required boxes. E1 submits; E2 ends with no tool call. | Exact state is insufficient when the goal requires Submit. |
| 1 | E1 is exact; E2 adds scared, courageous and home to the required response box. | Native failure through extra selections, not just omitted work. |

## Generic event panel

The four bounds below are one-sided exact upper bounds at alpha=0.05/4 for
new events per 200 paired questions, compared with the inherited 5 pp margin.
They control these four bounds jointly, not all secondary analyses in this report.

| Event | E1 episodes | E2 episodes | New / resolved | Upper bound on new-event probability |
|---|---:|---:|---:|---:|
| Wrong native submission | 8 | 33 | 31 / 6 | 22.12% |
| Voluntary no-tool stop | 0 | 5 | 5 / 0 | 6.26% |
| Any invalid action | 12 | 0 | 0 / 12 | 2.17% |
| Any repeated action | 1 | 0 | 0 / 1 | 2.17% |

The wrong-submission and no-tool preservation bounds fail; the latter alone
does not establish an increase above 5 pp. Invalid/repeated-event bounds pass.
There are 18 invalid action events and five repeated action events in E1,
versus zero of each in E2. Mechanical reductions in those events do not rescue
the observed success and exact-completion losses. The behavior distribution
is shifted in this observed pair; causal mechanism remains indeterminate.

E0-frozen assistant-token thresholds are P10=572.5 and P75=1632.75. The
short-success flag rises from 58.64% to 95.68%; the long-success flag falls
from 1.05% to 0%. These flags describe length against one ruler, not harmful
underthinking or proof of reasoning quality. The exact-success control pairs
demonstrate legitimate shorter completions too.

## Integrity, recipe and execution boundary

The frozen E1/E2 configurations differ only in ID, description, enabling
relative successful-length cost, and the OpenEnv installation path. The home
manifest matches all 42 original training/evaluation source hashes. Python,
package pins, OpenEnv revision and episode-boundary stamps match. Native
migration controls reproduced all 16 original records. Source and receipt
hashes are recorded in `analysis.json`.

Both trained policies start fresh from pinned Qwen3-1.7B, seed 4232, bf16
LoRA rank 16/alpha 32, 300 updates, batch four, eight rollouts, DAPO,
`naive_sum`, `scale_rewards: none`. E2 alone enables relative successful-length
cost at 0.1. Training seeds are 4232000000-4232000499; all seven evaluations
use 4232100000-4232100199. Prompt/trajectory/context settings remain
4096/4096/8192 with eight turns. No warm start or protocol change occurred.

These are the accepted batch/continuation semantics. The trajectory allowance
includes feedback when deciding whether generation can continue; it does not
interrupt an already emitted action batch between calls or certify a hard cap
on all retained terminal feedback. Assistant tokens include reasoning, content
and serialized tool-call arguments; they exclude tool feedback. Native results
remain primary. This review did not retokenize all feedback for a new strict
per-action or saved-prefix sensitivity; it makes no claim that such a different
boundary would reproduce these numbers.

All 300 update records exist in each trained arm. All 416 recorded rollout
rewards per arm were recomputed and agree with the declared arithmetic and
seed/token constraints. Mean active prompt-group fraction is 27.67% in E1
versus 99.58% in E2; 115 E1 updates have zero gradient versus none in E2.
In the recorded last ten updates, native training success is 300/320 versus
264/320. Task reward varies in 9/40 versus 22/40 groups; composed reward
varies in 9/40 versus 40/40. Gradient activation is part of the intervention;
this comparison does not separate it from length-cost assignment.

## Host confound and costs

| Condition | Hardware and OS | Kernel / libc | NVIDIA driver |
|---|---|---|---|
| E0/E1 | gpu-l4, L4, Ubuntu | 6.8.0-146-generic / glibc 2.39 | 580.178.04 |
| E2 | omarchy-home, RTX 4090, Arch/Omarchy | 7.2.5-3-omarchy / glibc 2.44 | 610.57.04 |

Python 3.12.3 and the pinned Python packages are identical, but that does not
remove the hardware/kernel/libc/driver confound. The home desktop shares GPU
resources with graphics and voxtype. Earlier L4 checkbox work shared host CPU,
RAM and disk with the concurrent tab campaign until the requested checkbox
stop. Timing is therefore descriptive and cannot establish a shaping speedup.
Hardware may also change numerical training trajectories; the confound applies
to the outcome comparison, not only its speed.

| Ledger item | Training allocation-hours | Evaluation allocation-hours | Total |
|---|---:|---:|---:|
| E0, L4 | 0 | 1.9680 | 1.9680 |
| E1, L4 | 24.1348 | 7.6236 | 31.7583 |
| E2, RTX 4090 | 13.2307 | 1.8752 | 15.1058 |
| Completed trio | 37.3654 | 11.4668 | 48.8322 |
| Discarded original E2, L4 | 3.3742 | 0 | 3.3742 |
| Recorded study expenditure including discarded E2 | 40.7396 | 11.4668 | 52.2064 |

Every attempt has matched start/end ledger entries, counted once. The discarded
E2 was intentionally interrupted at 31/300 without a checkpoint and never
reused. Evaluation totals include checkpoints 100/200/300. Allocation-hours
across L4 and 4090 are not equivalent compute, energy or monetary cost.
Qualification, home setup/download and small unrecorded CLI overheads are
outside these phase totals. Both trained-arm costs are study expenditure;
their sum is not the incremental cost of choosing one policy for deployment.

Final all-episode mean latency is 31.45 seconds for E1 and 8.04 for E2;
mean policy-call time is 26.19 and 4.24 seconds. Failures are included in these
deployment averages. No same-hardware E2-versus-E1 latency saving or finite
shaping break-even estimate is identifiable from these cross-host timings.
Token savings on joint successes remain separately reported above.

For the separate E0-to-E1 specialization comparison on L4, all-episode mean
policy-call time falls from 29.99 to 26.19 seconds. Its descriptive training
amortization estimate is 86,885.11 training seconds / 3.80161 seconds saved
per episode = about 22,855 episodes. This arithmetic excludes research
evaluation expense, assumes the measured question mix and resource sharing
persist, and is not a controlled latency benchmark or an E2 shaping result.

## Reproduction and disposition

The report follows `docs/plans/click-checkboxes-soft-family-extension.md` and
the authorized home continuation. Analysis artifacts are in
`click-checkboxes-soft-home-s4232-ops/final-analysis-20261017/`:

- `analyze.py` and `analysis.json`: deterministic paired statistics, reward
  checks, receipt hashes, checkpoint metrics and costs.
- `behavior-review.json`: paired state/action evidence and the disclosed
  joint-exact control sample.
- `comparison.png` and `comparison.pdf`: checkpoint and fixed-endpoint figure.
- `validation.json`: analysis rerun, CPU checks and review disposition.

Independent counterpart review with Claude Opus 5.5 at xhigh completed on
2026-10-18 after authentication was restored. It identified three reporting
issues: two joint-success timing pairs lacked a retained calculation, the
control-sampling expression was unspecified, and visible timing fields could
reveal the arm in the supplemental masked sample. The timing pairs were removed,
the exact sampling expression was verified and recorded, and masking limits
were corrected. A follow-up review confirmed that no required fix remained.
Primary results and frozen artifacts are unchanged. The full verdict,
disposition and follow-up are retained as `counterpart-review-20261018-native.json`,
`counterpart-disposition-20261018.json` and
`counterpart-corrections-review-20261018.txt`. The earlier authentication failure
remains in `counterpart-review.txt`; `validation-counterpart-20261018.json`
records the final review status separately from the original `validation.json`.

Run from the repository root with the existing analysis Python environment:
`python3 pipeline/runs/click-checkboxes-soft-home-s4232-ops/final-analysis-20261017/analyze.py`.
The script checks the frozen audit, paired calculations, empty-success handling,
regression bounds and input hashes before writing; it refuses to replace a
different analysis output. The current CPU suite passed 561 tests with seven
skips; this supplements the frozen deployment's recorded 515-test pass and
does not change or revalidate its GPU implementation by itself.

The authorized checkbox comparison is complete with a negative quality result.
No further run, dose, replication, stricter boundary, or E3 is launched.
Close its task only after reporting these findings. Tab monitoring remains
independent; at this check its SSH host was unavailable and its last verified
snapshot was preserved. Live state belongs in root `RUNNING.md`.
