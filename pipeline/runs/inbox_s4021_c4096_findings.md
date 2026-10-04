# Corrected mixed-inbox E0/E1/E2: initial comparison

Reviewed: 2026-10-05 (Europe/Berlin). Status: **initial comparison reviewed;
decision required**. All three authorized arms are complete. No additional run
is admitted by this review.

## Result

At the declared update-300 endpoint, E2 improves held-out success from E1's
183/200 to 199/200 and reduces paired successful-response tokens by a median
91.96%. Both declared success-preservation and compression criteria pass.
Tool behavior changes substantially: every final E2 episode contains an invalid
action, and 66 contain an exact repeated action. The declared off-target margin
fails. The result supports successful compression with a behavioral shift;
it does not establish unchanged behavior or a causal mechanism for the errors.

The protocol is `docs/plans/inbox-family-extension.md`, carrying forward the
fixed analysis in `docs/plans/e0-e2-campaign.md`. These runs use seed 4021,
physical GPU 1, and fresh `-c4096` IDs. All arms and checkpoints retain 4,096
trajectory tokens including feedback, 4,096 prompt tokens, 8,192 context tokens
and eight turns. E1/E2 start independently from the original base. Their frozen
configs differ only in metadata and enabling relative successful-length cost
at weight 0.1. The earlier 5,120-token attempt is separate and supplies no baseline.

## All declared evaluations

Each row uses the same 200 held-out questions and byte-identical initial
observations, seeds 4021100000-4021100199. Update 300 is primary; the earlier
checkpoints describe the trajectory and were not selected as endpoints.

| Policy | Correct / 200 | Success, Wilson 95% CI | Mean assistant tokens on correct episodes |
|---|---:|---:|---:|
| E0 | 103 | 51.5% [44.61%, 58.33%] | 1338.03 |
| E1 update 100 | 128 | 64.0% [57.14%, 70.33%] | 1554.57 |
| E2 update 100 | 154 | 77.0% [70.69%, 82.29%] | 980.29 |
| E1 update 200 | 170 | 85.0% [79.39%, 89.29%] | 1329.12 |
| E2 update 200 | 150 | 75.0% [68.57%, 80.49%] | 260.28 |
| E1 update 300 | 183 | 91.5% [86.81%, 94.63%] | 1475.91 |
| E2 update 300 | 199 | 99.5% [97.22%, 99.91%] | 121.75 |

The means condition on different sets of solved questions. The paired token
comparison below is the efficiency endpoint.

| Operation | Questions | E0 | E1 100 | E2 100 | E1 200 | E2 200 | E1 300 | E2 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reply | 54 | 3 | 13 | 25 | 40 | 26 | 51 | 53 |
| Forward | 60 | 19 | 40 | 46 | 46 | 42 | 47 | 60 |
| Delete | 41 | 40 | 37 | 39 | 41 | 40 | 41 | 41 |
| Important | 45 | 41 | 38 | 44 | 43 | 42 | 44 | 45 |

The fixed sample does not support four independent operation-level equivalence
claims. Within-seed episode intervals do not measure training-seed uncertainty.

## Primary paired comparison: E2 versus E1 at update 300

- Success: 17 gains and one regression, a net gain of 8 percentage points;
  exact McNemar p=0.00014496.
- The one-sided exact 95% upper bound on an E1-correct/E2-wrong question is
  2.350%, below the declared 5-percentage-point margin. Gains do not offset
  losses in this criterion.
- Both correct: 182 questions. Median paired token change is -1326 tokens
  [95% bootstrap CI -1410, -1230.5].
- Median paired percentage change is -91.962% [95% bootstrap CI -92.364%,
  -91.575%], using 20,000 resamples and RNG seed 0. Its upper bound passes
  the declared -10% compression target.
- The sole regression is episode 91: E2 replies to Iolande with `Feugiat`
  instead of the required `Feugiat.`. The recipient is correct; the period is
  missing. The same question fails for E2 at update 100, succeeds at 200,
  and fails again at 300.

E0 versus final E1 remains a separate task-training contrast: 86 gains, six
regressions and 97 jointly correct questions. Median paired token change is
-0.81% [-8.65%, +18.27%]. Task training alone does not establish compression
relative to E0. Final E2 has 96 gains and no regressions versus E0.

## The intermediate trajectory matters

At update 100, E2 has 45 gains and 19 regressions versus matched E1, with median
paired token change -60.79% [-68.14%, -37.96%]. At update 200 it has 18 gains and
38 regressions, with token change -81.99% [-82.76%, -81.07%]. Its success falls
to 75%, below E1's 85%, while invalid-action episodes reach 94%.

Between E2 updates 100 and 200 there are 27 gains and 31 regressions. Reviewed
failures include repeated invalid fills, invented tool names, wrong recipients,
overwritten forward bodies and a reply used instead of forwarding. Several
traces exhaust the trajectory budget through tool feedback despite short
assistant output. At update 300, E2 gains 50 successes and loses one versus its
update-200 checkpoint. These repeated observations show a temporary degradation
followed by recovery; they are not independent replications or grounds to replace
the declared final endpoint.

Inbox exposes more of this training trajectory than the retained table run:
E1 reaches 64% at update 100 here versus 91.5% for read-table-2. However, final
E2 again approaches the task ceiling. This family has not established durable
complexity. Different seeds, prompts and observation formats prevent a causal
family-only comparison. The retained table results remain valid first evidence.

## Tool behavior changes despite high final success

| Event, counted per episode | E1 final | E2 final | Newly introduced | One-sided 98.75% upper bound on new-event probability |
|---|---:|---:|---:|---:|
| Incorrect terminal action | 12/200 | 1/200 | 1/200 | 3.148% |
| Voluntary no-tool stop | 2/200 | 0/200 | 0/200 | 2.167% |
| Any invalid action | 10/200 | 200/200 | 190/200 | 97.835% |
| Any exact repeated action | 1/200 | 66/200 | 66/200 | 40.970% |

These four bounds use alpha=0.05/4 as declared. Invalid and repeated actions
clearly fail the 5-percentage-point margin. All final E2 episodes terminate,
compared with five nonterminal E1 episodes. Mean dispatched actions increase
from 3.15 to 4.18 per episode.

Every final E2 episode starts with a valid email-opening click followed by an
invalid action. In 197/200, the first turn contains exactly these two calls;
three contain additional calls. The model then generally uses the valid detail
controls. All 206 action errors retain the same visible page as before the
error. All 66 exact repetitions are failed clicks on now-hidden elements: 57
forwarding episodes and nine deletion episodes. Final correct forwards retain
the requested source email, exact recipient name and original body.

For example, episode 0 opens Floris's message, attempts the now-hidden list
star, receives an error, then correctly clicks the detail star. Episode 5 opens
Rene's email, attempts to fill the now-hidden sender element, then opens the
reply form, enters the exact text and sends successfully.

These are observable action errors and recovery, not evidence that the native
success scorer was bypassed. They do not prove the model intentionally seeks
errors or that errors cause compression. The one-knob intervention, one seed
and saved trajectories cannot separate those mechanisms. The final contrast
does establish that token efficiency and clean tool use can move in different
directions. Real-world harmlessness or transfer to a changed interface remains
unmeasured.

The frozen E0 short-response flag marks 100% of correct final E2 episodes as
"underthinking". That flag is not a task-grounded harm label. Likewise,
verification depth counts preceding actions, including unsuccessful ones; its
increase cannot establish more meaningful verification.

An exploratory Bonferroni sensitivity across the two primary requirements and
four off-target endpoints leaves the interpretation unchanged: success-loss
upper bound 3.374%, compression interval [-92.415%, -91.478%], and failed
invalid/repeated-action margins. This does not add training-seed replication.

## Integrity and trace review

All 25 raw E2 files, including both intermediate checkpoint evaluations, match
remote SHA256 values. The 41 source/lock hashes, stack, clean pinned OpenEnv and
MiniWoB clones, frozen configs, adapter provenance and immutable E0/E1 review
inputs pass. Final and update-300 adapter hashes agree. The process exited zero;
there is no logged fatal traceback or OOM.

All 600 E2 evaluation trajectories match their reports, seeds and E0 initial
observations. Metrics recompute with E0's immutable thresholds. Tokenizer
recounts verify trajectory feedback charges and stops. The native inbox state
auditor validates terminal sender/recipient/content/operation conditions and
rejects corrupted success labels. Every successful evaluated and captured
training forward uses the requested source email.

All 300 optimizer logs are finite. All 416 captured training records have
matching token recounts, seed/prompt/slot alignment with E1, and independently
reconstructed reward arithmetic. Of these, 405 retain complete saved-state
traces. Eleven failed nonterminal captures omit final feedback through the
existing pinned oversized-feedback rollback. Their retained prefixes and
reward/token counts pass; discarded final states cannot be independently
inspected. This is the same limitation documented in E1, where 16 captures
were affected. It does not affect the complete evaluation traces.

Review covered all 68 distinct E2 regression traces from E0-to-E2 checkpoints,
matched E1-to-E2 checkpoints and consecutive E2 checkpoints, including the sole
final failure. A seed-17 stratified sample contains six cases each for budget
endings, voluntary stops, invalid/repeated actions and efficient correct
behavior. Its arm/checkpoint/index identifiers were masked until 24 judgments
were recorded; outcomes and aggregates were already known, and one previously
seen case was recognized. This was not an independent blinded review.

The masked sample's efficient stratum excludes error-containing episodes, so it
cannot represent final E2. Supplemental review therefore uses the shortest
correct trace per operation at each E2 checkpoint, ties by episode index
(12 controls). All preserve the requested successful outcome; most include
error recovery. The selection rule and judgments are retained with the run.

## Training signal and costs

E1 has varying task reward in 49.67% of groups and at least one active group in
90.33% of updates; 29 updates have zero gradient. E2 has varying composed reward
in 92.17% of groups, every update has an active group, and none has zero gradient.
The final ten E2 updates have 99.375% sampled task success, task-success variation
in only 2/40 groups, and varying successful lengths in 26/40. E1's corresponding
counts are 92.8125%, 15/40 and 40/40. Thus shaping changes gradient availability;
this comparison does not isolate cost assignment from that effect.

| Arm | Training GPU-hours | Evaluation GPU-hours | Total GPU-hours |
|---|---:|---:|---:|
| E0 | 0 | 2.715 | 2.715 |
| E1 | 38.235 | 11.759 | 49.994 |
| E2 | 33.026 | 4.690 | 37.717 |
| Total | 71.261 | 19.164 | 90.425 |

All nine phase attempts have complete timing and allocation records. These
totals cover only the corrected trio, not qualification or superseded attempts.
At final evaluation, mean inference time falls from E1's 64.81 to E2's 5.84
seconds per episode; total episode time falls from 69.22 to 11.73 seconds,
including tool execution. Paired inference savings average 58.98 seconds
[bootstrap 95% CI 55.90, 62.12]. Timing applies to this stack and hardware.

Charging all E2 training to deployment, its inference-only break-even against
E1 is approximately 2,016 episodes [conditional interval 1,914, 2,127]. This
omits evaluation costs and does not price real deployment tools. E2's observed
training also costs 5.209 fewer GPU-hours than E1 in this pair.

## Evidence and decision boundary

The run directory `e2-email-inbox-noscroll-s4021-c4096/` retains the technical
audit, initial paired comparison, behavior audit, regression cases/judgments,
masked cases/key/judgments, efficient controls and final hash-bound `review.json`.
The reproducible offline helpers and remote evidence are in
`inbox-campaign-s4021-c4096-ops/`. Prior receipts and results remain unchanged.

This completes the authorized initial comparison. The watcher is retired and
the campaign stops for the user's decision. Further weights, placebo, seeds,
families or E3 require new explicit approval.
