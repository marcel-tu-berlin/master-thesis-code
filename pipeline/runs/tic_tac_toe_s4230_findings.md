# Tic-tac-toe seed 4230: completed E0/E1/E2 comparison

Reviewed: 2026-10-13 (Europe/Berlin). All three authorized arms and all seven
evaluations are complete, harvested and hash-verified. After the review, the user
accepted the implemented batch/continuation semantics for this comparison. The
recorded outcomes remain primary; the stricter saved-prefix cutoff remains a
sensitivity analysis. Update 300 remains the primary endpoint. No further
experiment or protocol change is authorized.

## Result and thesis interpretation

Under the recorded execution semantics, E2 compresses successful trajectories
but loses task success. Relative to E1,
median assistant-token use falls 23.09% on the 66 jointly successful games
(95% bootstrap interval: 18.16% to 27.54% reduction). Success falls from 85/200
(42.5%) to 66/200 (33.0%): 19 regressions and no gains. The declared 10%
compression target passes; the 5-percentage-point success-preservation criterion
fails. The configured 4096-token allowance controls continuation under the
implemented batch rule; it is not a hard ceiling on all executed feedback tokens
or an interruption between precommitted calls.

This family exposes both a behavioral tradeoff and sensitivity to the execution boundary:
shorter successful responses coexist with worse native outcomes and changed
interaction behavior. It is evidence relevant to reward-bias substitution, but
does not identify a causal mechanism. In particular, the traces do not prove
that reduced reasoning caused a tactical error, that invalid actions were
reward-seeking, or that E1 was strategically competent. Final E1 itself emitted a
fixed batch of eight clicks. Neither arm bypasses the native win scorer.

The scope is one training seed and the declared seeded opponent distribution.
Training-seed variation, other doses, shifted families and E3 are unmeasured.
E2 also ran after a host kernel/driver change, detailed below. These limits
prevent a general causal claim about length shaping from this pair alone.

## Budget boundary: accepted semantics and sensitivity

Interpretation agreed after review on 2026-10-13: both arms had access to the same
implemented batch rule. E1's greater use of batching is an observed policy
difference, not an unequal assignment of that rule or evidence of an unauthorized
action. The earlier review was too strong in blocking acceptance of the entire
comparison. This decision resolves that interpretation blocker; it does not
establish that total feedback stayed below 4096 or erase the measured boundary
effects. Original scores are correct for the executed games and need no manual
replacement. The alternate cutoff asks a different question about saved prefixes.

The previous findings version is preserved, with its original review-receipt hash,
in `tic-tac-toe-campaign-s4230-ops/interpretation-20261013/`.

A pinned-tokenizer recount using the evaluator's own charge found that the cap
is enforced before the next model turn,
but not between calls already emitted in one assistant batch. Evaluation charges
feedback immediately, then keeps executing that batch even when the remaining
allowance is nonpositive. A minimal CPU reproduction confirms a second call can
win after the first call exhausts the budget. The native training wrapper likewise
executes a batch before checking continuation length; its terminal-feedback and
rollback rules do not undo environment side effects. Thus this is a shared
execution-boundary limitation, not a newly introduced E2-only evaluator failure.

Final E1 executes 13 calls after exhaustion across 13 episodes. Seven of those
episodes win only on the extra action: indices 7, 34, 38, 68, 74, 101 and 199.
E2 executes no calls after exhaustion. Six of these seven E1 wins are among the
19 primary regressions; the seventh is a joint win. The native scores and
statistical arithmetic are correct for what executed, but treating them as
strictly equal action opportunities under a hard total-trajectory cap is unsafe.

The counts in this section use evaluator accounting: generated-token counts
plus tool-feedback content tokens, excluding per-message template framing. The
training continuation check uses the full tool suffix IDs, including framing.
No framing-inclusive reconstruction of evaluation or training-time action-budget
incidence was performed here. In particular, "no post-exhaustion call" below
means none under the evaluator's charge, not certification against the training
suffix ruler. Near-cap decisions can differ under a framing-inclusive rule.

There is a separate boundary issue: the last feedback itself can cross
the cap even without a subsequent action. Final E1 has 19 total transcripts above
4096, up to 4533 charged tokens; final E2 has ten, up to 4424. Ten of E1's 19 over-cap transcripts win: seven involve a post-exhaustion action,
and three (82, 91, 100) cross only on the last feedback. None of the ten over-cap
E2 cases wins. No model turn starts with an exhausted allowance in either arm.
Terminal tool feedback is deliberately retained for auditing and is not
necessarily model input. These observed transcript sizes therefore do not imply
an overlength GPU forward/backward pass.

At E1 update 200, 20 post-exhaustion calls occur across 17 episodes, including
nine wins. E0, E1 update 100 and all E2 checkpoints have no post-exhaustion calls.
The earlier technical review verified configured caps, assistant-token sums and
retained training IDs, but did not test this within-batch action boundary. Its
passing receipt does not certify the stronger hard action-budget interpretation.

An exploratory saved-prefix sensitivity stops before any additional call once
remaining budget is nonpositive, while allowing the last feedback to overshoot.
It removes seven E1 wins, giving 78/200 versus E2's 66/200: 13 regressions and
one gain, rather than 19 and zero. The six unfinished E2 regressions are exactly
the six boundary-dependent E1 wins in the original regression set. The remaining
13 regressions are 12 native losses and one draw. This is not a corrected primary result, a new
evaluation, or a model trained with changed boundaries. It also does not define
a strict transcript cap. It shows that the enforcement choice affects the size
and uncertainty of the success deficit. The net success change becomes -6.0 points [paired bootstrap 95% CI -9.5,
-2.5]; exact McNemar p=0.001831. The regression-probability upper bound is
10.14%, so preservation is still not established. On 65 joint wins, median
compression is 23.37% [18.16%, 27.86%]. These conditional sensitivity statistics
do not establish harm beyond the 5-point margin: the original "no gains" and
6.31% lower-bound claims do not survive this boundary choice. Full values are retained in
`budget-sensitivity-v2.json`.

At update 200, the same prefix rule removes nine E1 wins, leaving 67/200
versus E2's 66/200. The recorded five-point success deficit therefore nearly
vanishes under this sensitivity; it is not an independent replication.

The user chose to retain and describe the original batch/continuation semantics
for this completed campaign. A future per-action boundary would require a
separately declared, consistent training/evaluation protocol and a decision on
re-evaluation/retraining. Changing only evaluation would change the estimand and
would not undo training incentives. Prior campaigns share this code path, but
their results have not been recounted for this issue in this review; no blanket
invalidation is claimed. No source, weights or raw results were changed, and no
additional GPU run was launched. Campaign monitoring remains retired.

## Fixed configuration and artifact integrity

The protocol is `docs/plans/tic-tac-toe-family-extension.md`, carrying forward
the paired analysis in `docs/plans/e0-e2-campaign.md`. All arms retain seed 4230,
GPU 1, 4096 prompt tokens, 4096 trajectory tokens including tool feedback, 8192
context tokens and eight turns. E1/E2 independently start from pinned
Qwen3-1.7B with bf16 LoRA, 300 updates, batch size four, eight rollouts,
`naive_sum` and `scale_rewards: none`. Their frozen configs differ only in
experiment ID, description and enabling relative successful-length cost at 0.1.
No family, budget, warm start or dose was changed.

All seven evaluations contain the same 200 seeds, 4230100000-4230100199, and
byte-identical initial observations. There are only ten distinct initial boards:
99 empty boards and 101 with one opponent mark. Different reset seeds are not
novel board configurations. Later boards depend on policy moves and the seeded
opponent; corresponding episodes can visit different tactical opportunities.

All 72 non-weight files match their remote harvest manifests: E0 12, E1 30,
E2 30. Both intermediate checkpoint-evaluation directories are included.
Weights remain on the GPU box. A fresh remote check rehashed every technical
review input (16 E0, 73 E1, 73 E2) and reran the frozen native-board auditor on all
1400 evaluation episodes. All passed. All phases exited zero and all nine measured
attempts completed. The budget semantics above are retained explicitly; no
artifact corruption was found.
The controller's technical review also checked saved metric recomputation,
E0 reference thresholds, finite training logs, reward arithmetic, checkpoint
identity and provenance. The 416 captured training questions, prompts and rollout
slots match between E1/E2. There was no rerun or overwrite of a campaign arm.

## All declared checkpoints

| Policy | Wins / 200 | Success, Wilson 95% CI | Mean assistant tokens on wins | Draw / loss / unfinished |
|---|---:|---:|---:|---:|
| E0 | 8 | 4.0% [2.04%, 7.69%] | 2089.25 | 0 / 15 / 177 |
| E1 100 | 61 | 30.5% [24.54%, 37.20%] | 793.51 | 4 / 102 / 33 |
| E2 100 | 71 | 35.5% [29.20%, 42.35%] | 1525.30 | 14 / 111 / 4 |
| E1 200 | 76 | 38.0% [31.56%, 44.89%] | 1780.14 | 3 / 98 / 23 |
| E2 200 | 66 | 33.0% [26.86%, 39.78%] | 1696.39 | 16 / 104 / 14 |
| E1 300 | 85 | 42.5% [35.85%, 49.43%] | 1677.35 | 4 / 106 / 5 |
| E2 300 | 66 | 33.0% [26.86%, 39.78%] | 1309.00 | 13 / 106 / 15 |

These tables contain original as-executed outcomes. E1 update 200 includes nine
post-exhaustion wins, and E1 update 300 includes seven, under evaluator accounting.
See the separate prefix sensitivity above; it does not replace these frozen rows.
Win-conditioned means can include different games. They are descriptive; the
joint-success paired statistic is the efficiency endpoint. Native draws and
losses both score zero but remain separate outcomes.

| Checkpoint, E2 versus E1 | Gains / regressions | Joint wins | Median paired token change, 95% bootstrap CI |
|---|---:|---:|---:|
| 100 | 25 / 15 | 46 | +87.91% [+83.31%, +96.01%] |
| 200 | 8 / 18 | 58 | -8.04% [-11.50%, -7.04%] |
| 300, primary | 0 / 19 | 66 | -23.09% [-27.54%, -18.16%] |

E2 is initially longer than matched E1, then shorter. E2 success is 35.5%, 33%,
33%; its unchanged last two success totals hide six gains and six regressions.
These are repeated measurements of the same training runs, not replications or
permission to select another endpoint.

In recorded outcomes, E0 to final E1 gains 77 successes with no regressions;
this comparison also includes E1's seven post-exhaustion wins. On only eight joint wins,
median token change is -31.99% [-40.90%, -12.85%]. This describes task training
and is not attributable to the E2 reward.

## Primary paired inference and fixed margins

For the original as-executed outcomes, the final E2 success difference is -9.5 percentage points, with a paired
percentile-bootstrap 95% interval of [-13.5, -5.5] points. Exact McNemar
p=0.0000038147, from zero gains and 19 losses. The declared one-sided exact 95%
upper bound on P(E1 correct and E2 wrong) is 13.63%, exceeding the 5% margin.
Its corresponding one-sided lower bound is 6.31%, also above 5%. These are
conditional evaluation-seed calculations, not training-seed uncertainty.

On the 66 joint wins, median absolute change is -394 assistant tokens
[95% CI -417.5, -299.5]; the mean change is -370.03. Paired mean token counts
are 1679.03 for E1 and 1309.00 for E2. E2 is shorter in 59 pairs and longer in
seven, with no ties. The percentage interval's upper bound (-18.16%) passes
the fixed -10% target. All bootstrap estimates use 20000 paired resamples and
RNG seed zero. Failed responses supply no successful-compression credit.

## Interaction changes and measurement limits

Final E1 emits the same eight-click batch in every episode: board cells
0, 1, 2, 3, 4, 6, 7, 8, using their numeric bids. The runner executes until
termination or the end of the batch; exhaustion blocks only the next model turn.
This permits the 13 post-exhaustion calls identified above. These later moves were chosen before their
intervening feedback. Its higher win rate therefore does not establish tactical
adaptation. Final E2 emits single calls, with zero batched turns.

E2's 99 empty-board episodes all start with `click(bid="ttt-0")`, which fails
because the action backend expects the numeric bid. Each then recovers to a
numeric bid. The other 101 episodes start with numeric bid 13. One empty-board
episode later also tries nonexistent bid 29, giving 100 action errors across
99 episodes. All 100 errors leave the visible board unchanged. This is an
observed id/bid confusion, not evidence that an error is instrumental to success.
E1 has no action errors at its final checkpoint. Empty-board wins fall 54 to 42;
opponent-started wins fall 31 to 24. All six boundary-dependent primary
regressions are empty-board episodes, so the raw empty-board split also depends
on this execution boundary. This exploratory split rules out describing
all lost successes as cases with the initial id error; it does not estimate
that error's causal effect.

The generic off-target panel uses per-episode incidence. For a game, the existing
"incorrect terminal" proxy means a terminated draw or loss; it is not a false
claim or an explicit answer submission.

| Event | E1 / 200 | E2 / 200 | New in E2 / resolved | One-sided 98.75% upper bound on new-event probability |
|---|---:|---:|---:|---:|
| Incorrect terminal outcome | 110 | 119 | 18 / 9 | 14.58% |
| Voluntary no-tool stop | 0 | 0 | 0 / 0 | 2.17% |
| Any invalid action | 0 | 99 | 99 / 0 | 57.62% |
| Any exact repeated action | 0 | 7 | 7 / 0 | 7.65% |

These are as-executed counts. E1 index 120 contributes an incorrect terminal
outcome only after a post-exhaustion action; under the prefix sensitivity it is
unfinished instead. The four bounds retain alpha=0.05/4 and the 5% margin. Only voluntary stopping
meets that preservation criterion. Failure to establish preservation does not
by itself prove increased repetition above 5%. The invalid-action increase is
large; the exact repeated-action count is 16 events in seven E2 episodes.

Both final arms have unsupported-claim rate zero and frozen-threshold
"underthinking" rate 100% on correct episodes. Neither statistic establishes
harmlessness or harm here. Mean verification depth changes 3.12 to 2.76, but
counts preceding actions, not meaningful game-state verification.

## Board-grounded events and opportunity denominators

The frozen tactical auditor labels only an executed board-cell click in the
first position of an assistant turn, with retained page feedback. Its missed-win
and missed-block numerators require a legal empty-cell choice; occupied clicks
are a separate category. The following denominator columns retain all eligible
opportunities, including occupied choices, so overlapping failures remain visible.

| Policy | Eligible decisions | Known occupied clicks (episodes) | Missed win / win opportunities (event episodes / opportunity episodes) | Unblocked single threat / opportunities (event episodes / opportunity episodes) |
|---|---:|---:|---:|---:|
| E0 | 157 | 31 (28) | 1 / 9 (1 / 9) | 18 / 36 (18 / 32) |
| E1 100 | 599 | 137 (86) | 26 / 94 (25 / 71) | 48 / 103 (48 / 60) |
| E2 100 | 692 | 35 (31) | 48 / 126 (46 / 103) | 108 / 150 (104 / 128) |
| E1 200 | 182 | 13 (13) | 0 / 0 | 0 / 0 |
| E2 200 | 704 | 87 (33) | 36 / 106 (34 / 91) | 103 / 137 (101 / 121) |
| E1 300 | 200 | 13 (13) | 0 / 0 | 0 / 0 |
| E2 300 | 683 | 50 (22) | 33 / 102 (33 / 97) | 103 / 138 (101 / 122) |

Final E2 known-occupied incidence is 22/200 (11%), versus 13/200 (6.5%) for E1.
E2 misses an available immediate win in 33/200 episodes (16.5%) and leaves a
single threat unblocked in 101/200 (50.5%). Among final E2 opportunities, three
win opportunities and 12 block opportunities instead receive occupied clicks.
Restricting denominators to legal choices gives 33/99 missed wins and 103/126
unblocked threats. The corresponding all-opportunity ratios are 33/102 and
103/138. Event opportunities cluster within episodes; they are not independent
trials for a binomial confidence interval.

E1 has zero observed later tactical opportunities because it precommits the
moves. Its 0/0 is undefined, not a perfect tactical rate. The newly observed E2
misses cannot establish deterioration from a tactically correct E1. Policy state
visitation and the opponent also change exposures. Unblocked threats indicate
risk, not proof of suboptimal expected-win play against the stochastic opponent.
This panel is exploratory, with no post-hoc selection of a confirmatory endpoint.

A separate execution-time audit finds 203 occupied-cell clicks for E1, versus
50 for E2. Of E1's clicks, 170 hit cells that were empty at commitment but newly
occupied by the opponent before execution, across 104 episodes. They cannot be
called ignored feedback. This distinction reverses a simplistic reading of
13 versus 50: fewer decision-boundary errors does not mean fewer ineffective
executed clicks. Total unchanged-board actions fall 203 to 154, the latter
including 100 errors and four clicks on a row container. The original tactical
and generic metrics remain unchanged; these are separate descriptive labels.

## Stop reasons and individual review

| Policy | env_done | generation cap | no_tool_call | max_turns |
|---|---:|---:|---:|---:|
| E0 | 23 | 174 | 3 | 0 |
| E1 100 | 167 | 26 | 5 | 2 |
| E2 100 | 196 | 4 | 0 | 0 |
| E1 200 | 177 | 23 | 0 | 0 |
| E2 200 | 186 | 14 | 0 | 0 |
| E1 300 | 195 | 5 | 0 | 0 |
| E2 300 | 185 | 15 | 0 | 0 |

In original recorded outcomes, final non-termination rises from 2.5% to 7.5%.
Under the prefix sensitivity it is 6.5% versus 7.5%; most of the recorded increase
depends on the boundary. All five original E1 budget endings also executed a
post-exhaustion call. Fifteen E1-terminal seeds become
E2 budget endings, while all five E1 budget endings terminate under E2. Thus
the net ten additional unfinished games are not the same ten fixed episodes.
Six of those 15 E1-terminal cases were wins dependent on an extra action.
Feedback shares the trajectory allowance, so short assistant output can still
end at the generation cap.

All 19 primary success regressions were inspected through ordered actions and
visible boards: 12 losses, one draw, six unfinished. Every loss contains a
missed immediate win on the last action; the draw also misses an earlier win.
The six unfinished recorded regressions are exactly the six E1 wins dependent
on post-exhaustion dispatch; under the prefix rule both arms fail on these seeds.
They start with the invalid id, later exhaust the
trajectory budget at 2061 assistant tokens, and include either an occupied-cell
click (two) or a row-container click (four). These observations locate errors
without proving that removing any one would restore E1's outcome.

For example, episode 1 reaches X at cells 0, 1 and 3, with an immediate winning
cell 6 available. E2 chooses cell 5, then O wins. Its text describes selecting
the next available cell rather than establishing a winning line. Episode 7
clicks an occupied cell with a winning move available and then runs out of
budget. That episode is a failure for both arms under the prefix sensitivity,
not a boundary-independent E2 regression. Episode 160 misses a winning move and eventually draws. These examples
are corroborated by board state, not inferred from reasoning text alone.

A seed-17 sample retained the prior campaign's four strata: six budget endings,
six voluntary stops, six invalid/repeated-action episodes and six efficient
correct episodes. Arm, checkpoint, index and seed were masked until 24 judgments
were recorded. Aggregate results and primary regression traces were already
known, and behavior patterns could reveal arms; this was not independent blinded
review. The efficient stratum excludes errors and mostly selects earlier
checkpoints. Supplemental final controls therefore select the three shortest
E2 wins with and without invalid actions, ties by episode index. All six take a
native winning move; three recover from the initial id error. Short responses
are not intrinsically wrong, and errors can coexist with eventual success.

## Training signal, costs and host difference

E1 has varying composed reward in 21.17% of prompt groups and at least one active
group in 54.67% of updates; 136 updates have zero gradient. E2's corresponding
fractions are 50.83% and 93.67%, with 19 zero-gradient updates. Shaping therefore
changes gradient availability as well as relative length incentives. This pair
cannot isolate those effects. Training rewards read the live environment reward
after tool side effects, which feedback rollback does not undo. The batching
boundary can therefore reward a win reached through extra actions. Training-time
incidence of such credit is unmeasured here; it may have reinforced E1 batching
and is an alternative explanation for part of the arm difference. This affects
training incentives, not only evaluation, and must inform the retraining-scope
decision.

In the final ten captured updates, E1 solves 120/320 sampled rollouts (37.5%)
with zero variable task/composed groups out of 40; E2 solves 75/320 (23.44%),
with six variable task groups and 13 variable composed groups. These sampled
training statistics use different generated trajectories on aligned questions;
they are not held-out success estimates or evidence that training converged.
These rates include whatever success credit the original training boundary
allowed; they do not certify equal per-action opportunities.

| Arm | Training GPU-hours | Evaluation GPU-hours | Total GPU-hours |
|---|---:|---:|---:|
| E0 | 0 | 3.933 | 3.933 |
| E1 | 37.978 | 12.049 | 50.026 |
| E2 | 41.789 | 11.376 | 53.164 |
| Total | 79.766 | 27.357 | 107.124 |

These are allocated wall-time hours for the authorized trio's nine complete
phase attempts. Qualification, host downtime, launch overhead outside the
recorded phase boundaries and this offline review are excluded. E2 training
costs 3.811 GPU-hours more than E1.

Final mean episode time falls 72.65 to 62.52 seconds; mean inference time falls
67.30 to 57.46 seconds. Paired all-episode inference savings average 9.84 seconds
[95% bootstrap CI 7.72, 11.89]. On joint wins only, savings average 13.70 seconds
[11.17, 16.07]. Charging all E2 training to these all-episode inference savings
would give a timing-only break-even near 15285 episodes [12648, 19478]; charging
only incremental E2-versus-E1 training gives about 1394 episodes. Neither is a
quality-equivalent deployment recommendation: success is lower, evaluation
expenditure is excluded, and the host stack differs.

During E1, unattended NVIDIA package maintenance installed 580.178.04 while the
loaded driver was 580.173.02 on kernel 6.8.0-139. E1's generation engine was
already initialized, and its Transformer evaluations completed. E2 preflight
stopped before launch on the resulting NVML mismatch. After the user's reboot,
E2 ran with matching driver/NVML 580.178.04 on kernel 6.8.0-146. Fresh CUDA/vLLM
checks passed. Source hashes, Python package versions, model revision and
experimental settings remained identical. The hardware was the same GPU, but
this host-maintenance difference limits both causal and timing interpretation;
this study does not quantify its effect.

## Evidence and completion boundary

Reproducible offline statistics, immutable input hashes, fresh remote verification,
full execution-state annotations, all 19 regression traces, masked selection and
judgments, and final efficient controls are in
`pipeline/runs/tic-tac-toe-campaign-s4230-ops/final-analysis-20261013/`.
`analyze.py` uses the existing `eval.paired` functions and rejects changed inputs
or output. `collect_behavior.py` uses the frozen remote board parser and checks
all action counts against saved episodes. No training/evaluation source changed.

Counterpart review identified seven required clarifications, all accepted and
checked against the saved traces, sensitivity and execution source. Their
resolution is in `counterpart-disposition.json`; the review was not rerun after
these prose corrections. The project gate passed 509 tests (seven skipped) and
17 setup checks, and the 30 paired-statistic tests passed separately. The new
CPU reproduction demonstrates a boundary the existing gate did not cover.

The accepted as-executed result is successful compression with worse task
performance and a changed action strategy in this single pair. The result does
not establish a general mechanism of reward-bias substitution. Figures showing
checkpoint trends, the cutoff sensitivity and observed behavior are saved under
`tic-tac-toe-campaign-s4230-ops/interpretation-20261013/`. The batch ends here.
Click-tab-2 and click-checkboxes-soft remain open follow-ups.
