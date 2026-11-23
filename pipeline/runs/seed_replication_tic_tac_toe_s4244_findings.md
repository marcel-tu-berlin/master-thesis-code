# Tic-tac-toe seed 4244: endpoint instability and budget sensitivity

Prepared 2026-11-17 from the completed, hash-verified quartet. Update 300 is
primary. Independent scientific review and the final reporting-correction review
passed on 2026-11-17 with Claude Opus 5.5 at xhigh. The analysis directory retains
`counterpart-focused-output.json`, `counterpart-corrections-output.json`, and
`report-finalization.json`, which bind the reviewed report and final metadata.

## Result

Final E2 wins 58/200 games, E1 wins 5/200, and matched placebo wins 70/200.
E2 passes the fixed compression and gross-regression preservation criteria
against E1, but that comparison contains only five jointly successful games.
E1 had won 58/200 at update 200 and falls to 5/200 at the fixed final endpoint.
This is evidence of endpoint instability, not a reliable advantage over a
competent baseline. The earlier checkpoint does not replace the declared endpoint.

Against placebo, E2 compresses strongly but fails native success preservation:
there are 13 regressions and one gain. All 13 regressions are games that placebo
wins after executing a click with the allowance already exhausted. The declared
saved-prefix sensitivity removes 17 placebo wins, changing the comparison to
E2 58/200 versus placebo 53/200, with zero regressions and five gains. Both fixed
criteria pass under that sensitivity, while the discrete exact McNemar test
(p=0.0625) does not reject equal success at 5%. Native results remain primary; this is a
different estimand from the saved traces, not a corrected score or a new run.

## Fixed final comparisons

P denotes the matched 0.1 placebo. Every contrast pairs the same 200 question
seeds and initial observations. Tokens count generated assistant content,
reasoning and serialized arguments; tool feedback is excluded from this outcome
but charged to the continuation budget. Token comparisons condition on joint
wins. Negative values mean shorter responses in the named arm.

| Contrast | Gains / regressions | Joint wins / 200 | Median token change %, 95% CI | Median token difference, 95% CI | Net success pp, 95% CI | Gross-regression upper95% | Exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2-v-E1 | 53 / 0 | 5 | -56.96 [-65.62, -54.22] | -904.00 [-1315.00, -809.00] | 26.50 [20.50, 32.50] | 1.49 | 2.22045e-16 |
| E2-v-P | 1 / 13 | 57 | -69.44 [-71.68, -66.50] | -1675.00 [-1729.00, -1604.00] | -6.00 [-9.50, -2.50] | 10.14 | 0.00183105 |
| P-v-E1 | 65 / 0 | 5 | 51.98 [-16.20, 61.66] | 825.00 [-318.00, 920.00] | 32.50 [26.00, 39.00] | 1.49 | 5.42101e-20 |
| E1-v-E0 | 5 / 5 | 0 | undefined | undefined | 0.00 [-3.00, 3.00] | 5.18 | 1 |

Preservation requires the one-sided exact 95% upper bound on
P(reference success AND arm failure), out of all 200 questions, to be <=5%.
Compression requires the upper 95% endpoint of the median per-question
percentage token change to be below -10%. Gains do not cancel regressions in
the preservation rule. E2/E1 passes both; E2/P passes compression and fails
native preservation. The latter does not establish a net deficit beyond five
points: its net interval includes -5 pp. P/E1 passes preservation but does not
establish compression. E1/E0 has equal success totals on disjoint sets of five
games, so paired compression is undefined and preservation fails narrowly.

E2/E1's five joint wins are all shorter: paired means 1716.40 -> 707.00 tokens.
E2/P has 54 shorter, zero equal and three longer joint wins, with paired means
2202.77 -> 903.25. The former intersection covers only 2.5% of questions; the
latter covers 28.5%. The five-game bootstrap interval cannot support broad
claims about efficient successful play. The fixed unconditional regression
rule is particularly undemanding when the reference succeeds on only 5/200.

Intervals use 20000 paired question resamples and RNG seed 0 through the retained
family logic. They estimate question uncertainty for these fitted policies,
not training-seed uncertainty. Marginal Wilson intervals below do not describe
paired differences. Secondary/checkpoint intervals are descriptive, without
simultaneous coverage across contrasts or the campaign. The four-event adjustment
below applies only within that panel. Replication was declared after the original
family outcomes and remains exploratory.

## All declared checkpoints

Successful-token means use each policy's own wins and are descriptive, unlike
the paired contrasts above. Losses, draws and unfinished episodes are distinct.

| Policy | Wins / 200 | Success %, Wilson 95% | Loss / draw / unfinished | Mean tokens on wins | Batched episodes |
| --- | --- | --- | --- | --- | --- |
| E0-0 | 5 | 2.50 [1.07, 5.72] | 24 / 0 / 171 | 2393.20 | 0 |
| E1-100 | 48 | 24.00 [18.61, 30.37] | 66 / 4 / 82 | 1489.27 | 0 |
| P-100 | 58 | 29.00 [23.15, 35.64] | 90 / 7 / 45 | 1616.69 | 168 |
| E2-100 | 57 | 28.50 [22.70, 35.12] | 80 / 1 / 62 | 1292.53 | 0 |
| E1-200 | 58 | 29.00 [23.15, 35.64] | 84 / 2 / 56 | 1214.90 | 0 |
| P-200 | 70 | 35.00 [28.73, 41.84] | 121 / 9 / 0 | 1997.73 | 200 |
| E2-200 | 35 | 17.50 [12.86, 23.36] | 66 / 2 / 97 | 1497.26 | 0 |
| E1-300 | 5 | 2.50 [1.07, 5.72] | 31 / 1 / 163 | 1716.40 | 8 |
| P-300 | 70 | 35.00 [28.73, 41.84] | 121 / 9 / 0 | 2237.41 | 200 |
| E2-300 | 58 | 29.00 [23.15, 35.64] | 114 / 4 / 24 | 921.48 | 0 |

| Contrast | Update | Gains / regressions | Joint wins | Median token change %, 95% CI | Net success pp, 95% CI | Gross-regression upper95% |
| --- | --- | --- | --- | --- | --- | --- |
| E2-v-E1 | 100 | 15 / 6 | 42 | -12.77 [-27.46, 2.54] | 4.50 [0.00, 9.00] | 5.83 |
| E2-v-E1 | 200 | 3 / 26 | 32 | 27.36 [17.66, 44.76] | -11.50 [-16.50, -6.50] | 17.58 |
| E2-v-P | 100 | 15 / 16 | 42 | -34.92 [-40.65, -25.19] | -0.50 [-6.00, 5.00] | 11.90 |
| E2-v-P | 200 | 0 / 35 | 35 | -30.11 [-39.60, -18.40] | -17.50 [-23.00, -12.50] | 22.52 |
| P-v-E1 | 100 | 18 / 8 | 40 | 22.80 [9.58, 51.00] | 5.00 [0.00, 10.00] | 7.10 |
| P-v-E1 | 200 | 17 / 5 | 53 | 91.79 [75.10, 102.21] | 6.00 [1.50, 10.50] | 5.18 |

E1 loses 53 of its update-200 wins by update 300, gains none, and retains five.
Of those lost wins, 51 become unfinished at the generation cap and two become
native losses. Final E1 has 163 cap stops. E2's checkpoint wins are 57, 35, 58;
placebo's are 58, 70, 70. E1's and E2's trajectories rule out a
monotonic-improvement story for those two arms. Every checkpoint belongs to its existing training run, not another seed.

Fresh E0 pins the short/long thresholds at 2446 and 3225 assistant tokens.
The ruler uses the 10th and 75th percentiles of all 200 E0 completions,
including failures. The flagged fractions condition on wins; E0 itself has only
five. These length flags do not establish harmful underthinking or overthinking.
Short-success counts (<=2446) are E0 2/5; E1 44/48, 57/58, 5/5; E2 56/57,
34/35, 58/58; P 56/58, 70/70, 70/70. No successful episode exceeds 3225.
All ten reports use those same thresholds.

## Accepted execution boundary and saved-prefix sensitivity

The native rule checks continuation between model turns and can execute later
calls already emitted in a batch after feedback exhausts the allowance. The
user accepted this shared rule. A 4096-token allowance is not a hard per-action
transcript limit. No runtime, config, score or checkpoint was changed here.

The pinned evaluation tokenizer was loaded read-only on the L4 CPU. Its backend
matches the receipt-bound checkpoint tokenizers. All 2000 episode-file hashes
match the local harvest. Recounting charges generated token IDs and tool-feedback
content without message-template framing, as the evaluator does. It does not
reconstruct the training suffix ruler or independently re-tokenize held-out
assistant text. A last feedback can overshoot without any subsequent action;
retained terminal feedback also need not be model input.

| Policy | Transcripts >4096 | Maximum charged | Calls after exhaustion | Affected episodes | Affected wins |
| --- | --- | --- | --- | --- | --- |
| E0-0 | 9 | 4393 | 0 | 0 | 0 |
| E1-100 | 45 | 4414 | 0 | 0 | 0 |
| P-100 | 25 | 4680 | 10 | 10 | 10 |
| E2-100 | 27 | 4431 | 0 | 0 | 0 |
| E1-200 | 11 | 4636 | 0 | 0 | 0 |
| P-200 | 48 | 5184 | 43 | 27 | 15 |
| E2-200 | 44 | 4436 | 0 | 0 | 0 |
| E1-300 | 41 | 4681 | 0 | 0 | 0 |
| P-300 | 58 | 5514 | 70 | 34 | 17 |
| E2-300 | 28 | 4447 | 0 | 0 | 0 |

No model turn starts after exhaustion. Only placebo executes post-exhaustion
calls. Its final 34 affected episodes comprise 17 wins, ten losses and seven
draws. Every first-excluded-action board was replayed and is nonterminal.
The sensitivity stops before additional calls when remaining budget is <=0;
it retains the already-generated batch's assistant tokens and allows the last
feedback to overshoot. It neither simulates an alternative policy nor predicts
training under a different cap.

| Sensitivity contrast | Update | Reference / arm wins | Gains / regressions | Joint | Median tokens %, 95% CI | Net success pp, 95% CI | Gross-regression upper95% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2-v-P | 100 | 48 / 57 | 15 / 6 | 42 | -34.92 [-40.65, -25.19] | 4.50 [0.00, 9.00] | 5.83 |
| E2-v-P | 200 | 55 / 35 | 0 / 20 | 35 | -30.11 [-39.60, -18.40] | -10.00 [-14.50, -6.00] | 14.20 |
| E2-v-P | 300 | 53 / 58 | 5 / 0 | 53 | -69.44 [-71.85, -66.50] | 2.50 [0.50, 5.00] | 1.49 |
| P-v-E1 | 100 | 48 / 48 | 10 / 10 | 38 | 22.80 [12.17, 55.20] | 0.00 [-4.50, 4.50] | 8.33 |
| P-v-E1 | 200 | 58 / 55 | 7 / 10 | 48 | 97.25 [84.10, 104.48] | -1.50 [-5.50, 2.50] | 8.33 |
| P-v-E1 | 300 | 5 / 53 | 48 / 0 | 5 | 51.98 [-16.20, 61.66] | 24.00 [18.00, 30.00] | 1.49 |

E2/E1 and E1/E0 are unchanged by the sensitivity at every reported checkpoint.
Final E2/P loses all 13 native regressions plus four joint wins from its
reference-success set. The sensitivity passes both fixed criteria, but its five
discordant gains give exact McNemar p=0.0625. The percentile-bootstrap net interval
[0.5, 5.0] pp excludes zero while that discrete exact test does not reject at 5%;
these are different inferential procedures, not a robust superiority finding.

## Task-grounded behavior and coverage

The retained board audit labels the first executed board-cell click of a model
turn, using the board visible when that turn was chosen. Later calls in a batch
are precommitted: their intervening feedback was unavailable when chosen.
A missing tactical opportunity is unmeasured, not a passed decision.

| Final policy | Eligible decisions / episodes | Known-occupied events / episodes | Missed wins / legal win opportunities | Missed blocks / legal block opportunities |
| --- | --- | --- | --- | --- |
| E0-0 | 173 / 86 | 37 / 31 | 0 / 5 | 31 / 37 |
| E1-300 | 308 / 189 | 11 / 11 | 10 / 14 | 29 / 36 |
| P-300 | 200 / 200 | 10 / 10 | 0 / 0 | 0 / 0 |
| E2-300 | 898 / 200 | 262 / 125 | 34 / 92 | 114 / 130 |

The broad win-available/block-required exposures, including occupied choices,
are E0 5/57, E1 14/36, P 0/0 and E2 149/210. Miss numerators exclude occupied
choices; the legal-opportunity denominators above match that definition.
Events within a game are correlated and these policies induce different board
states. No independent-event binomial inference or common-opportunity tactical
preservation claim is made. Placebo's 0/0 win/block rates are undefined.

Final placebo emits the same nine board-cell clicks (bids 13,14,15,17,18,19,
21,22,23) in one batch on every question. Calls after native termination do not
execute. It has 232 occupied-at-execution clicks in 123 episodes; 200 clicks in
110 episodes target cells that were empty at commitment but became occupied by
the opponent. Those are not 200 feedback-aware occupied-cell decisions. E2 uses
one click per turn throughout and has 262 occupied clicks in 125 episodes;
its 36 repeated-action episodes coexist with zero action-error episodes.
E1 has 14 occupied clicks in 14 episodes, including three post-commitment
opponent collisions; it emits a batch in eight episodes.

All 13 final E2/P regressions are unfinished: 11 turn limits, one generation
cap, and one voluntary no-tool stop. Saved state/action summaries for these 13
cases and the five E1/E0 regressions were inspected, alongside all five E2/E1
joint wins and six reproducibly sampled E2/P joint wins. Index 1 shows E2
repeating clicks on occupied cells, then choosing cell 5 despite winning cells
7 and 8 being available at the last allowed turn. Index 40 misses a visible
win at cell 6, later clicks an occupied cell and ends without a tool call.
Index 4 is a joint win: E1 uses 1587 tokens over three turns, E2 683 over three,
and placebo generates a 2412-token nine-call batch of which three clicks execute.
The compact ledger also retains every E1 update-200 win lost at update 300;
its partition above is programmatic, not a claim that every reasoning trace
received independent manual or semantic review.

These diagnostics describe visible actions, native outcomes and coverage.
They do not establish intent, reward exploitation, or that shorter reasoning
caused a tactical mistake. Semantic off-target validation remains separate.

## Generic event panel

Counts are episode incidence out of 200. New means reference absent / arm present;
resolved means the reverse. Bounds are one-sided exact 98.75% upper limits for
new incidence, allocating alpha=0.05/4 across the four events per contrast.
The fixed non-increase margin is 5%.

| Contrast | Event | Reference / arm | New / resolved | Upper98.75% |
| --- | --- | --- | --- | --- |
| E1-v-E0 | incorrect_terminal | 24 / 32 | 27 / 19 | 19.85 |
| E1-v-E0 | invalid_action | 106 / 57 | 4 / 53 | 5.54 |
| E1-v-E0 | repeated_action | 5 / 24 | 20 / 1 | 15.78 |
| E1-v-E0 | voluntary_no_tool | 2 / 0 | 0 / 2 | 2.17 |
| E2-v-E1 | incorrect_terminal | 32 / 118 | 88 / 2 | 52.17 |
| E2-v-E1 | invalid_action | 57 / 0 | 0 / 57 | 2.17 |
| E2-v-E1 | repeated_action | 24 / 36 | 32 / 20 | 22.69 |
| E2-v-E1 | voluntary_no_tool | 0 / 1 | 1 / 0 | 3.15 |
| E2-v-P | incorrect_terminal | 130 / 118 | 0 / 12 | 2.17 |
| E2-v-P | invalid_action | 0 / 0 | 0 / 0 | 2.17 |
| E2-v-P | repeated_action | 0 / 36 | 36 / 0 | 24.92 |
| E2-v-P | voluntary_no_tool | 0 / 1 | 1 / 0 | 3.15 |
| P-v-E1 | incorrect_terminal | 32 / 130 | 99 / 1 | 57.62 |
| P-v-E1 | invalid_action | 57 / 0 | 0 / 57 | 2.17 |
| P-v-E1 | repeated_action | 24 / 0 | 0 / 24 | 2.17 |
| P-v-E1 | voluntary_no_tool | 0 / 0 | 0 / 0 | 2.17 |

E2/E1 fails this panel through incorrect terminal outcomes and repeated actions;
E2/P fails through repeated actions. Neither meets the joint off-target
non-increase criterion, even though E2/P has no new incorrect terminal outcomes.
An incorrect terminal game includes an ordinary native loss/draw. E1 leaves
most games unfinished, while placebo finishes all 200; more terminal losses
alone do not establish worse strategic intent. This panel is a set of measured
proxies, not exhaustive safety or semantic-behavior coverage. Zero invalid
calls does not imply correct play, and unknown/unexposed tactics are not zeros.

## Training signal and measured costs

All three arms match optimizer/prompt/slot identities in the saved captures.
The sampled tail is 320 rollouts across 40 groups (updates 291-300); it is not
an independent evaluation or complete training-trajectory archive.

| Arm | Mean live-group fraction | Updates with live groups | Zero-gradient updates | Tail wins /320 | Tail task-variable / composed-variable groups /40 |
| --- | --- | --- | --- | --- | --- |
| E1 | 34.67% | 237/300 | 63 | 48 | 11 / 11 |
| E2 | 37.17% | 251/300 | 49 | 62 | 4 / 9 |
| P | 35.58% | 244/300 | 56 | 72 | 0 / 9 |

Placebo changes 90 of 416 captured assignments. Its last 40 groups have no task
reward variance but nine have composed-reward variance; E2 has four task-variable
and nine composed-variable groups. These counts show altered gradient activity,
not a causal decomposition into shorter reasoning versus additional gradient.
All 300 updates in each run are finite and complete; the E1 collapse is a
scientific result, not a reason to restart a technically successful run.

| Arm | Training allocated GPU h | All evaluation allocated GPU h |
| --- | --- | --- |
| E0 | 0.0000 | 3.8427 |
| E1 | 46.8797 | 16.8678 |
| E2 | 44.3547 | 12.1604 |
| P | 41.8959 | 15.3168 |

Total allocation is 181.3179 GPU-hours across 13 completed attempts: three
training runs, nine trained-checkpoint evaluations, and one E0 evaluation.
These are allocated wall-hours on one L4, not utilization-normalized compute,
energy, currency, or end-to-end project cost. Queue/host downtime and this CPU
analysis are excluded.

| E2 contrast | Mean inference seconds saved, 95% CI | Mean episode seconds saved, 95% CI | Joint-win inference seconds saved, 95% CI | Joint-win episode seconds saved, 95% CI |
| --- | --- | --- | --- | --- |
| E2-v-E1 | 90.07 [85.88, 94.17] | 87.81 [83.60, 91.90] | 41.53 [35.43, 49.15] | 41.36 [35.25, 48.98] |
| E2-v-P | 44.51 [40.84, 48.18] | 44.29 [40.61, 47.97] | 53.08 [45.99, 59.71] | 52.88 [45.78, 59.52] |

Using point estimates of all-question inference savings, amortizing E2's full
44.3547 training hours takes about 1773 episodes against E1 or 3588 against P;
on joint wins the arithmetic is about 3845 or 3008. Including environment time
gives 1818/3605 all-question or 3861/3020 joint-win episodes. E2 training costs
2.5250 hours less than E1 and 2.4588 hours more than P; the latter incremental
cost corresponds to about 199 all-question inference episodes (167 joint-win),
or 200/167 with environment time. Negative incremental cost versus E1 does not
mean zero total cost. These calculations omit evaluation expenditure and assume
stable sequential throughput. They are not quality-equivalent deployment
recommendations, especially with only five E1 joint wins, native E2/P losses,
and no randomized timing experiment. E1 is slower than E0 with no joint wins,
so no positive-savings break-even is defined for that comparison.

## Integrity, reproducibility and seed interpretation

The four run IDs are `e0-tic-tac-toe-s4244-c4096`,
`e1-tic-tac-toe-s4244-c4096`, `e2-tic-tac-toe-s4244-c4096`, and
`e2placebo-l010-tic-tac-toe-s4244-c4096`. All use seed 4244; held-out question
seeds are 4244100000 through 4244100199. Training uses fresh pinned Qwen3-1.7B
LoRA, 300 updates, batch four, eight rollouts, naive_sum, scale_rewards none,
and the declared 4096 trajectory / 4096 prompt / 8192 context / eight-turn limits.
E2 and P use dose 0.1. No warm start, model, dose, runtime or budget change occurred.

The analysis verifies all harvested non-weight hashes against the four passing
technical receipts and the deployment manifest; all 2000 episode/report sample
pairs, 900 optimizer records, 1248 captured rollouts, and 13 cost attempts match.
Frozen E2/E1 configs differ only by identifiers/descriptions and enabling the
length component; E2/P differs only by identifiers/descriptions and shuffling.
The retained native auditor reproduces all ten audits and rejects deliberate
success-label corruption in each evaluation. Tactical synthetic controls, the
fixed preservation-bound threshold, empty joint-success handling and prefix
relabeling checks pass. Remote technical receipts separately cover model-token
recounts for captured training completions, source/config bindings, checkpoint
presence and final/update-300 weight identity. No weights were loaded for this
scientific analysis. Local source and evidence bindings are in `analysis.json`.

The local analysis uses the existing family helpers without rerunning original
completed analyses. The old temporary parser directory had lost its source
files; pinned beautifulsoup4 4.15.0 and soupsieve 2.9.1 were installed in a new
isolated temporary directory. The first install attempt hit a local cache
permission error; a temporary cache resolved it. Neither the project environment
nor remote experiment runtime changed. Reproduction uses `.venv-test/bin/python`
with `PYTHONPATH=/private/tmp/ttt-s4244-analysis-deps`.

All four new arms share L4 physical GPU 1 and the frozen stack (Linux
6.8.0-146-generic, glibc 2.39). Original seed 4230 E1 ran under kernel 6.8.0-139
and driver 580.173.02; E2 ran after maintenance on kernel 6.8.0-146 and driver
580.178.04. Original placebo ran on the home RTX 4090/Arch stack, with unverified
cross-host layout/hit-testing equivalence. The new host-consistent quartet does
not erase those historical limitations. Only gpu-l4 was contacted here.

The original seed 4230 had native E1/E2/P wins 85/66/58. Its E2/E1 contrast
compressed by 23.09% but lost 19 E1 wins with no gains; E2/P compressed by 6.28%
and failed both original criteria. Seed 4244 reverses E2/E1's success direction
because the final E1 endpoint deteriorates sharply, while E2/P now compresses
69.44% but loses native success. Budget sensitivity changes both seeds' placebo
comparisons materially; only the new seed's final sensitivity passes preservation.
The original seed's sensitivity also reduced E2/E1 regressions from 19 to 13,
without establishing preservation. Both seeds show shorter final E2 responses
on joint wins, with very different
coverage and outcome effects. There is no stable unconditional success advantage
or completed three-seed robustness conclusion. Seed 4245 is still pending.
No pooled-family or training-seed confidence interval is inferred.

Artifacts: `pipeline/runs/seed-replication-20261025-ops/ttt-s4244-analysis-20261117/`
contains the source-bound analysis, CPU budget recount, execution behavior,
boundary checks, diagnostic cases and review evidence. Task 2 remains open
until the entire 40-cell campaign and all required analyses are complete.
