# Tic-tac-toe seed 4230: matched 0.1 placebo

Reviewed 2026-10-20 (Europe/Berlin). The placebo completed at
2026-10-19 21:54 UTC. Its 29 non-weight files, including both intermediate
checkpoint evaluations, were harvested and hash-verified. This report adds the
authorized placebo comparisons; it does not replace the completed original
E0/E1/E2 findings. Update 300 is primary.

## Result

E2 has a small token advantage over placebo on their joint wins, but does not
meet the declared 10% compression target. Its native win-rate advantage is
uncertain: 66/200 versus 58/200, a paired change of +4.0 percentage points
(95% bootstrap interval -3.5 to +11.5). There are 34 gains and 26 regressions,
so the gross-regression preservation criterion also fails. This is not evidence
that the arms are equivalent, nor evidence of a native success advantage.

On 32 joint wins, E2 uses a median 6.28% fewer assistant tokens
(95% interval 2.55% to 12.76% fewer). The interval does not establish at least
10% compression. Placebo itself is shorter and less successful than E1. These
results do not isolate successful compression with preserved performance from
correctly assigning the length cost in this family.

The main behavioral distinction is execution strategy. Final placebo emits the
same 15-click batch in 193 episodes; E2 emits single calls. Placebo benefits
substantially from the accepted within-batch execution boundary: 21 of its 58
wins occur only after the continuation allowance is exhausted. Native results
remain primary, as agreed for this family. A saved-prefix sensitivity removes
those wins and materially changes the success comparison, detailed below.

This is one training seed and a subsequent exploratory mechanism comparison.
Placebo ran on the home RTX 4090/Arch stack; E1 and E2 ran on L4/Ubuntu.
The user accepted that host difference, but it remains confounded with the new
arm. Neither paired evaluation intervals nor matching initial observations
establish training-seed robustness or host equivalence. In particular, the
placebo's container clicks depend on rendered layout and browser hit-testing,
which were not checked for equivalence across hosts. The result does not
identify intent or a general causal mechanism of reward-bias substitution.

## Design and integrity

The protocol is `docs/plans/placebo-l010-campaign.md`, decision 0028, and the
original `docs/plans/tic-tac-toe-family-extension.md`. The new run is
`e2placebo-l010-tic-tac-toe-s4230-c4096`, compared primarily with
`e2-tic-tac-toe-s4230-c4096` and secondarily with
`e1-tic-tac-toe-s4230-c4096`.

Frozen placebo/E2 configs differ only in ID, description, enabling placebo
assignment, and the host's OpenEnv path. Weight is 0.1. All retain fresh LoRA
from pinned Qwen3-1.7B, 300 updates, batch four, eight rollouts, `naive_sum`,
`scale_rewards: none`, and 4096 trajectory / 4096 prompt / 8192 context / eight
turn budgets with the accepted batch/continuation semantics. No warm start,
new seed, new dose or changed evaluator was used.

The historical runtime source hashes match the original manifest except the
two recorded physical GPU routing changes in `training/train.py` and
`eval/runner.py`. Python, package versions, OpenEnv pin and episode-boundary
stamp match. The platform stamps differ: E1 Linux 6.8.0-139, E2 6.8.0-146,
both glibc 2.39; placebo Linux 7.2.5-3-omarchy, glibc 2.44. Hardware and host
differences also prevent interpreting wall-time differences as a reward effect.

All three checkpoints contain 200 paired seeds, 4230100000-4230100199, with
identical initial observations across the three arms. The small set of possible
initial boards is unchanged; these seeds are not 200 novel starting positions.
Later state visitation and opponent responses depend on the policy's actions.
The 416 captured training seed/prompt/group/slot identities match as well.

The technical receipt passes all 300 optimizer records, all 600 evaluations,
checkpoint/final identity, native-board outcomes, token recounts, metric
recomputation, frozen sources/config/stack and four complete cost attempts.
The local analysis rechecked every harvested file and replayed the new
placebo's native auditor and capture check. The seeded shuffle audit finds 82
changed cost assignments among 416 captured rollouts. Observer component values
remain unshuffled by design; composed rewards contain the actual assignment.

## Checkpoints and paired comparisons

| Policy | Wins / 200 | Success, Wilson 95% CI | Mean assistant tokens on wins | Draw / loss / unfinished |
|---|---:|---:|---:|---:|
| E1 100 | 61 | 30.5% [24.54%, 37.20%] | 793.51 | 4 / 102 / 33 |
| Placebo 100 | 66 | 33.0% [26.86%, 39.78%] | 731.59 | 2 / 81 / 51 |
| E2 100 | 71 | 35.5% [29.20%, 42.35%] | 1525.30 | 14 / 111 / 4 |
| E1 200 | 76 | 38.0% [31.56%, 44.89%] | 1780.14 | 3 / 98 / 23 |
| Placebo 200 | 71 | 35.5% [29.20%, 42.35%] | 899.63 | 10 / 119 / 0 |
| E2 200 | 66 | 33.0% [26.86%, 39.78%] | 1696.39 | 16 / 104 / 14 |
| E1 300 | 85 | 42.5% [35.85%, 49.43%] | 1677.35 | 4 / 106 / 5 |
| Placebo 300 | 58 | 29.0% [23.15%, 35.64%] | 1306.90 | 13 / 122 / 7 |
| E2 300 | 66 | 33.0% [26.86%, 39.78%] | 1309.00 | 13 / 106 / 15 |

The nearly equal final placebo/E2 win-conditioned means use different games.
Joint wins supply the paired efficiency endpoint. Checkpoints are repeated
measurements of these runs, not independent replications or a selection rule.

| Policy | Env done | Voluntary no-tool | Generation cap | Turn cap |
|---|---:|---:|---:|---:|
| E1 100 | 167 | 5 | 26 | 2 |
| Placebo 100 | 149 | 47 | 4 | 0 |
| E2 100 | 196 | 0 | 4 | 0 |
| E1 200 | 177 | 0 | 23 | 0 |
| Placebo 200 | 200 | 0 | 0 | 0 |
| E2 200 | 186 | 0 | 14 | 0 |
| E1 300 | 195 | 0 | 5 | 0 |
| Placebo 300 | 193 | 0 | 7 | 0 |
| E2 300 | 185 | 0 | 15 | 0 |

The 47 voluntary no-tool stops at placebo update 100 are distinct from budget
exhaustion and disappear by update 200; the final panel alone does not show
this earlier behavior.

| Contrast | Update | Gains / regressions | Joint wins | Median paired token change, 95% CI |
|---|---:|---:|---:|---:|
| E2 minus placebo | 100 | 23 / 18 | 48 | +101.11% [+88.49%, +104.32%] |
| E2 minus placebo | 200 | 3 / 8 | 63 | +81.29% [+78.73%, +86.08%] |
| E2 minus placebo | 300 | 34 / 26 | 32 | -6.28% [-12.76%, -2.55%] |
| Placebo minus E1 | 100 | 9 / 4 | 57 | -10.09% [-10.55%, -7.02%] |
| Placebo minus E1 | 200 | 5 / 10 | 66 | -50.58% [-50.58%, -50.58%] |
| Placebo minus E1 | 300 | 21 / 48 | 37 | -23.77% [-23.77%, -21.21%] |

Primary E2/placebo: exact McNemar p=0.3663. The one-sided exact 95% upper
bound on P(placebo wins, E2 fails) is 17.58%, above the fixed 5% margin.
The positive net point estimate does not establish preservation under this
gross-regression rule. Median absolute token change is -78.5
[-173.5, -29.5]; joint means are 1303.13 placebo and 1223.38 E2. E2 is
shorter in 23 joint wins and longer in nine. The 26 regressions comprise
17 losses, five draws and four unfinished E2 games. The 34 gains replace
31 placebo losses and three draws.

Secondary placebo/E1: native success changes -13.5 points [-21.5, -5.5],
McNemar p=0.001550. Its regression upper bound is 29.49%, so preservation
fails despite passing the 10% compression target on joint wins. Median
absolute token change is -424 [-424, -305]; 36 of 37 joint wins are shorter.
The 48 regressions are 43 placebo losses and five draws. Gains replace
17 E1 losses, one draw and three unfinished games. Point-mass bootstrap
intervals at update 200 reflect repeated fixed outputs, not zero general
uncertainty. All bootstrap intervals use 20000 paired resamples and seed zero;
they describe evaluation-question uncertainty conditional on these policies.

## Accepted execution boundary and saved-prefix sensitivity

The original family review established that the implementation checks budget
before continuing to another model turn, not between precommitted calls.
The user accepted this behavior. A 15-call batch in one turn is therefore not
a violation of the eight-turn limit. This report does not alter that rule or
reinterpret native scores as corrupt.

The new CPU recount uses the saved checkpoint tokenizer, whose `tokenizer.json`
hash matches both original arms. It uses the original evaluator charge:
assistant generation tokens plus tool-feedback content tokens, excluding
message-template framing. It is not a training-suffix or hard transcript-cap
certification. No model turn starts with exhausted allowance in placebo.

At final placebo, 142 actions execute after exhaustion across 47 episodes.
Their native outcomes are 21 wins, 13 losses and 13 draws. All 47 saved boards
at the first excluded action are nonterminal. Ninety-three final transcripts
exceed 4096 charged tokens, up to 6470. These include feedback retained for
audit, and do not establish an overlength GPU forward/backward pass. At placebo
100 there is one final-feedback overshoot (4284), with no later action; at
200 there are no overshoots (maximum 4064). Neither earlier checkpoint has
post-exhaustion actions.

The retained original recount reports seven final E1 wins dependent on a later
action, and none for E2. Applying exactly the same saved-prefix rule to all
three final arms gives E1 78 wins, placebo 37 and E2 66. The rule stops before
an additional action once budget is nonpositive, permits the last feedback to
overshoot, and retains the already generated assistant batch for token counting.

| Final sensitivity contrast | Gains / regressions | Success change, 95% CI | Joint wins | Median token change, 95% CI |
|---|---:|---:|---:|---:|
| E2 minus placebo | 42 / 13 | +14.5 points [+7.5, +21.5] | 24 | -11.40% [-15.15%, -5.37%] |
| Placebo minus E1 | 11 / 52 | -20.5 points [-27.51, -13.5] | 26 | -22.49% [-23.77%, -14.97%] |

For E2/placebo, sensitivity McNemar p=0.0001136 and the regression upper
bound is 10.14%. Thus neither the 10% compression target nor the 5%
gross-regression criterion is established even under this alternate endpoint.
For placebo/E1, the regression upper bound is 31.60%. At update 200 the
secondary native -2.5-point difference becomes +2.0 points under the prefix
rule; the primary E2/placebo contrast is unchanged at both earlier checkpoints.
All checkpoint sensitivity values are retained in `analysis.json`.

This sensitivity is a different estimand on saved prefixes, not a new evaluation,
an amended primary protocol, or a prediction of retraining with another boundary.
It shows why a single native win-rate comparison is insufficient here.

## Task-grounded behavior

In all 193 final placebo episodes that act, the sole assistant turn emits
`click` calls for numeric bids 9 through 23 in order. Execution stops when the
game terminates. The remaining seven episodes generate 4096 assistant tokens
without an action. Bid 9 is the enclosing `div` with id `wrap`, not a board
cell. Six of the 15 targets are containers: wrap, query, area and three
ttt-row divs. Across execution,
898 non-cell-targeted clicks include 357 board changes and 541 unchanged boards.
Every acting episode has at least one board change after a non-cell click.
These observations do not imply an invalid API call or bypass of the win scorer.

At index 0, clicking the area container (bid 11) places X in board cell 4.
This is consistent with a click at a container's centre hitting a descendant
cell through rendered browser hit-testing. Saved HTML and board traces do
not establish equal pixel layout or hit-testing on the two hosts. Final E1
has no such non-cell clicks and E2 has four that leave the board unchanged,
so this unverified host-dependent pathway is concentrated in the placebo
comparison. Identical initial HTML and pinned packages do not remove it.

The frozen tactical auditor requires an explicit board-cell click in the first
position of a model turn. Final placebo has zero eligible decisions. Its zero
missed-win and missed-block counts are therefore 0/0, not perfect performance.
Later calls were precommitted, so their consequences cannot be called ignoring
feedback received after commitment. The fixed auditor is retained; its coverage
limit is reported rather than silently changing the measure.

| Policy | Eligible decisions | Known occupied events (episodes) | Missed win / opportunities | Missed single block / opportunities |
|---|---:|---:|---:|---:|
| Placebo 100 | 414 | 117 (60) | 1 / 9 | 46 / 116 |
| Placebo 200 | 200 | 13 (13) | 0 / 0 | 0 / 0 |
| Placebo 300 | 0 | 0 (0) | 0 / 0 | 0 / 0 |
| E1 300 | 200 | 13 (13) | 0 / 0 | 0 / 0 |
| E2 300 | 683 | 50 (22) | 33 / 102 | 103 / 138 |

At placebo 100, win opportunities occur in nine episodes and block opportunities
in 68; missed events occur in one and 46 episodes respectively. Denominators
above include occupied choices, while missed-event numerators require legal
choices. The legal-choice denominators are 6 and 50 for placebo 100, and 99
and 126 for final E2. E2's missed-win/block episodes are 33 and 101, with
opportunities in 97 and 122 episodes. Opportunities cluster within episodes
and differ with state visitation; these ratios are not independent binomial
trials or evidence that placebo was tactically superior.

The missed-win and missed-block 0/0 entries for placebo 200 and E1 300 are
also undefined: their eligible decisions only expose opening states without
these opportunities. The retained `analysis.json` field `board_pairs` treats
unobserved tactical events as false. Its cross-arm differences against final
placebo are coverage artifacts, not interpretable error-rate contrasts.
For example, the 101 "new" missed-block episodes for E2 versus placebo and
13 "resolved" occupied-choice episodes for placebo versus E1 do not establish
tactical worsening or improvement. Use the explicit opportunity counts above.

| Placebo update | Occupied-cell executions (episodes) | Post-commitment opponent collisions (episodes) | Non-cell clicks |
|---|---:|---:|---:|
| 100 | 232 (113) | 98 (88) | 0 |
| 200 | 207 (117) | 173 (104) | 3 |
| 300 | 346 (149) | 155 (104) | 898 |

At update 200 the policy still targets board cells almost exclusively in a
nine-call batch; only three executed clicks target non-cell elements. The fixed
15-call container-inclusive batch therefore appears between updates 200 and
300, rather than describing the whole placebo training trajectory.

Execution-time occupied-cell clicks total 346 for placebo, 203 for E1 and 50
for E2. Of placebo's clicks, 155 hit cells newly occupied by the opponent after
commitment, across 104 episodes. For E1, 170 of its 203 occupied clicks are
also post-commitment opponent collisions, across 104 episodes. Final placebo
and E1 calls are all committed in batches before their within-batch feedback.
E2 has zero such collisions: its 50 occupied clicks are single calls selected
with the preceding board feedback available. These are different decision
conditions. Total unchanged-board actions are 887 placebo,
203 E1 and 154 E2. The latter includes 100 rejected calls. Generic invalid and
exact-repeat metrics alone therefore miss much of placebo's ineffective action
sequence. Non-cell clicks can also change the board, so they are not all no-ops.

| Final generic event | E1 / 200 | Placebo / 200 | E2 / 200 | New / resolved, E2 versus placebo |
|---|---:|---:|---:|---:|
| Incorrect terminal outcome | 110 | 135 | 119 | 29 / 45 |
| Voluntary no-tool stop | 0 | 0 | 0 | 0 / 0 |
| Any invalid action | 0 | 0 | 99 | 99 / 0 |
| Any exact repeated action | 0 | 0 | 7 | 7 / 0 |

The primary one-sided 98.75% upper bounds on new-event probability are 20.99%,
2.17%, 57.62% and 7.65%, respectively. Only voluntary stopping meets the
5% criterion. In the secondary comparison the bounds are 32.54% for incorrect
terminal outcomes and 2.17% for each other event. A game loss/draw is the
incorrect-terminal proxy, not a false claim. Counts are native outcomes, not
prefix-sensitive replacements. Generic underthinking is 100% on wins and
unsupported claims zero in all three final arms; neither proves harmlessness.
Verification depth counts preceding calls, not meaningful board verification.

Trace inspection is unblinded and diagnostic. Cases 0, 1, 2, 17, 24, 25 and 88
cover opposite paired transitions and boundary examples. All seven placebo
cap stops and all 47 boundary-affected prefixes were checked. Six joint wins
per contrast were selected with `random.Random(17)`; their indices, complete
action/board traces, output text and outcome partitions are retained in
`behavior-review.json`. These cases illustrate the aggregate counts, not a
separate confirmatory test. They do not prove that less reasoning caused an
error or that the model intentionally exploited batching.

## Learning activity and cost

| Training statistic | E1 | Placebo | E2 |
|---|---:|---:|---:|
| Mean fraction of groups with nonzero reward SD | 21.17% | 38.08% | 50.83% |
| Fraction of updates with a live group | 54.67% | 84.33% | 93.67% |
| Zero-gradient updates / 300 | 136 | 47 | 19 |
| Native successes in last 320 captured rollouts | 120 | 63 | 75 |
| Task-variable groups in last 40 captured groups | 0 | 5 | 6 |
| Composed-reward-variable groups in those 40 | 0 | 10 | 13 |

The shuffled arm is not inert, but its realized learning activity differs from
E2. Shuffling does not hold trajectory distributions, total reward variance or
gradient noise constant. The placebo contrast therefore does not identify a
pure assignment effect independently of all training dynamics and host effects.

Placebo cost is 16.8205 allocated GPU-hours for training plus 3.2572 for the
three evaluations: **20.0777 hours, about 20 hours 5 minutes**. The four measured
attempts completed without a retry. Original E1 and E2 totals are 50.0263 and
53.1645 L4 hours, respectively, retained only for accounting. Allocated hours
are wall time while reserving the GPU, not active GPU utilization or monetary
charges. The faster home run is not evidence of a reward-induced speedup.

## Reproducibility and remaining review

Analysis and evidence are in
`placebo-l010-20261018-ops/ttt-analysis-20261020/`: `analyze.py`,
`analysis.json`, `recount_budget.py`, `budget-recount.json`, `behavior.json`,
`review_cases.py`, `behavior-review.json` and `validation.json`.
The counterpart correction adds `review_supplement.py` and
`review-supplement.json`, retaining per-checkpoint stops/execution counts,
the 898/357/541 non-cell derivation, all 193 affected episode indices and the
area-click example. The supplement checks episode identities and reconciles
board-change flags before summarizing. Earlier outputs remain unchanged;
`board_pairs` is retained only with the coverage limitation stated above.
The paired functions reuse the retained implementation. Direct arithmetic,
native audits, deterministic reproduction and the campaign tests were checked.
No original raw result, training source, evaluator or checkpoint was changed.

Authentication is restored. Independent Opus 5.5 at xhigh confirmed the
reported main results and returned five findings about tactical coverage,
rendering dependence, precommitment, checkpoint behavior and retained
derivations. Corrective Opus 5.5/xhigh review passed on 2026-10-20;
its verdict, input hashes and execution receipt are retained as
`counterpart-correction-{output,receipt}-20261020`. The home run queue is complete; campaign task 9
remains open for the other families and outstanding review.
