# Catalog qualification: alignment with the thesis

Date: 2026-10-07. This follow-up revises the candidate ranking after reading
`../../thesis.pdf` and checking the completed read-table and inbox campaigns.
The [original screen](catalog_s4026_qualification_findings.md), its measurements,
and its delivery manifest remain unchanged. No new training run is authorized
or launched by this assessment. The two social-media variants remain deferred.

## Assessment

The original recommendation gave initial native success and an immediately
demonstrable checker gap too much weight. Click-checkboxes-soft is a useful
diagnostic of efficiency pressure acting through an incomplete task checker.
It is not the strongest primary extension for studying richer sequential
behavior under the thesis's specialization and reward-substitution questions.

For that purpose, prioritize **tic-tac-toe for a bounded learning pilot**, with
**click-tab-2 as the browser-workflow alternative**. Their lower base success
does not disqualify them. Both already have successful sampled paths, varying
task reward within question groups, observed successful-length variation and
completed native forward/backward captures under the retained GPU/budgets.
Whether either actually learns under nonzero optimization remains unmeasured.

## What the thesis requires

The inspected PDF has SHA-256
`0d5e7f2367a5e358c2aa7843a4bae5ad324e06014ac4f47b85b69e4007c47739`.
Page references below use printed page numbers, not PDF viewer indices.

- Sections 1.1-1.3, pages 1-4: specialize a small base model through task-only
  training, then measure whether shaping improves efficiency relative to that
  trained baseline while preserving task completion and relevant behavior.
- Section 4.5, page 22: substitution concerns deterioration in another behavior
  or independently measured task quality after optimizing the shaped reward.
- Sections 5.1-5.4, pages 24-26: evaluate fresh trained-policy trajectories,
  distinguish E1-minus-E0 specialization from E2-minus-E1 shaping, and validate
  behavior labels against task state. Fewer preceding actions alone do not show
  harmful under-verification.
- Section 6.2, page 28: require a goal-related reward, within-group task signal,
  and an efficiency target that remains relevant under E1. These concern learning
  and post-training behavior, not a universal minimum base success percentage.

The 40-80% preference was a historical difficulty heuristic, retained in the
repository and the older menu-task description in Section 6.3. It should not
exclude a harder family with useful learning signal. The completed screen
already tested several such families for 128 rollouts; the correction changes
their interpretation and priority, not the observations or recorded admission
rules. No replacement numeric threshold is proposed here.

The PDF contains unfinished and historical protocol text. Section 6.3 still
describes the earlier menu-only campaign. Section 6.2 also asks for E1 success
clearly below one; the table campaign shows that near-perfect success can still
leave substantial measurable E2 compression. What matters is the remaining
cost variation and diagnostic behavior, not requiring errors by construction.
The introduction on page 4 asks the panel to change a verdict, while Section 5.6
on page 27 explicitly permits a sufficiently precise null. These should be
reconciled when updating the thesis. Family selection should create meaningful
opportunities to detect harm, without requiring the eventual result to be harm.
The PDF was not edited; its deferred dose, replication, shifted-family and E3
plans do not authorize additional runs.

## Prior learning is the relevant counterexample

| Family or operation | Greedy held-out E0 | Task-only E1 at update 300 |
|---|---:|---:|
| read-table-2 | 43/200 (21.5%) | 199/200 (99.5%) |
| Mixed inbox, all operations | 103/200 (51.5%) | 183/200 (91.5%) |
| Inbox reply | 3/54 (5.6%) | 51/54 (94.4%) |
| Inbox forward | 19/60 (31.7%) | 47/60 (78.3%) |
| Reply and forward combined | 22/114 (19.3%) | 98/114 (86.0%) |

Sources: [read-table comparison](e0_e2_s4016_findings.md) and
[corrected inbox comparison](inbox_s4021_c4096_findings.md). These are the valid
4,096-token campaign results. The inbox qualification plan also explicitly
accepted an earlier 11/32 development screen, at its diagnostic settings, below
the old preferred band because three groups supplied task signal. These
observations support admitting low-starting
candidates for learning tests; they do not prove that every low-starting task
will improve.

The catalog screen uses temperature-1 training samples grouped eight per reset
seed, whereas the campaign E0 numbers above are greedy held-out evaluations.
Their percentages are not directly comparable measures of base competence.

## Revised priorities

All catalog figures below describe 128 frozen-policy rollouts on 16 reset seeds.
Mixed groups contain successes and failures for the same question. Length-live
groups use the original screen's successful-peer variation criterion.

| Candidate | Native success | Mixed groups | Length-live groups | Reason to retain |
|---|---:|---:|---:|---|
| tic-tac-toe | 32/128 (25.0%) | 10/16 | 6/16 | Opponent changes the board after a move; later choices depend on the updated state. Existing audits identify occupied-cell clicks and tactical opportunities. |
| click-tab-2 | 26/128 (20.3%) | 12/16 | 8/16 | Hidden target search, keeping track of inspected tabs, responding to newly visible links and deciding when to commit or abandon search. |
| click-tab-2-hard | 33/128 (25.8%) | 8/16 | 7/16 | Similar mechanism, but almost all observed success is on initially visible targets; aggregate accuracy overstates demonstrated search competence. |
| form-sequence-3 | 13/128 (10.2%) | 6/16 | 3/16 | Low starting success is not a rejection reason, but the actual task is dropdown selection followed by a named button, so its added sequential complexity is modest. |

The [tic-tac-toe task](https://miniwob.farama.org/environments/tic-tac-toe/)
requires winning. In the pinned implementation, the opponent mixes random and
tactical moves; its response makes later board observations consequential.
The existing audit identifies 21 clicks on known occupied cells in 19 episodes,
three missed available wins, and 27 unblocked single threats. These are different
from form-field omissions or fixed inbox action sequences. Threat exposure is
descriptive risk, not a claim about optimal expected win probability against
this opponent. The audit separates actions chosen after feedback from subsequent
actions precommitted in the same tool batch. Draws are native failures.

Tic-tac-toe is a game rather than an ordinary business workflow, which limits
external validity for the on-demand web-workflow example. It also has many
budget endings: 48 generation caps and 15 feedback-budget rejections. These
make learning progress and residual censoring important pilot outcomes; they do
not establish that the base policy is unlearnable. Its sampled peak memory is
20,650 MiB (20.2 GiB), with a maximum observed padded sequence of 4,813 tokens.

[Click-tab-2](https://miniwob.farama.org/environments/click-tab-2/) is the more
direct browser-workflow choice. The new initial-visibility audit finds:

| Variant | Initially hidden target | Initially visible target |
|---|---:|---:|
| click-tab-2 | 17/112 successes (15.2%) | 9/16 (56.2%) |
| click-tab-2-hard | 1/64 successes (1.6%) | 32/64 (50.0%) |

This split uses only the initial policy-visible link text and native outcomes;
it does not infer that every hidden-target success used a good search strategy.
Ordinary click-tab-2 supplies more observed examples of successful hidden-target
resolution, despite lower pooled accuracy. The variants use different questions,
so their difference is descriptive, not a paired causal difficulty estimate.
Click-tab-2 also has 34 generation caps and 28 feedback-budget rejections. Its
sampled memory peak is 20,623 MiB, with maximum observed padding of 5,166 tokens.
The [visibility labels and input hashes](catalog-qualification-s4026-ops/thesis-alignment-visibility.json)
can be regenerated with `thesis_alignment_visibility.py` in the same directory.
The parser's three self-checks and all 256 retained-record checks pass.

The official [hard-tab description](https://miniwob.farama.org/environments/click-tab-2-hard/)
varies the number of tabs. The [form-sequence-3 description](https://miniwob.farama.org/environments/form-sequence-3/)
and pinned source confirm the compact dropdown/button task. Low success and a
name suggesting complexity are not sufficient evidence of useful complexity.

## Role of the checkbox result and the next learning test

Keep click-checkboxes-soft as a separate checker-gap diagnostic. Its exact-goal
audit and offline E2 preference findings remain valid. However, a scorer already
accepting incomplete work introduces a pre-existing objective mismatch. Training
would have to show that E2 changes the behavior relative to E1 before attributing
the result to efficiency shaping. Choosing it merely because the fixed traces
already show an adverse E2 ranking would narrow the thesis to a favorable
demonstration case. This distinction revises its priority, not its scientific use.

For the primary extension, the remaining question is whether the richer family
learns while retaining informative decisions and measurable efficiency headroom.
A bounded, nonzero-learning-rate E1 pilot should measure progress, budget endings
and the validated behavior panel on separate development questions. Family
selection should use that trajectory, without imposing a 40% base-score floor.
Zero observed successes and zero task gradient remain reasons to investigate
signal or interface problems, distinct from a 10-25% starting rate with mixed
groups. A zero-LR capture cannot settle either family's learning curve.

If a pilot supports proceeding, declare a fresh E1/E2 comparison and any shifted
split before its final test data are inspected. Train both arms independently
from base under matched settings; pilot checkpoints do not supply a warm start.
Freeze the task-grounded indicators and opportunity denominators before evaluating
the trained policies. Preserve the existing token, turn and batch limits. The
completed captures support trying the current GPU; they do not guarantee every
later long batch or a complete campaign will fit.

Next steps: propose a bounded tic-tac-toe learning pilot, or click-tab-2 when
browser-workflow transfer is the priority. No experiment was launched here.
