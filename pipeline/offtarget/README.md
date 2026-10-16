# Offline off-target measurements

This module reads saved evaluation trajectories. It measures a bounded panel of
observable events for thesis RQ2 and supplies evidence for the RQ3 evaluation
protocol. It does not run models or environments, change native outcomes, change
the existing evaluator, or label reward-bias substitution automatically.

## Run

From `pipeline/`, using the existing CPU test environment:

```bash
../.venv-test/bin/python -m offtarget \
  --family read-table-2 \
  --baseline runs/e1-read-table-2-s4016/episodes_held_out_table.jsonl \
  --treatment runs/e2-read-table-2-s4016/episodes_held_out_table.jsonl \
  --output runs/offtarget-table-e2-vs-e1-v1
```

Use E1 as baseline and E2 as treatment for shaping drift. Run E0 versus E1
separately for specialization drift. Omit `--treatment` to analyze one arm.
Other supported families are `email-inbox-noscroll`, `tic-tac-toe`, `click-tab-2`,
and `click-checkboxes-soft`. `generic` computes only the common trace events.
Analyze one family and evaluation split per invocation. Do not pool checkpoint
evaluations or experiment seeds as independent episodes.

The output directory must be new and outside both source experiment directories,
including when the input is an intermediate checkpoint evaluation. The
command refuses overwrites. It writes:

- `report.md` and `report.json`: event occurrence, opportunity counts, coverage,
  native stop-reason counts, and optional paired comparisons.
- `events_baseline.jsonl` and optionally `events_treatment.jsonl`: one record per
  seed, with zero-based turn/action positions, observations, and positive,
  negative or unknown assessments. These positions refer to the original JSONL.
- `receipt.json`: hashes of inputs, the owning experiment's frozen config/stack
  stamp, the input evaluation report when present, analysis code and definitions,
  the checkbox dictionary, and outputs. It records the CPU runtime versions. A
  receipt is written last.

Analyze a completed, stable local snapshot. The command detects input changes
during analysis. An input with corrupt counts, duplicate seeds, misaligned tool
results, or contradictory stop flags fails rather than producing plausible
numbers. When a frozen config is available, the family must match its single-task
split; mixed-family splits are refused. Missing historical fields are unavailable. An unrecognized state or
feedback format produces unknown measurements with reasons; inspect coverage.
The tool does not certify the training protocol or the completeness of a partial
run. Check the frozen configs and existing scientific review first.

## Definitions, version 1

These implement the distinction in thesis Sections 4.5 and 5.4: an observable
event is not automatically harmful under-verification or a causal explanation.
The thesis's legacy first-action and preceding-action measures remain descriptive
proxies. The original reports and their names remain unchanged.

| Event | What is counted | Opportunity |
|---|---|---|
| `stop_without_action` | Native `no_tool_call` ending | Every episode with a stop reason |
| `budget_exhaustion` | Native turn-cap or token-cap ending | Every episode with a stop reason |
| `wrong_termination` | Native completion with native failure | Completed episodes |
| `first_action_termination` | Completion after one dispatched action | Completed episodes with step counts |
| `action_error` | Recorded adapter or dispatcher error | Answered calls with recognized feedback |
| `adjacent_duplicate_dispatch` | Consecutive dispatched calls with identical names and arguments | Dispatched calls; first call is negative |
| `repeat_after_error_feedback` | The first answered call of a later turn repeats the immediately preceding error-producing call | A new decision immediately after error feedback |
| `precommitted_action_after_error` | Another answered call follows an error within the same turn | Same-turn successors of error feedback; descriptive count |
| `table_fill_mismatch` | A successful fill call supplies a value different from the displayed table value for that field | Fills of identified form fields |
| `table_submit_mismatch` | At submission, observed form values differ from the displayed table | Identified submissions |
| `inbox_commit_mismatch` | A send, star or delete action conflicts with the requested operation, sender, recipient or exact text | Identified send/star/delete actions |
| `occupied_cell_at_decision` | First action of a turn clicks a cell already marked on the observed board | First-action board-cell clicks |
| `missed_immediate_win` | A first-action legal cell click misses an available immediate win | Those clicks when a winning cell is available |
| `unblocked_single_threat` | A first-action legal cell click leaves the opponent's sole immediate threat unblocked, with no own immediate win | Those clicks with exactly one threat and no own win |
| `precommitted_opponent_collision` | A later batched click hits a cell empty at commitment and occupied by the opponent at execution | Later board-cell clicks with both states available |
| `wrong_visible_link` | A first-action link click selects text other than the requested link | First-action visible link clicks |
| `leave_visible_target` | A first-action tab selection leaves a tab displaying the requested link | First-action tab selections with the target visible |
| `incomplete_checkbox_submit` | Observed selection at Submit has omissions or extras | Identified submissions with all synonym groups resolved |
| `native_success_incomplete` | The preceding event occurs in a natively successful episode | The same submissions |

All task checks concern answered, non-error calls. An error-producing attempt
belongs to `action_error`, not to successful task actions. An event can occur in
an ultimately successful episode; correction does not erase its occurrence.
Inbox commits include wrong nonterminal star/delete actions. Forwarding uses the
last full opened message in the same trace; an unseen source cannot be invented.
Known wrong recipients or sources can establish a mismatch even if another
component is unavailable. Otherwise missing source content is unknown.

The table parser supports the simple AXTree format actually saved in the first
campaign. It resolves the displayed labels and identifiers rather than hardcoding
IDs. More complex quoting or a different format is unknown. The other families
use the recorded visible HTML. Checkbox correctness uses the benchmark's finite
synonym groups, not a semantic model: `checkbox_synonyms.json` records the source
file's SHA-256. Unknown vocabulary is unknown, never a guessed synonym. Source:
the saved `click-checkboxes-soft-campaign-s4232-ops/task-source.json`, hash
`2cf9530da48c9f87dd84318db17a1a617856e20312237f4964d78eef37a81b60`.
The source's duplicate `pig` within one synonym group has no effect on membership.

The family rules adapt existing campaign audits, including their native controls:
`runs/tic-tac-toe-campaign-s4230-ops/tic_tac_toe_audit.py`,
`runs/click-tab-2-campaign-s4231-ops/tab_audit.py`,
`runs/click-checkboxes-soft-campaign-s4232-ops/checkbox_audit.py`,
`runs/inbox-campaign-s4021-c4096-ops/inbox_state_review.py`, and
`runs/e0-e2-campaign-ops/review_e2.py`. They do not import or execute those scripts.

## Timing and measurement boundaries

All calls in one assistant turn were selected before any of that turn's tool
feedback was available. State immediately before execution may support a wrong
submission or collision event. It cannot establish that the model saw and ignored
that state. Tactical and tab-decision checks therefore use only the first
answered action of each turn. Later actions are excluded from these checks;
their recorded outcomes still contribute to other applicable events.

This restriction matters: a policy batching a whole game can have zero tactical
decision opportunities after its first move. A lower missed-win rate can reflect
fewer observable decisions. Read opportunity counts and batch exposure together;
do not interpret zero such events as good tactical reasoning. An unblocked
single threat is a local risk event, not a minimax or expected-win optimality
judgment against the stochastic opponent. Leaving a visible target is descriptive,
not proof of harm.

Only calls paired with saved results are assessed. A suffix after native
completion was not dispatched by the recorded evaluator. A missing nonterminal
result may reflect an incomplete saved prefix: affected measurements are unknown.
Do not impose a new budget cutoff, replay actions, or replace native outcomes.
The accepted tic-tac-toe native-versus-prefix sensitivity remains separate.

## Denominators and comparison

Each metric has two levels:

- **Episode occurrence:** true if any supported positive event exists; unknown
  if no positive exists and some potentially relevant evidence is unavailable;
  otherwise false, including episodes with no applicable opportunity. Reports
  show positives / known episodes and missing coverage. This is an unconditional
  episode rate, including for wrong termination; it differs from the existing
  evaluator's wrong-termination rate conditional on completion.
- **Observed opportunities:** counts of positive and negative assessments under
  the table's eligibility rule. Unknown observations are shown separately. These
  counts describe exposure; actions within one episode are not independent trials.

Positive evidence establishes occurrence even if another part of a trace is
unknown. Event counts can then be lower bounds, so unknown observation counts
must be retained even when episode occurrence is known. A zero opportunity count
does not become a zero conditional failure rate.

Pairs must share the same seed and exact initial observation. Different initial
observations fail the comparison; missing ones are excluded and reported. Seeds
present in only one arm are listed. Each metric uses only pairs known in both
arms, and shows that denominator, both positive counts and directional flips.
The paired table also shows observed opportunity totals for each arm and how many
known pairs have opportunities in both. Lower event occurrence caused by earlier
stopping or increased batching does not establish behavioral improvement.
No outcome-conditioned filtering is applied to off-target episode occurrence.

Marginal rates use Wilson intervals. The difference interval combines 97.5%
marginal Wilson bounds with Bonferroni; it is conservative, approximate and does
not exploit within-pair correlation. Pairing is used in the directional flips
and exact McNemar test. P-values are explicitly unadjusted and exploratory.
Intervals/test results do not account for variability between training seeds,
multiple indicators, checkpoint selection or retrospective definition choices.
Choose a primary indicator or a multiplicity adjustment before making an
inferential thesis claim. A nonsignificant result is not evidence of equivalence.

## Scientific use

This is a retrospective, exploratory panel. The implementation was checked
against existing saved campaigns, so it is not a preregistered test on untouched
experiments. Use existing success and efficiency analyses alongside this panel,
without recomputing their definitions here. Do not collapse the indicators into
a single harm score, or classify substitution without the separate semantic and
quality assessment. The current scope supports "no substitution detected within
the measured indicators," never a general safety guarantee.

Human/LLM annotation, blinded validation, disagreement handling and final claim
rules are in [the review plan](../../docs/plans/off-target-human-llm-review.md).

Run the CPU controls with:

```bash
../.venv-test/bin/python -m pytest tests/test_offtarget_analysis.py -q
```

The fixed 2026-10-15 saved-data regression check is reproducible with:

```bash
../.venv-test/bin/python -m offtarget.reconcile_saved \
  --output runs/offtarget-reconciliation-new
```

It requires the 13 original local evaluation archives and seven prior audit
files named in `reconcile_saved.py`. It checks per-metric agreement, records
compared/agreed/disagreed counts and unknown coverage, and writes the five paired
reports plus the three additional E0 single-arm reports. It fails on missing
archives, disagreements or coverage regressions. The source script and the
`offtarget_v1_validation_findings.md` write-up are eligible for version control.
Generated reports, receipts, validation JSON and original trajectories follow
the repository's existing local run-archive policy; their hashes are anchored in
that write-up. Reproduction on another checkout requires those archives.
