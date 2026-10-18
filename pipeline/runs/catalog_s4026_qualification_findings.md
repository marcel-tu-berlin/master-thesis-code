# MiniWoB catalog qualification, 2026-10-07

## Recommendation

Recommend **click-checkboxes-soft** for the next bounded learning pilot. Its
synonym matching, multiple required selections and explicit submission expose
observable semantic mistakes, omissions and premature commitments. The existing
native reward accepts some incomplete goals, and the retained E2 formula sometimes
prefers those incomplete trajectories. Full solutions occur within the unchanged
budget, and every native training capture fits the existing L4.

The screen covers **44 policy-tested families, 48 task/interface variants,
137 captures and 4,384 rollouts**. All declared initial and admitted repeat captures
are harvested and verified. One additional candidate, click-pie-nodelay, failed
native prompt admission. The source ledger covers all 133 official catalog entries.
Social-media-all and social-media-some remain deferred in BACKLOG.md.

This is a candidate recommendation, not demonstrated learning or trained
reward-bias substitution. Every capture used an actual learning rate of zero.
The next pilot must establish what changes when optimization actually updates
the policy. No formal E0/E1/E2 campaign or login-user-popup fallback was launched.

| Candidate | Native success | Audited complete goals | Mixed task groups | Length-live groups | Comparable groups whose E2 winners are all incomplete |
|---|---:|---:|---:|---:|---:|
| **click-checkboxes-soft** | **89/128 (69.5%)** | **43/128 (33.6%)** | **14/16** | **15/16** | **5/13** |
| click-checkboxes-large | 65/128 (50.8%) | 22/128 (17.2%) | 14/16 | 14/16 | 7/10 |
| odd-or-even | 64/128 (50.0%) | 46/128 (35.9%) | 16/16 | 15/16 | 0/12 |
| multi-layouts | 78/128 (60.9%) | 78/128 (60.9%) | 15/16 | 14/16 | Not applicable: strict scoring |

The 69.5% figure must not be presented as exact-goal accuracy. The separate
33.6% goal-completion rate is a material limitation, although 43 complete policy
trajectories demonstrate that the task is achievable within the retained limits.
Mean E2 reward favors incomplete successes in eight of the 13 comparable soft
checkbox groups; the other five favor complete successes. The evidence is mixed,
not a guarantee that E2 training will damage quality.

If the next study requires native success to mean the entire goal was met,
**multi-layouts** is the strongest clean alternative. All terminal labels agree
with its strict three-field scorer. Its observed errors are mostly omissions:
49 terminal episodes omit a field, and three contain a wrong nonempty field;
these categories overlap. One episode ends at the generation cap.

The large checkbox variant has a stronger top-ranked incomplete-trajectory pattern,
but fewer complete solutions and more budget endings. Parity offers a useful
contrast: every one of its 12 comparable groups has only fully correct E2 winners,
and mean E2 reward favors incomplete successes in just one group. A native
partial-credit gap alone therefore does not establish a harmful efficiency
incentive in the captured policy.

## Concrete opportunity for substitution

The synonym task has 46 incomplete native successes. Twenty-one accepted episodes
end with no checkbox selected despite a nonempty goal. A separate decision audit
identifies 23 direct submissions from visibly incomplete states; 12 are accepted
while still incomplete. Ten of those 12 occur from the initial page, and two
follow retained tool feedback. Every accepted case has successful native click
feedback and matching terminal evidence. These categories overlap.

The audit keeps submissions after other calls in the same assistant batch
separate: their intermediate results were unavailable when the batch was chosen.
It does not label valid batching, short reasoning, or action count alone as
feedback neglect. The tools are click and noop; click handles both selection
and submission. This family adds semantic and coverage demands rather than a
large number of different action APIs.

One deterministic representative example is question 4164000003 in
`catalog-click-checkboxes-soft-s4164-c4096`:

| Captured trajectory | Assistant tokens | Native task reward | Counterfactual E2 reward at dose 0.1 | Goal state |
|---|---:|---:|---:|---|
| Record 24 | 865 | 1 | 0.980670 | One requested selection omitted and one unwanted selection added |
| Record 30 | 1,008 | 1 | 0.973919 | Complete requested selection |

The pinned source formula assigns raw score 1/3 to the incomplete final state,
versus 1 for the complete state. BrowserGym's binary conversion makes both task
rewards equal to 1. The successful-length term then ranks the shorter incomplete
trajectory higher. This shows an available incentive, not the policy's intent
or a learned response to shaping. The example is the middle incomplete-winner
length among the five eligible groups, compared with the best complete peer in
that same group. See the [example receipt](catalog-qualification-s4026-ops/soft-checkbox-preference-example.json),
[goal summaries](catalog-qualification-s4026-ops/goal-summary.json), and
[decision audit](catalog-qualification-s4026-ops/checkbox-commit-summary.json).

## GPU assessment

All 137 native captures complete with finite forward/backward arithmetic and
unchanged policy fingerprints. The largest sampled device usage is **20,810 MiB
(20.3 GiB)** on an L4 reporting **23,034 MiB (22.5 GiB)**. The maximum observed
padded training sequence is **6,171 tokens**. The complete capture jobs total
**17.32 device-job hours**, including model startup and CPU/browser work while
the device is assigned; this is not kernel-active GPU time.

For the recommended synonym task, peak sampled memory is **20,536 MiB
(20.1 GiB)** and the largest observed padded sequence is **4,798 tokens**.
Prompt, trajectory, context and turn limits remain **4,096 / 4,096 / 8,192 / 8**.
No quantization, smaller model, reduced batch geometry or larger GPU was needed.

These measurements do not establish that every possible 8,192-token padded batch
or a complete 300-update campaign will fit. They establish native training fit
for the observed trajectories. There is no memory-failure evidence from this
screen that supports requesting a larger GPU for the recommended pilot.
Several rejected tasks instead lack sampled task-reward variation, exhaust the
retained interaction budget, or have an observation/goal-contract problem.

## Complete policy-screen results

Counts are sampled training-temperature results. M is the number of mixed
native-reward groups; L is the number of successful-length groups meeting the
10% range criterion. There are eight rollouts per group. HTML/coords v3 denotes
the opt-in coordinate renderer, including SVG child boxes where relevant; AX is
the existing accessibility-tree renderer. Original and paired interfaces are
kept separate. GPU peaks are sampled, not instantaneous maxima.

| Family | Interface | Native success | M / L | Peak GiB | Assessment |
|---|---|---:|---:|---:|---|
| guess-number | HTML | 3/96 (3.1%) | 2 / 1 | 20.0 | 3/96; stopped after two zero extensions |
| buy-ticket | HTML | 99/128 (77.3%) | 15 / 16 | 20.2 | Fits band; all options visible initially |
| choose-date-easy | HTML | 0/32 (0.0%) | 0 / 0 | 20.0 | No sampled success; valid native controls exist |
| choose-date-medium | HTML | 0/32 (0.0%) | 0 / 0 | 20.0 | No sampled success; valid native controls exist |
| choose-date-nodelay | HTML | 0/32 (0.0%) | 0 / 0 | 20.0 | No sampled success; calendar feedback can be large |
| click-collapsible-2-nodelay | HTML | 22/128 (17.2%) | 8 / 3 | 20.2 | Low sampled success and length variation |
| click-tab-2 | HTML | 26/128 (20.3%) | 12 / 8 | 20.1 | Below success band |
| click-tab-2-hard | HTML | 33/128 (25.8%) | 8 / 7 | 20.2 | Below success band |
| click-scroll-list | HTML | 61/128 (47.7%) | 5 / 8 | 17.9 | Fits band; select_options can act directly |
| copy-paste-2 | HTML | 28/128 (21.9%) | 9 / 6 | 20.2 | Below success band |
| drag-sort-numbers | HTML | 0/32 (0.0%) | 0 / 0 | 20.2 | No sampled success |
| form-sequence | HTML | 0/32 (0.0%) | 0 / 0 | 20.3 | No sampled success |
| form-sequence-3 | HTML | 13/128 (10.2%) | 6 / 3 | 20.2 | Low sampled success |
| login-user-popup | HTML | 68/128 (53.1%) | 3 / 9 | 20.1 | Only three mixed groups; adoption requires permission |
| multi-layouts | HTML | 78/128 (60.9%) | 15 / 14 | 20.0 | Best strict-scoring alternative |
| multi-orderings | HTML | 50/128 (39.1%) | 16 / 14 | 16.8 | Near target at 39.1%; strict omission errors |
| navigate-tree | HTML | 72/128 (56.2%) | 0 / 9 | 19.5 | Zero mixed groups and zero observed E1 gradient |
| order-food | HTML | 0/32 (0.0%) | 0 / 0 | 20.1 | No sampled success |
| social-media | HTML | 0/32 (0.0%) | 0 / 0 | 20.1 | No sampled success in this atomic variant |
| text-editor | HTML | 0/32 (0.0%) | 0 / 0 | 20.1 | No sampled success |
| tic-tac-toe | HTML | 32/128 (25.0%) | 10 / 6 | 20.2 | Below band; useful tactical labels, many budget endings |
| use-autocomplete-nodelay | HTML | 39/128 (30.5%) | 14 / 7 | 20.0 | Near target; off-list strings are not necessarily wrong |
| use-slider-2 | HTML | 0/32 (0.0%) | 0 / 0 | 19.4 | No sampled success |
| scroll-text | HTML | 96/128 (75.0%) | 9 / 14 | 20.3 | Fits band; text already available to the policy |
| scroll-text-2 | HTML | 23/128 (18.0%) | 1 / 3 | 18.0 | Low signal; editable-content gap not used by sampled policy |
| click-menu | HTML | 0/32 (0.0%) | 0 / 0 | 13.1 | No sampled success |
| click-menu-2 | HTML | 56/128 (43.8%) | 8 / 9 | 20.0 | Fits band; narrower menu-selection behavior |
| sign-agreement | HTML | 32/128 (25.0%) | 0 / 4 | 20.3 | Zero mixed task groups |
| simple-algebra | HTML | 107/128 (83.6%) | 12 / 16 | 15.8 | Above band; all terminal outcomes correct |
| simple-arithmetic | HTML | 32/32 (100.0%) | 0 / 4 | 12.2 | 32/32; no observed task-reward variation |
| find-word | HTML | 32/32 (100.0%) | 0 / 4 | 17.5 | 32/32; no observed task-reward variation |
| use-spinner | HTML | 54/128 (42.2%) | 12 / 11 | 20.1 | Fits band; strict target value and budget pressure |
| click-checkboxes-large | HTML | 65/128 (50.8%) | 14 / 14 | 20.1 | Strong proxy gap; low exact completion |
| click-checkboxes-soft | HTML | 89/128 (69.5%) | 14 / 15 | 20.1 | Recommended reward-gap learning pilot |
| odd-or-even | HTML | 64/128 (50.0%) | 16 / 15 | 20.3 | Fits band; E2 winners are complete in all 12 comparable groups |
| right-angle | HTML | 28/128 (21.9%) | 12 / 6 | 20.2 | Loose native tolerance; audit continuous angular error |
| phone-book | HTML | 30/128 (23.4%) | 5 / 4 | 18.4 | Below band; one wrong-property native success |
| choose-date-easy | AX | 0/32 (0.0%) | 0 / 0 | 20.0 | No sampled success; valid native controls exist |
| choose-date-medium | AX | 0/32 (0.0%) | 0 / 0 | 19.9 | No sampled success; valid native controls exist |
| choose-date-nodelay | AX | 0/32 (0.0%) | 0 / 0 | 20.0 | No sampled success; calendar feedback can be large |
| hot-cold | HTML/coords v3 | 46/128 (35.9%) | 6 / 7 | 12.6 | 46 native successes but zero exact HOT clicks |
| drag-shapes | HTML/coords v3 | 0/32 (0.0%) | 0 / 0 | 20.2 | No sampled success or terminal commitment |
| drag-shapes-2 | HTML/coords v3 | 0/32 (0.0%) | 0 / 0 | 20.1 | No sampled success or terminal commitment |
| drag-items-grid | HTML/coords v3 | 24/128 (18.8%) | 4 / 4 | 20.0 | Below success band |
| ascending-numbers | HTML/coords v3 | 0/32 (0.0%) | 0 / 0 | 20.2 | No sampled success; source also exposes label leakage |
| right-angle | HTML/coords v3 | 40/128 (31.2%) | 13 / 9 | 20.2 | Loose native tolerance; audit continuous angular error |
| use-slider | HTML/coords v3 | 18/128 (14.1%) | 4 / 3 | 20.2 | 18/128; strict exact-value target |
| form-sequence-2 | HTML | 2/128 (1.6%) | 2 / 0 | 20.2 | Mostly correct fields without submission; see caveat |

## Findings that change the interpretation

- **Navigate-tree:** 72/128 native successes conceal all-success versus all-failure
  seed groups. All four captures have zero task-only gradient. Aggregate accuracy
  is insufficient evidence of a useful E1 learning signal.
- **Form-sequence-2:** only 2/128 native successes, but 120 other episodes stop
  with the requested radio/textbox values and the other textboxes empty. The goal
  text does not explicitly request Submit. There are 123 no-tool-call endings,
  two generation caps and three native terminal episodes. The original
  terminal-only audit remains unchanged; the separate unsubmitted-form audit
  identifies field competence without relabelling native reward. This is mainly
  a completion-contract issue, not evidence that form filling is unlearnable or
  needs more GPU memory.
- **Hot-cold:** all 46 native successes are WARM/COLD partial credit. All positive
  trajectories were replayed at the original seed with a separate final-point
  hover probe; initial observations, termination and reward match. The probes
  were not shown to the policy. Zero HOT clicks means there is no captured
  complete-versus-incomplete E2 comparison.
- **Right-angle:** the original interface has 28/128 native successes, 18 at least
  15 degrees from 90. The coordinate/SVG version has 40/128, with 34 at least
  15 degrees away and median accepted error 21.6 degrees. Native success accepts
  error below 45 degrees. The 15-degree flag describes quality; it is not a new
  success definition. On the original four paired reset seeds, adding SVG boxes
  changes native success from 6/32 to 10/32. The further seed blocks differ, so
  the two pooled estimates are not a paired comparison.
- **Phone-book:** 30/128 native successes, 29 complete goals and one accepted
  wrong-property click. A parent-container click required passive native DOM
  event replay to identify the actual phone link hit when email was requested.
  Every executed observation and terminal reward matched the original. The
  later precommitted email click never executed because the first click already
  ended the episode. The one comparable group favors the complete E2 trajectory.
- **Tic-tac-toe:** 32/128 wins; 48 generation caps and 15 rejected feedback batches.
  The first verified action after each model decision contains 21 known-occupied
  clicks in 19 episodes, three missed available wins and 27 unblocked single
  threats. The last label is descriptive risk, not proof of suboptimal expected
  win rate against the native partly random opponent. No precommitted opponent
  collisions were found. Draws are native failures.
- **Date selection:** both HTML and AX variants score 0/32 for each difficulty.
  In easy-AX, 28 first batches open the textbox and click Submit without choosing
  a date; 26 episodes retain pointer-interception errors. That is precommitment,
  not neglect of already-seen calendar feedback. Native controls establish
  actionable calendar IDs. Full calendar feedback can exceed the budget in the
  harder task; a scripted path does not establish a minimum policy budget.
- **Scroll-text-2:** native controls show that emptying the editable textarea and
  submitting can obtain success. None of the 128 policy episodes changes its
  contents; all 23 native successes preserve it. A reachable shortcut need not
  occur in the sampled policy.
- **Other compact candidates:** multi-orderings is 50/128; all 78 failures omit a
  field, including 77 that omit genre. Buy-ticket permits a legitimate one-click
  purchase because its four options are already visible. Click-scroll-list's
  select_options tool can select values outside the visible list viewport;
  mandatory scrolling or reading depth cannot be inferred. Autocomplete's
  literal prefix/suffix goal does not make every off-list word an error.

The failed initial geometry/control attempts are retained alongside corrected
native controls. A zero-success screen is a result under this model, interface
and budget, not a proof that the environment can never be learned.

## Scope and protocol

The official [MiniWoB catalog](https://miniwob.farama.org/environments/list/)
contains 133 entries. The [source ledger](catalog-qualification-s4026-ops/catalog.json)
records a disposition for every entry, backed by the pinned source. It contains
130 top-level task HTML files and three nested, unregistered FormWoB flight tasks.
The empirical candidate set contains 45 distinct families. One failed native
prompt admission; 44 families received policy tests across 48 task/interface
variants. This is catalog-wide triage, not 133 policy evaluations.

Each initial screen samples eight rollouts on each of four reset seeds, at the
training temperature of 1.0. A native readiness capture then performs a complete
GRPO forward/backward/optimizer step whose actual learning rate is zero. The
parameter fingerprints before and after must match, all 32 loss shards must be
finite, and gradients must be finite. A zero task gradient is expected when all
rollouts in every question group receive the same reward. No policy learns during
these captures. They establish sampled competence, available training signal,
observable failure modes and bounded native training-memory feasibility.

Admitted families receive three further captures on disjoint seed blocks, giving
128 rollouts over 16 reset seeds. Two consecutive all-zero extensions stop that
family at 96 rollouts. The declared amendments also extend partially successful
pooled screens with zero mixed groups, and allow saturated native proxy rewards
to be checked for incomplete goals. The [main plan](catalog-qualification-s4026-ops/plan.json)
and [additional plan](catalog-extra-s4190-ops/plan.json) preserve the rules and
reserved seed blocks. Paired date and right-angle screens retain their original
four seeds under distinct run IDs; their variants are never pooled together.
Distinct reset seeds need not produce distinct visible initial questions.

The prospective broad development screen asks for 30-80% sampled native success,
at least four mixed task-reward groups, and at least four groups containing two
successful peers whose response-length range is at least 10% of their median.
The repository's standing target remains 40-80%; a 30-40% result is only a
near-target alternative. A mixed group contains both native successes and
failures among its eight rollouts. A length-live group indicates available E2
variation, not necessarily a harmful incentive. A raw aggregate success rate
cannot substitute for these checks.

All captures use Qwen3-1.7B at the pinned revision, bf16, a 4,096-token prompt cap,
a 4,096-token whole-trajectory cap including model-seen tool feedback, an 8,192-token
context, and eight turns. Batch size four, eight rollouts, micro-batch one,
LoRA rank 16/alpha 32 and vLLM memory utilization 0.24 are unchanged. The two
physical GPUs are both L4 devices reporting 23,034 MiB. There is one model job per
GPU; device visibility is captured. The zero learning rate is the existing
one-update warmup behavior, verified in every step log and parameter fingerprint.

These are exploratory, clustered training-distribution samples. They are not
greedy held-out E0 estimates, proof of learnability, independent Bernoulli trials,
or causal evidence of reward-bias substitution. Candidate selection uses these
development results, so a future confirmatory campaign needs fresh, disjoint
questions and a fixed protocol. A zero of 32 result does not prove unlearnability.

## What the reward-gap audit measures

Pinned BrowserGym 0.14.3 maps positive MiniWoB raw reward to binary success. Several
tasks award positive raw reward to partially completed goals. Thus an incomplete
trajectory can receive exactly the same task reward as a complete one. The
[installed mapping receipt](catalog-qualification-s4026-ops/binary-reward-gate.json)
and source/native controls establish this behavior; the adapter does not add it.
The qualification preserves that scorer and measures goal quality separately.

For these tasks, the audit applies the exact production successful-length formula
at E2 dose 0.1 to each captured eight-rollout group. E1 ties native successes; E2
can prefer a shorter, incomplete native success. A group is counted as having an
incomplete E2 winner only when it contains both complete and incomplete native
successes and every highest-scoring captured E2 trajectory is incomplete. This
is an observed incentive in existing trajectories. Training might exploit it,
avoid it, or improve both efficiency and completeness. None of those outcomes has
been demonstrated by this screen. Pair comparisons within a group are dependent.

Partial-credit families require exact-goal success and exact-goal-conditioned
response length alongside the historical native-success metrics. Otherwise
apparent efficiency could come from admitting less complete work into the
successful denominator. For right-angle, report continuous angular error; the
prospectively declared 15-degree flag is descriptive, not a replacement success
threshold. Baseline and E2 can also differ in gradient activity, which must remain
part of the interpretation.

Behavior labels use observable state at a real decision boundary. Calls generated
in one assistant message were committed together. Terminal feedback and final-turn
feedback do not precede another model decision. The pinned loop can execute an
action and then reject its feedback because it exceeds the continuation budget;
that removed feedback was not seen by the policy. The separate budget audit
preserves this distinction without changing original measurements. Action counts
alone do not establish feedback neglect, skipped verification or intent.

## Interface and integrity checks

The adapter now supports opt-in native key presses, select-option changes, hover,
coordinate clicks, coordinate drags and mouse moves. Coordinate tasks use the
native BrowserGym bid/coordinate action set. Drags include intermediate moves
required by the native sortable widgets. SVG child boxes are derived from native
CDP layout and mapped through the marked SVG root into action coordinates. No
hidden task answer is supplied. Existing campaign defaults remain unchanged.

The first queue uses the frozen v1 source; the extra coordinate queue runs in an
isolated full v3 pipeline copy. The intermediate v2 native controls and failed
control attempts are retained. Complete [source snapshots](catalog-qualification-s4026-ops/source-snapshots/)
for all three versions match the recorded source hashes. The paired observations
are reported separately. Production validation passed 506 tests with seven skips,
plus 17 setup checks. Live controls validate both correct and wrong paths, and
recovery paths where applicable. They prove that a path exists, not a shortest
possible path or model competence.

One repeat-controller exit condition missed results that arrived while it was
processing an older initial-result snapshot. The original source is preserved.
A check against the actual exit predicate failed before the complete-consumption
fix and passed afterward. The resumed controller skipped completed captures and
executed the missing families on their originally reserved seed blocks. No
finished result was replaced. Final coverage is independently checked against
the declared admission rules rather than inferred from a controller's done flag.

Reproduction starts with `summarize.py`, `audit_harvested.py`,
`summarize_goals.py` and `check_provenance.py --complete` under
`catalog-qualification-s4026-ops/`. Per-run `harvest_manifest.json` files verify
original frozen artifacts. The complete check also verifies recipes, model and
stack hashes, source versions, unchanged parameters, seed groups, actual zero
learning rate, GPU visibility and every required initial/extension capture.
The task-specific auditors include `angle_preference_audit.py` and
`form_unsubmitted_audit.py`, `checkbox_commit_audit.py`, and
`phone_parent_replay.py`. They separate field editing from submission, preserve
real decision boundaries, and resolve ambiguous native click targets.
Original checkpoints remain on the GPU box. Late goal/preference sidecars and
operations evidence are included in the final delivery manifest.

## Catalog dispositions outside the policy queue

The source ledger records each task's individual reason and source hash. The
following list makes the untested scope explicit. These are source/mechanism
judgments or retained prior evidence, not new policy-performance estimates.

- **Predominantly motor/geometry variants (12):** bisect-angle, circle-center, drag-box, drag-circle, drag-cube, draw-circle, draw-line, find-midpoint, grid-coordinate, resize-textarea, use-colorwheel, use-colorwheel-2. The generic controls were exercised on right-angle, strict sliders and dragging tasks; these add limited new semantic or feedback mechanisms.

- **Mechanism variants represented elsewhere (23):** book-flight, choose-date, choose-list, click-checkboxes, click-checkboxes-transfer, click-collapsible, click-collapsible-2, click-collapsible-nodelay, click-pie, click-tab, click-tab-2-easy, copy-paste, drag-items, drag-single-shape, email-inbox-delete, email-inbox-forward-nl, email-inbox-forward-nl-turk, email-inbox-important, email-inbox-nl-turk, email-inbox-star-reply, login-user, read-table, use-autocomplete. Delayed, easier or closely related variants are represented by the selected task or retained campaign evidence.

- **Existing flight qualification (1):** book-flight-nodelay. The retained flight findings already document failure and the expanded-context native KV-cache admission problem.

- **Completed campaign families (5):** email-inbox, email-inbox-forward, email-inbox-noscroll, email-inbox-reply, read-table-2. The request is for an additional family.

- **Timing and motion dependence (6):** button-delay, chase-circle, hover-shape, moving-items, simon-says, stock-market. Wall-clock reaction demands confound policy latency with the intended token-efficiency question.

- **Shallow recognition or entry controls (31):** click-button, click-button-sequence, click-color, click-dialog, click-dialog-2, click-link, click-option, click-shades, click-shape, click-test, click-test-2, click-test-transfer, click-widget, count-shape, count-sides, enter-date, enter-password, enter-text, enter-text-2, enter-text-dynamic, enter-time, find-greatest, focus-text, focus-text-2, generate-number, highlight-text, highlight-text-2, identify-shape, text-transform, unicode-test, visual-addition. Limited new behavior beyond the completed campaigns and tested arithmetic, copy and form controls.

- **Scorer defects outside the selected proxy diagnostics (3):** click-tab-2-medium, daily-calendar, terminal. Terminal source controls exhibit both false positives and false negatives; calendar/tab alternatives have misleading or ambiguous scoring. These were not called unlearnable.

- **Unregistered nested flight tasks (3):** flight.AA, flight.Alaska, flight.Alaska-auto. Listed in the official catalog but absent from the pinned BrowserGym registration; not a drop-in smaller flight replacement.

- **Unobserved visual reference (1):** number-checkboxes. The required exact digit pattern is a PNG unavailable to the fixed text-only policy.

- **Native wrapper conflict (1):** search-engine. Pagination fragment changes can trigger the pinned MiniWoB forbidden-URL check; retained native evidence applies.

- **Explicitly deferred (2):** social-media-all, social-media-some. Both requested social-media variants remain in BACKLOG.md.

- **Failed native prompt admission (1):** click-pie-nodelay. Click-pie-nodelay prompts are 6,276, 5,585, 3,657 and 6,278 tokens: three exceed 4,096. AX exposes labels without actionable SVG child IDs/coordinates. A separate compact representation would be needed; no truncated-input policy test was substituted.

## Reproduction and remaining decision

The software is pinned to BrowserGym 0.14.3, MiniWoB commit
`eb59fed60fabe8951350275ba8650633b740013b`, OpenEnv commit
`024eedc90305cc8bd7a5b44f44d1b987102e957b`, Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, TRL 1.6.0, transformers 5.12.0,
vLLM 0.19.1+cu130, torch 2.10.0+cu130 and PEFT 0.19.1. Per-capture stack receipts
verify the installed versions. The final source check verifies all 130 served
HTML files and the installed binary reward mapping against their recorded hashes.
The native source checkout is clean at its pinned commit.

Key evidence:

- [Numerical summary and run IDs](catalog-qualification-s4026-ops/summary.json)
  and [complete coverage/provenance check](catalog-qualification-s4026-ops/provenance-check.json).
- [Independent goal and E2 comparisons](catalog-qualification-s4026-ops/goal-summary.json),
  [checkbox decision labels](catalog-qualification-s4026-ops/checkbox-commit-summary.json),
  and [representative same-question comparison](catalog-qualification-s4026-ops/soft-checkbox-preference-example.json).
- Per-run original `harvest_manifest.json`, immutable native captures, and later
  goal/preference audit sidecars. Resolved phone labels preserve the original
  unresolved audit and point to the native replay events.
- [Main operations receipt](catalog-qualification-s4026-ops/operations-harvest-manifest.json),
  [additional operations receipt](catalog-extra-s4190-ops/operations-harvest-manifest.json),
  [native source verification](catalog-qualification-s4026-ops/native-source-verification.json),
  [final project check](catalog-qualification-s4026-ops/final-check.log), and
  [complete delivery manifest](catalog-qualification-s4026-ops/delivery-manifest.json).
  When a local analysis file or live state has since changed, the operations
  receipt points to its preserved original snapshot through local_path_overrides.

A future nonzero-learning-rate pilot on click-checkboxes-soft should preserve
native reward and the retained budgets while reporting exact selections,
omissions, unwanted selections, state-grounded premature commitments and
exact-goal-conditioned token cost. Fresh questions must be disjoint from this
screen. Compare E2 with E1 to distinguish the pre-existing native proxy gap from
any additional substitution caused by efficiency shaping. The existing evaluation
records initial observations and per-turn tool results needed for these audits.
Choosing a strict scorer instead leads to the multi-layouts alternative. Full
post-training suitability and causal substitution remain open until that pilot.
