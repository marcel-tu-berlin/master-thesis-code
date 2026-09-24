# Bounded family suitability check

2026-09-21: the user selected read-table-2 through the first informative E1/E2
result. Decision 0018 supersedes the final-E1 success ceiling. The earlier
failures and declarations below remain historical evidence. Continue under
[the first-contrast plan](read-table-first-contrast.md); no further family screen.

Declared: 2026-09-17, before sampled read-table-2 results. The user authorized
continuing the agreed stages without pausing between passing gates, then
presenting the family verdict and waiting for their family choice.

First execution outcome: the screen stopped after twelve questions on a confirmed
native action-error feedback defect. The sixteen-question statistical gate was
not completed; no E1 diagnostic or pilot is admitted. Evidence, candidate
verdicts and the recommended next choice are in
[`family_screen_findings.md`](../../pipeline/runs/family_screen_findings.md).
The preregistered rules below are retained unchanged.

Authorized restart: the user selected read-table-2 and approved repairing its
feedback path. [Decision 0015](../decisions/0015-surface-native-action-errors.md)
records the observation change and the transport correction found during its
first diagnostic. Fresh readiness uses `g3-feedback-v2-e1/e2/e3.yaml`
and development seed block 4010. After it passes, restart the full screen with
`family-table-s4011-feedback.yaml` through `s4014-feedback.yaml`, sixteen new
questions at offsets 0-3. Earlier samples are not pooled into this screen.
All thresholds, budgets and conditional E1 stages below stay unchanged.

Restart outcome: corrected E1/E2/E3 readiness passed and all sixteen fresh
questions were screened. The base-success point estimate missed the fixed
40% floor; the remaining triage rules passed. The diagnostic and pilot are
not admitted. [Reviewed findings](../../pipeline/runs/feedback_requalification_findings.md)
record the complete evidence and uncertainty. Await the user's family/protocol
decision; do not extend the screen or change the gate retrospectively.

## Success-floor amendment and seed-4008 diagnostic

The user reviewed that outcome and chose to continue with read-table-2.
[Decision 0016](../decisions/0016-development-screen-success-floor.md) changes
the active sampled base-success band to 30-80%, retaining every other screen
criterion. This amendment follows inspection of the development results; the
original 40% failure and original rules below remain historical evidence.
The existing screen admits a diagnostic under the amended rule. No additional
screen samples or readiness reruns are needed: runtime source and recipe stay
at their freshly qualified versions.

Declare before launch:

- `table-diagnostic-e0.yaml`: `table-diagnostic-e0-s4008`, base-policy reference.
- `table-diagnostic-e1.yaml`: `table-diagnostic-e1-s4008`, 30 updates, save at 30,
  500 training questions, task reward only, unchanged accepted recipe.
- Both use `development_table`, 100 greedy episodes, seed block 4008 and offset
  300000. Training offsets 0-499 are disjoint; final-test offsets remain unused.
- E0 completes and passes technical/trajectory review before E1 starts. Freeze
  its report as E1's reference for token thresholds. E1 is observed at updates
  1, 10, 20 and 21-30: thirteen batches, including ten late batches.
- Run E1 evaluation in a fresh process after training, using checkpoint-final
  at step 30. Report paired outcomes on the same 100 questions and apply the
  unchanged post-training criteria below. Do not demand improvement after only
  30 updates or apply the sampled screening floor to greedy E0 evaluation.

Only technical failure stops a phase before its declared endpoint. A completed
diagnostic that fails feasibility stops progression; an inconclusive rate
permits only the existing paired 100-question extension at offset 300100.
The 300-update pilot remains conditional. No reward sweep is authorized here.

ASSUMPTION: the measured readiness mean of about 563 seconds per update gives
about 4.7 hours for 30 training updates. Allow roughly 2-3 hours per 100-episode
greedy evaluation initially; update estimates from observed progress without
changing the fixed horizon or sample count. Use completion-triggered review.

## Active amendment: core efficiency admission

The paired 200-question diagnostic was reviewed under the preceding rules and
failed their E3 lower-bound requirement. That historical verdict is preserved.
After seeing it, the user accepted
[decision 0017](../decisions/0017-budget-exhaustion-is-secondary.md): learning
headroom and an informative compression signal govern admission; budget
exhaustion is a separate secondary diagnostic. The original rules below describe
the earlier declaration wherever they conflict with this amendment.

Keep the 10% budget-exhaustion reference, its Wilson interval and within-group
cost diagnostics. A lower bound below 10% is a warning, never a blocker or a
reason for more evaluation. Technical validity, sampled learning signal,
success headroom and E2 opportunity remain hard requirements. Only an ambiguous
success-headroom criterion can admit the existing single paired extension.

The reviewed diagnostic passes these core criteria. Admit the fresh pilot:

- `table-pilot-e0.yaml`: `table-pilot-e0-s4009`, base-policy reference.
- `table-pilot-e1.yaml`: `table-pilot-e1-s4009`, 300 task-reward updates,
  500 training questions, checkpoints and evaluation at 100/200/300.
- Both use 100 greedy `development_table` questions at offset 300000 of
  seed block 4009, disjoint from training offsets 0-499 and final-test offsets.
- Review E0 before E1; freeze its reference report inside E1. Observe updates
  1, 100, 200 and 291-300. Review step 300 against the core criteria.
- Preserve the qualified runtime, stack, geometry, rewards and trajectory
  budgets. No new readiness pass is needed for an admission-only amendment.

Automatically harvest and review each completed stage, then launch its admitted
successor. After the final pilot verdict, review campaign choices with the user.
E3 stays an optional extra; it does not gate the core E1/E2 research.

## Scope and fixed settings

Only read-table-2 survived the four-family oracle audit. The other three retain
their recorded observation/reward failures; do not spend model time on them with
the same interface. Keep the qualified source, pinned stack, Qwen3-1.7B revision,
4 x 8 geometry, temperature 1, 4096 trajectory tokens, eight assistant turns,
naive_sum, scale_rewards none and env_done_or_budget_v1. Enable the verified fill
tool for read-table-2. Do not shorten budgets to manufacture difficulty.

## Original frozen-policy screen declaration

Use configs `configs/readiness/family-table-s4004.yaml` through `s4007.yaml`.
Each uses four distinct development questions at offsets 0-3 of its own seed
block, with eight sampled rollouts each. First inspect seed 4004 for a valid
training path, complete observations, correct tool arguments and reward/ending
alignment. If technically valid, extend the sole surviving candidate by the
other twelve questions: sixteen independent questions and 128 rollouts total.
This fixed extension avoids declaring the whole family from four question draws.
No final-test offsets are used. Stop on a technical invariant failure.

Reuse one ordinary trainer step with readiness capture per config. Under the
installed constant-with-warmup scheduler, the first step has learning rate zero
when max_steps is one and warmup_ratio is 0.1. Verify unchanged before/after
parameter hashes, the logged zero learning rate, full geometry and finite
captured arithmetic in every screen. These are base-policy samples, not trained
policies or Gate 3 passes. No evaluation is requested during this screen.

Report per-question success fractions and reward/cost variation, plus pooled
counts. Use a question-cluster bootstrap (seed 17) for a descriptive interval;
do not treat the 128 rollouts as 128 independently sampled questions. Inspect
successes and failures against visible tables and input fields. Retain voluntary
stops separately from budget exhaustion. A successful minimal fill/fill/submit
path is an efficient control, not unsupported behavior by definition.

Operational triage rules, fixed before seeing results:

- Base success must be in the standing 40-80% sampled band and at least four
  of sixteen groups must contain both successes and failures.
- E2 opportunity needs at least four groups with two or more correct rollouts
  whose assistant-token range is at least 10% of their median. Report whether
  valid shorter solutions exist; pooled failure lengths cannot satisfy this.
- E3 opportunity needs at least 10% budget-exhausted rollouts and at least two
  groups with varying budget-exhaustion cost. Voluntary stopping does not count.
- Only a candidate meeting all three advances toward the joint E2/E3 campaign.
  Report E2 and E3 limitations separately. A failed screen means unqualified
  under this model/interface/budget, not that the family can never work.

The screen ceiling is 90 minutes of run time. There is no further sample
extension after these sixteen questions; uncertain cases remain unqualified.

## Original conditional E1 diagnostic and final pilot

If the screen passes, declare a fresh seed-4008 E1 diagnostic before launch:
30 updates, a 500-question training pool, the same geometry/recipe, and 100
same-family development evaluation questions at offset 300000. Create the
paired E0 reference first. Observe training groups with the existing observer.
Apply the existing Gate 4 success ceiling, E3 cost and E2 opportunity rules to
the diagnostic endpoint. An inconclusive rate permits only the existing single
100-question extension. No improved success after 30 updates is required.

An early ceiling/floor, absent target cost, or unresolved measurement failure
ends qualification. A diagnostic pass permits one fresh seed-4009, 300-update
pilot with a 500-question pool, scheduled checkpoints 100/200/300, its own paired
E0 calibration reference, 100 same-family development questions, and the same
fixed feasibility thresholds and bounded extension. Declare its concrete split
configuration before launch. Step 300 remains the endpoint; do not substitute an
earlier checkpoint. This final pilot, including the prescribed trajectory audit,
is required before claiming post-training family suitability.

After the tested verdict, present all candidate outcomes and ask the user which
family to continue with. Do not begin a reward sweep, repair a rejected family's
observation/reward contract, change models, or upgrade OpenEnv before that choice.
