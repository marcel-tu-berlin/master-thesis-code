# Read-table-2 diagnostic, seed 4008

Protocol: decision 0016 and the current amendment in
`docs/plans/family-suitability.md`. The user selected read-table-2 and authorized
lowering the exploratory sampled-success floor to 30% after reviewing the
completed screen. The original 40% failure remains in
`feedback_requalification_findings.md`; it has not been relabelled.

The separate amended admission is `table-diagnostic-s4008-ops/admission.json`.
It binds the reviewed screen, fresh passing E1/E2/E3 readiness, current configs,
protocol and launcher. Its source remains
`89c6c77e1e7d643fd44495dda4c3395342785afb8eaaf61977175ae134dc14ec`
and stack remains
`74570e8f708dece02bd348b714ea0db0bb1b063ab84b5ed44169da3a74697245`.
No model, reward, budget or measurement implementation changed.

The paired E0 reference and E1 endpoint each use 100 `development_table`
episodes, seeds 4008300000-4008300099. E1 uses 30 updates and training offsets
0-499, with observation at 1, 10, 20 and 21-30. It starts only after the E0
technical review, reads a frozen copy of that reference from its own run, and
evaluates step 30 in a fresh process. The original post-training feasibility
criteria and single bounded paired extension remain in force.

Both configs validate and the native E0 preflight passes. The full local gate
passes (`450 passed, 6 skipped`; setup `17 passed, 0 failed`). The launch-guard
regression reproduced admission of a failed E0 with a stale pass review; the
corrected guard rejects that case and changed exit evidence while admitting a
completed, reviewed reference. E1 still refuses launch before its E0 review.
Code review's remaining request was to make that CPU check directly runnable;
it now sets its pipeline import path and passes without a PYTHONPATH override.
No launch-safety findings remain open.

The automatic handoff was exercised through launchd and a separate CLI review
thread while the desktop writer stayed open. It returned a review successfully.
The scheduled `com.openai.codex.table-diagnostic-review` job uses that separate
thread path, checks completion every fifteen minutes and claims each E0/E1 run
once. It defers while the original research thread is active. Results and job
status are under `/private/tmp/table-diagnostic-review-claims/<run>/`; inspect
them before starting a duplicate manual review. The Mac must be awake and able
to reach the box. The next long pilot needs its own completion trigger.

## E0 review

E0 completed on 2026-09-17 at 21:59 UTC in about 126 minutes. All eleven
harvested files match the remote SHA256 hashes. The offline audit
`table-diagnostic-s4008-ops/review_e0.py` passes on all 100 distinct declared
seeds. All initial tables and requested input fields are visible; all 42
terminal outcomes agree with the final visible field values. Action counters,
assistant-token totals, stop reasons, report samples and recomputed metrics
agree. No post-submit generation/actions or transport failures were found.
Frozen configuration, source and runtime stack match the admitted inputs.
The source-bound `review_gate.json` admits E1.

Greedy E0 solves 22/100 (95% Wilson interval 15.0-31.1%). There are 20 wrong
terminal submissions, 52 generation-budget endings and six voluntary stops.
Correct episodes average 1886.5 model tokens. The frozen reference thresholds
are 1209.8 and 4096 tokens. These are descriptive reference results, not a
second application of the sampled screening floor. They do not establish
learnability or post-training suitability; the planned E1 diagnostic tests
that question without changing its protocol.

The user reaffirmed automatic harvest, review and progression through passing
authorized stages on 2026-09-18 (local date). Routine completion needs no new
approval. Stop for a failed gate or a new finding that requires changing the
agreed scientific plan. Read `RUNNING.md` for live state.

E1 was admitted and launched at 22:11 UTC on 2026-09-17. Startup verification
confirmed the training process on physical GPU 1 and live environment sessions.
The unchanged launch guards pass and the full local gate remains green.
The scheduled review routing check passes for both completion and failure,
defers during live training/evaluation or an active research turn, and rejects
duplicate claims. E1 evaluates automatically after training; its completed
review conditionally admits the declared seed-4009 pilot. The Mac must remain
awake with SSH access for the completion review to begin.

## E1 diagnostic review

E1 completed training at 02:36 UTC and its fresh-process evaluation at 05:12 UTC
on 2026-09-18. The harvested evidence matches the box SHA256 values. Both exit
records report return code 0, the final adapter is present, and the admitted
source and runtime-stack digests remain
`89c6c77e1e7d643fd44495dda4c3395342785afb8eaaf61977175ae134dc14ec` and
`74570e8f708dece02bd348b714ea0db0bb1b063ab84b5ed44169da3a74697245`.

All 30 optimizer updates have finite loss and gradient values. The observer has
the required 13 batches at updates 1, 10, 20, and 21-30, with 416 trajectories.
Their task rewards and composed rewards agree, and the late groups retain
within-question correct-trajectory length variation. At update 30, one shared
question has valid correct controls at 642 and 3940 model tokens: both submit
the same two visible values. The 100 evaluation episodes use exactly seeds
4008300000-4008300099, match the report samples, preserve E0's finite token
thresholds (1209.8 and 4096), and pass the visible-table terminal audit. There
are no transport failures or post-terminal actions or generation.

E1 solves 60/100 (Wilson 95% interval 50.2-69.1%), so the success ceiling
passes. Relative to E0, 44 questions change from wrong to correct, six change
from correct to wrong, and 16 are correct under both policies. For those 16,
the median E1-minus-E0 token change is -292.5. The blinded trajectory audit
uses six budget endings, one available voluntary stop, six invalid-action cases,
and six efficient correct controls; it reports the voluntary-stratum shortage
without replacement. Native action errors are retained as model outcomes.

The E3 rate is 12/100 generation-budget endings, with Wilson 95% interval
7.76-20.98%. Its lower bound is below the fixed 10% requirement, but the
interval spans it. This is the protocol's bounded inconclusive outcome, not a
pass or a failure. The only admitted next stage is the paired 100-question E0
and step-30 E1 extension at offset 300100, using the original E0 thresholds;
pool the fixed 200 afterward. The seed-4009 pilot is not admitted yet.

## Predeclared extension execution

The automatic reviewer completed its review at 05:32 UTC but did not launch
the already-admitted extension. This was a continuation omission, not a new
scientific decision. The main thread resumed the authorized work on 2026-09-18.

The extension uses `table-diagnostic-e0-s4008-extension` followed by
`table-diagnostic-e1-s4008-extension`, exactly 100 episodes each at seeds
4008300100-4008300199. Both retain the original E0 thresholds; the second
loads the existing step-30 adapter. No training, source, stack, recipe or
measurement changes are involved. Separate configs and a hash-bound admission
are frozen before launch. The existing completion watcher now recognizes both
extension IDs and is instructed to launch the next admitted phase in the same
review turn, instead of ending with an unexecuted next step.

The paired 200-question review applies the original fixed gates. No further
extension is allowed. An external-checkpoint report has `checkpoint_step: null`
by design; `checkpoint_origin.json` and the bound adapter hashes preserve its
step-30 identity without changing that measurement API.

The paired extension launched at 08:52 UTC on 2026-09-18. Startup verified
the E0 process on physical GPU 1 with the expected seed base and a live
environment connection; the existing checkpoint evaluation is queued next.
The launch check verifies the unchanged qualified source/stack, parent evidence,
adapter hashes and extension allocation. CPU guard checks reject a failed
parent, output collision and changed config. The completion-route check first
rejected the new extension ID, then passed after the watcher admitted the two
declared IDs. The full local gate passes; no runtime measurement code changed.

The E0 extension completed at 10:56 UTC and the controller automatically
started the saved-checkpoint E1 evaluation. All twelve E0 extension artifacts,
including checkpoint-origin metadata, were harvested with matching remote
SHA256 hashes; all 100 declared extension seeds are present. The paired
scientific verdict awaits the completed E1 extension.

## Paired extension and tested diagnostic verdict

Both extension evaluations completed with return code 0 at 13:38 UTC on
2026-09-18. Their harvested files, including `checkpoint_origin.json`, match
the GPU box SHA256 values. The E0 extension is a base-policy evaluation
(`checkpoint: null`, step 0); the E1 extension names the unchanged
`table-diagnostic-e1-s4008` final adapter and records its owning step as 30.
Both use exactly seeds 4008300100-4008300199, the admitted source and stack,
the original E0 thresholds, and complete 100-episode records matching their
reports. The visible-table audit passes all 200 extension trajectories: tables
and input fields are complete, measurements are finite, terminal rewards agree
with submitted fields, and there are no transport or post-terminal-action
errors. Native action errors remain recorded as policy outcomes.

Pooling the fixed 200 final-E1 episodes gives 118/200 success (Wilson 95%
interval 52.08-65.58%) and 24/200 budget-exhaustion outcomes (8.20-17.23%).
Success headroom therefore passes, but the E3 lower bound remains below the
predeclared 10% floor. This is a failed pooled gate, not an inconclusive result:
the one permitted extension has been used. E2 evidence remains present: the
original observer review found valid shorter within-group correct controls, and
the pooled paired evaluation has 33 both-correct questions with median E1-minus-
E0 model tokens of -129. The paired success table is 64 wrong/wrong, 85
wrong/right, 18 right/wrong, and 33 right/right.

The blinded audit used seed 17 and selected six records from each prescribed
stratum after masking policy/checkpoint labels: budget endings, voluntary stops,
invalid or repeated actions, and lowest-quartile correct controls. Each selected
record supports its assigned behavior from its observation and action trail; no
stratum was short. The diagnostic therefore has a tested family verdict under
the fixed protocol: read-table-2 does not meet the E3 opportunity criterion for
the qualified recipe and budgets. No seed-4009 pilot, reward sweep, extra sample
extension, model change, stack change, or final-test evaluation is admitted.

## Core admission under decision 0017

After reviewing the pooled E3 failure, the user made budget opportunity a
secondary warning and retained learning headroom and compression opportunity
as the core admission criteria. The original failed review and measurements
above remain unchanged. This is an amendment after observing development data,
not a prospective pass under the original rule.

The pooled success interval remains below the 90% ceiling. The sampled screen
has 15 mixed-success groups and 12 groups meeting the correct-length criterion;
33 late diagnostic groups also have at least two correct trajectories whose
length range reaches 10% of their median. Previously reviewed valid shorter
controls remain the behavioral evidence. The E3 lower bound of 8.20% now warns
without blocking. No extra seed-4008 evaluation or runtime change is needed.

The separately bound admission is
`runs/table-pilot-s4009-ops/amended_diagnostic_review.json`. It admits the
fresh seed-4009 E0 reference and, after technical review, the declared
300-update E1 pilot. Actual shaping effects remain untested; the task-only
E0/E1 comparison cannot establish an E2 treatment effect.
