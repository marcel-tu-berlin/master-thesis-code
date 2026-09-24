# Family suitability screen - 2026-09-17

Follow-up: the user selected the recommended read-table-2 repair. Decision 0015
and [fresh requalification findings](feedback_requalification_findings.md)
track that work. This report retains the interrupted screen's original verdict
and evidence under the old feedback protocol.

The follow-up also found three keepalive-error messages in s4004 slot 7 that
training had treated as policy feedback. The pooled rates below describe the
preserved artifacts and must not be interpreted as a clean model-difficulty
estimate. The fresh screen follows both the feedback and transport repairs.

## Verdict

**No shortlisted family is qualified for the next E1 pilot.** The read-table-2
screen stopped on a confirmed action-feedback defect after twelve of the
sixteen planned questions. Its task reward is consistent with visible field
values, and it has sampled reward and cost variation, but native action errors
are hidden from the policy. This is an interface qualification failure, not a
verdict that the family is intrinsically too hard or unsuitable for learning.

The [declared plan](../../docs/plans/family-suitability.md) requires stopping on
a technical invariant failure. The fourth batch, the short E1 diagnostic and
the full pilot were not launched. No reward sweep is admitted. All completed
artifacts are harvested; checkpoints remain on the box.

| Candidate | Current decision |
|---|---|
| read-table-2 | Promising action/cost variation, but blocked by missing native action-error feedback. Repair and requalify before judging its model difficulty. |
| click-tab-2-medium | Rejected under the current interface: reward can disagree with the goal; another checked instance hides required action IDs. |
| click-collapsible-2-nodelay | Rejected under the current observation contract: target IDs are absent from accessibility text. |
| search-engine | Rejected on the pinned stack: correct clicks receive zero after URL-fragment validation. |

The other three outcomes and their model-free evidence are owned by
[the oracle findings](family_oracle_findings.md). They were not given model runs.
Agent World Model remains an untested alternative, not a qualified fallback.

## What was measured

Runs `family-screen-table-s4004`, `-s4005` and `-s4006` each contain four distinct
development questions and eight sampled rollouts per question. Each used the
real trainer path at the accepted settings, with a single zero-learning-rate
warmup step. All 392 trainable tensors retained identical before/after hashes
in each run. These are base-policy samples, not learned policies.

| Completed batch | Successful | Wrong terminal | Voluntary stop | Budget exhausted |
|---|---:|---:|---:|---:|
| 4004 | 16 | 2 | 9 | 5 |
| 4005 | 7 | 15 | 5 | 5 |
| 4006 | 7 | 7 | 4 | 14 |
| Total | 30/96 | 24/96 | 18/96 | 24/96 |

Observed success is 31.25%; budget exhaustion is 25%. Descriptive 95% intervals
from 10,000 question-cluster bootstrap resamples (seed 17) are 19.8-43.8% and
13.5-37.5%, respectively. The unit resampled is one of twelve questions, not
one of 96 rollouts. The incomplete screen cannot be scored against the
predeclared sixteen-question gate, and these intervals do not repair the
interface defect or establish post-training headroom.

Ten of twelve groups mix success and failure; eight have at least two correct
rollouts with a length range of at least 10% of their correct-length median.
Ten mix budget exhaustion with other endings. Successful fill/fill/submit paths
exist, including shorter and longer correct paths on the same question. Other
trajectories fill table cells instead of inputs, invent extra fields, overwrite
correct values, click an unrelated element instead of Submit, or stop before
submitting. Budget failures include both no-action reasoning and unfinished
action sequences. These distinguish failure types without equating short
successes with harmful behavior.

The three runs took 1,708.397 seconds (28.5 minutes) in total. All twelve initial
tables and their two requested fields are complete in the captured observations.
All 54 terminal outcomes match the visible final input values. Captured reward
arithmetic, centered advantages, masks, native loss/gradient replay, fixed
dose/placebo replays, source/stack identity and termination checks passed. No
post-completion generation was found. These checks do not cover the missing
action feedback described below.

## Confirmed feedback defect

On the pinned OpenEnv wrapper, ordinary BrowserGym action failures return
`observation.error == ""` and `observation.last_action_error == true`. The
actual message is retained in
`observation.metadata["browsergym_obs"]["last_action_error"]`.
The shared adapter reads only `observation.error`, so both training and
evaluation omit these messages from the policy observation. The wrapper's
exception path does populate `error`, which is why the existing error-feedback
unit test does not catch the normal BrowserGym path.

A real-server replay of development question 4005000001 confirmed:

| Action | Native result | Policy feedback |
|---|---|---|
| Fill the actual color input | No action error; value changes | Updated page |
| Fill a table cell | Element is not an input, textarea or editable element | Unchanged page, no error message |
| Fill nonexistent ID 999 | Could not find element with that ID | Unchanged page, no error message |

A separate correct fill/fill/submit control still earns reward 1. The defect
does not imply incorrect terminal scoring. It does mean that unsuccessful
actions lose feedback that could help the policy recover, and the capture's
count of explicit error strings undercounts native action failures. Do not use
that count as an invalid-action rate. The size of the effect on success or
length is unknown until a corrected, fresh screen is run.

Reproduction and analysis live under `family-screen-review/`:
`check_error_feedback.py`, `family_error_feedback.json`, `review_batch.py`,
`audit_tables.py`, `summarize.py` and `summary.json`. Each sampled run retains
its frozen configuration, admission, native capture/replay and visible-table
audit. The admitted source digest remains
`bccf65f81204384b566fada6813aa9ac0f9dd45be2eb983447157fffbbc41aa3`.
No pipeline source, observation protocol, package or model was changed during
this screen.

The existing local gate still passes (`422 passed, 6 skipped`; formatting,
lint and type checks pass; setup harness `17 passed, 0 failed`). Those checks
do not cover this pinned native error response. The new real-server replay
confirms the missing-feedback contract failure despite that green gate.
The live family admission is now blocked locally and on the box; frozen
admissions within completed runs retain their original launch provenance.

## Recommended choice

Keep read-table-2 as the next candidate. It already has verified solvable paths,
truthful rewards and observable fields; repairing the shared error-feedback
bridge is a smaller change than repairing Search Engine or adding another
environment. First propagate native action errors consistently through the
shared adapter, add a regression check for the actual pinned response shape,
and repeat the affected readiness checks and a fresh declared screen. Preserve
these runs as diagnostic evidence under the old observation protocol, not
matched controls for the corrected one. Keep the recipe, budgets and screening
thresholds fixed.

The user should choose whether to continue with read-table-2, repair/requalify
another shortlisted family, or move to the proposed Agent World Model audit.
No family repair or new model run is queued before that choice.
