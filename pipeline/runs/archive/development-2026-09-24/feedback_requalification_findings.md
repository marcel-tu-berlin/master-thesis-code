# Native action-feedback requalification

Follow-up: after reviewing the outcome below, the user chose read-table-2 and
authorized lowering the development-screen floor. Decision 0016 admits the
seed-4008 diagnostic under a 30-80% band with all other criteria unchanged.
The original screen failure below and its machine-readable verdict are retained.

Decision: [0015](../../docs/decisions/0015-surface-native-action-errors.md).
Status: repaired setup passed fresh E1/E2/E3 readiness. The complete fresh
read-table-2 screen failed the predeclared base-success floor; no diagnostic
or pilot is admitted. All seven completed runs are harvested. The family is
unqualified under this model/interface/budget, not established as unlearnable.

## Reviewed outcome

The sequence finished on 2026-09-17 at 18:48 UTC. Each readiness arm completed
three updates, changed finite trainable parameters, reloaded its checkpoint and
completed four development evals. Native arithmetic/loss/gradient replay and
the three-batch trajectory audits passed. The latter cover 96 trajectories per
arm. No repeated keepalive-disconnection messages were found. These short runs
qualify execution and measurement, not treatment effects or post-training
family suitability.

The cumulative readiness time, including the failed first attempt, was
7,022.323 seconds against the unchanged 7,200-second ceiling. The four fresh
screen batches took 2,396.082 seconds against their 5,400-second ceiling.

| Fresh screen batch | Correct | Wrong terminal | Voluntary stop | Budget exhausted |
|---|---:|---:|---:|---:|
| 4011 | 12 | 4 | 1 | 15 |
| 4012 | 11 | 11 | 0 | 10 |
| 4013 | 14 | 2 | 2 | 14 |
| 4014 | 9 | 5 | 0 | 18 |
| Total | 46 | 22 | 3 | 57 |

All 128 rollouts come from sixteen distinct development questions. Each batch
used fresh base weights and one zero-learning-rate step; before/after parameter
hashes are identical. No old-interface samples enter this result.

| Predeclared criterion | Observed | Verdict |
|---|---:|---|
| Base success 40-80% | 46/128 = 35.94% | Fail |
| At least four mixed-success groups | 15/16 | Pass |
| E2 length opportunity in at least four groups | 12/16 | Pass |
| At least 10% budget exhaustion | 57/128 = 44.53% | Pass |
| At least two groups with varying budget cost | 16/16 | Pass |

The descriptive question-cluster bootstrap interval is 25.78-46.88% for
success and 33.59-55.47% for budget exhaustion (10,000 resamples, seed 17).
The success interval crosses the 40% floor. The data therefore do not establish
that population success is below 40%, but the predeclared point-estimate gate
still fails. The fixed screen allows no further sample extension.

All 128 initial tables and requested fields are complete. All 68 terminal
rewards agree with visible final input values. Forty-three episodes receive
explicit action-error feedback; fifteen of those subsequently succeed. This
confirms that the repaired feedback path is exercised, without establishing a
causal improvement over the earlier, different samples. Of the 57 budget
failures, 45 have no executed tool action: generation exhausts the budget before
acting. The remaining failure modes are policy behavior under the fixed
interface and budget; the reviewed captures show no new infrastructure defect.

Local review checked source identity, raw outcome totals, the question bootstrap,
unchanged screen parameters, replay tolerances and hashes binding the audits to
their inputs. A checksum-based rsync comparison found no differences after
harvest. Evidence is in `feedback-requalification-v2/screen_summary.json`,
`readiness_feedback_v2.json` and each run's capture/review artifacts.

The other shortlisted families retain the interface/reward failures in
`family_oracle_findings.md`; they have not become qualified alternatives.
Read-table-2 has informative reward groups and both targeted costs, so retaining
it for a separately declared short learnability diagnostic is worth considering.
That would require an explicit protocol decision revising the failed admission
floor. It is not an automatic continuation of this plan. The original failure
must remain reported; neither a long pilot nor a reward sweep is launched.

The GPU controller completed every admitted stage automatically. The scheduled
follow-up review triggered at 18:57 UTC but the CLI could not resume a thread
still owned by the desktop app (`already has an active writer`). Review was
completed in the existing thread on the subsequent status request; the failed
one-shot watcher was unloaded. This affected the review handoff, not the runs.

## Repair and verification record

The shared BrowserGym adapter now exposes ordinary native action errors from
observation metadata. Its existing explicit-error path takes precedence. This
covers click, fill and noop across families, in training and evaluation. The
training/reward/evaluation core does not branch on read-table-2. Readiness now
accepts an explicitly configured family while retaining its recipe and matched
arm guards. Shared config validation also rejects malformed family lists in
training and evaluation overrides; omitted/null defaults remain supported.

The regression tests failed before these fixes and pass after them. The full
local gate passes formatting, lint, types, `443 passed, 6 skipped` and the setup
harness (`17 passed, 0 failed`). Native code review found the malformed-list
gap, which was fixed at the shared validation boundary. Its follow-up requested
explicit null-default compatibility coverage; that coverage now passes in both
training and split overrides.

The pinned-server controls pass on both click-menu-2 and read-table-2: valid
actions keep normal feedback; invalid click/fill actions expose the actual
native message; training and eval tool schemas match; correct and wrong
terminal controls keep rewards 1 and 0; the correct path recovers after an
invalid action; reset determinism and the post-completion guard hold.
Evidence and the runnable control are under `feedback-requalification/`.

The first attempt's native evidence and deployed controller matched source digest
`847f9e9724150b6ae7a4f4cdcd4cf88ec5a87531262a7400d80af4f6884a1782`
and stack digest
`f524a0bb67c23ad83a96527f696a30196a20b38f7de28d7cada270968fa2d940`.
The controller uses passing Mac local-gate evidence only after checking source
and config hashes; native arithmetic replay executes on the box. The canonical
interpreter path is used throughout. Old captures remain unchanged.

The first configs `g3-feedback-e1/e2/e3.yaml` use seed block 4010, three full-geometry
updates, and four development evaluation questions at offset 700000. Readiness
capture and the existing group observer record all three batches. Each arm
requires review before the next starts; the combined ceiling is two hours.
After these pass, configs `family-table-s4011-feedback.yaml` through
`s4014-feedback.yaml` repeat the predeclared sixteen-question screen from fresh
blocks. No earlier-interface samples are pooled into that screen.

## First E1 attempt and transport correction

`readiness-g3-e1-s4010-feedback-v1` stopped after its first optimizer step when
the next reset encountered a closed WebSocket. The first batch already contains
41 keepalive-error tool messages across seven of 32 trajectories (slots 7, 10,
23, 26, 27, 28 and 29). TRL's broad tool-exception handler had turned these
infrastructure failures into policy feedback. Its finite loss/gradient does not
make that batch scientifically valid. The controller failed E1; neither E2/E3
nor any new screen was launched. The failed capture is preserved and its
576.600 seconds count toward the same two-hour requalification ceiling.

The historical `family-screen-table-s4004` capture also contains three such
messages in slot 7. Its pooled screen rates therefore cannot establish base
policy difficulty. No keepalive-error messages were found in the s4005/s4006
captures or the three earlier episode-boundary qualification observers. This
string audit is bounded to those files; it does not certify unrelated campaigns.

Training now matches evaluation: check model arguments against the tool
signature, return invalid-argument/name feedback, and propagate exceptions from
inside a tool. Five exception regressions failed before the correction and pass
after it. Native installed-stack loop tests pass, including unchanged valid-path
token IDs, masks and log probabilities. Completed trajectories and ordinary
BrowserGym action-error feedback retain their definitions.

A model-free replay of all 32 environments, recorded actions, a 45-second idle
wait and a second reset passed. Its server pipe reached only 20,518 of 65,536
bytes, so pipe saturation did not explain that replay. A separate chatty-child
test did reproduce the unread-pipe deadlock; server logs now inherit the runner's
output. This also preserves evidence of later server failures.

A controlled 45-second native call that holds the client process's GIL reproduces
the exact `received 1011 ... keepalive ping timeout` error with the old transport.
The corrected transport survives the same pause, accepts an action and resets
deterministically. The same check passes on Reasoning Gym on configured port
8017, including a wrong terminal answer and repeated reset. This establishes the
latency failure mode; the exact native call that delayed the original model run
was not profiled.

The shared launcher and both clients keep Ping traffic but disable the Pong
deadline. Active request deadlines remain effective: both native clients raised
TimeoutError after about 0.20 seconds against a nonresponding test server with a
0.20-second request limit. Production retains its existing 10-second connection
and 60-second message limits. The keepalive distinction is documented by
[websockets](https://websockets.readthedocs.io/en/16.0/topics/keepalive.html).
No package or OpenEnv pin was changed. Runtime stamps now include uvicorn and
websockets versions.

The full local gate passes (`450 passed, 6 skipped`, plus `17 passed, 0 failed`
in the setup harness); the installed-stack termination suite passes all twelve
tests. Native code review's sole remaining finding compared an indented source
hash against the guard's dedented hash. Calling the actual production guard on
the pinned stack passes; no digest was changed to accommodate that false alarm.
Evidence lives under `feedback-transport-review/`.

Replacement configs `g3-feedback-v2-e1/e2/e3.yaml` preserve the recipe, seeds and
development split. Their source digest is
`89c6c77e1e7d643fd44495dda4c3395342785afb8eaaf61977175ae134dc14ec`.
`readiness_feedback_v2.json` and `feedback-requalification-v2/` own the fresh
admissions and reviews. The bounded continuation checks source, config, stack
and helper hashes, reviews every arm before the next launch, and stops on a
failure. The unused sixteen-question screen still uses blocks 4011-4014.

The replacement E1 started on 2026-09-17 at 16:15 UTC on physical GPU 1. A
completion-triggered review of this same research thread is registered on the
Mac. It inspects the completed screen or a stopped controller, harvests evidence,
and advances to the already-authorized conditional diagnostic only if the
predeclared screen passes. It does not treat the repaired infrastructure or a
readiness pass as proof that the family supports the thesis experiments.
