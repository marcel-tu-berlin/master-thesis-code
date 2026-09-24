# Gate 3 readiness - E1, E2 and E3

Date: 2026-09-13. Runs: `readiness-g3-e{1,2,3}-s4001-r2`.
These are diagnostic execution and arithmetic checks, not thesis effect estimates.

## Verdict

Gate 3 passed and was reviewed. The final deployed controller returned
`Readiness: pass` and `Next phase: gate4_e1_pilot`, with no contract errors.
All three arms completed their three full-geometry optimizer updates, adapter
save/reload and four-episode evaluation. Native code review found no remaining
actionable issue after the replay provenance fix. Gate 4 has not started.

The three captured runs total 5,372.439 seconds (89.54 minutes), within the
two-hour diagnostic ceiling. The supplemental arithmetic checks run on CPU and
generate no trajectories. Original capture artifacts remain unchanged, and
checkpoints remain on the box.

## E2 capture evidence

- All 32 rewards aligned with their captured environment reward and rollout.
  E2's raw task plus token-length reward reproduced each composed reward, and
  the trainer advantages matched reward minus the eight-rollout group mean.
- The 32 independent DAPO loss-shard recalculations agreed with the trainer.
  Capture records contained all selected-token masks, tool masks, old and
  sampling log probabilities, and importance-sampling ratios.
- The three captured gradients were finite and nonzero. Trainable parameters
  were finite and their before/after fingerprints differed.
- The controller found the frozen config, three completed optimizer steps,
  finite training loss and gradient norms, checkpoint-final, evaluation report,
  four complete episode records with observations and tool feedback, and no
  unexpected controller errors.

## Lifecycle evidence and warnings

The diagnostic evaluation reported 3/4 correct episodes. This confirms the
save/load and evaluation lifecycle only; it does not estimate E2 accuracy or a
reward effect. The run emitted TRL's compatibility warning for installed vLLM
0.19.1+cu130 and a shutdown warning that `destroy_process_group()` was not
called. Neither produced a failed controller invariant, but both remain runtime
warnings to retain with the capture.

The deployed rsynced runtime lacks `.claude/check.sh`, so the controller reused
the fresh matching-source local gate. The final report records that provenance.

## E3 execution and the missing first-batch signal

Run `readiness-g3-e3-s4001-r2` completed three optimizer updates and its
four-episode evaluation without an unexpected runtime error. The second and
third training steps reported non-termination reward means of -0.09375 and
-0.125, so the component became active during training.

The bounded readiness recorder saved only the first 32-rollout batch. That batch
contained 23 completed environments and 9 unfinished trajectories where the
model stopped on its own. All 32 non-termination penalties were therefore zero,
with no within-group variation. An independent replay found no disagreement with
the documented reward rule, but the capture cannot verify E3 reward assignment,
advantages, or loss when the penalty is active.

The original controller nevertheless reported `gate4_e1_pilot`. A regression
check reproduces that false pass; the updated controller requires an installed
trainer replay before admitting another gate. This changes admission checks,
not the training recipe or any reward or evaluation measurement.

## Installed trainer replay

`probes.readiness_replay` uses the installed TRL methods on CPU. Generation
and model forward are fixture boundaries; decoding uses the cached, pinned
Qwen tokenizer. Reward dispatch, advantage
centering, DAPO loss and autograd execute the native trainer code. The gradient
comparison is at selected-token log probabilities; the actual diagnostic runs
supply the separate evidence of finite model gradients and changed parameters.

- Every saved loss shard is checked against an independent scalar DAPO loss
  and analytic derivative, including completion/tool masks and the full-batch
  denominator.
- Captured raw components are replayed at weights 0, 0.5 and 1 and with a
  within-group shuffled assignment. These are arithmetic counterfactuals on
  fixed inputs, not estimates of what another trained policy would do.
- A labelled 4 x 8 E3 fixture exercises completed and failed-completed
  environments, voluntary stopping, turn exhaustion, an undispatched tool call,
  a voluntary stop at 4095 tokens, exhaustion at 4096, and completion at the cap.
  E3 passes at weights 0, 0.5 and 1, with both PPO clipping directions exercised.
- Negative controls detect an all-zero penalty and a change back to group
  reward scaling. The latter cancels the intended dose; `scale_rewards: none`
  remains the accepted recipe.

Declared tolerances are 2e-5 for loss/advantage checks and 5e-7 for the maximum
absolute selected-token gradient error. Each controller invocation recomputes
the evidence and writes `readiness/integration_replay.json` under each run.
Observed maxima across saved batches, dose/placebo replays and scripted endings
were 1.36e-7 for advantages, 5.43e-8 for loss and 4.86e-12 for gradients.
The all-zero first E1/E3 loss batches are supplemented by the active scripted
E3 cases and E2 dose replays; zero-loss agreement alone is not the proof.

## Sampler and token accounting review

The three initial captures have the same generated trajectories, as expected
from identical base weights, task seeds and sampling settings before an update.
The assistant-token loss mask contains 73,865 tokens and exactly matches the
recorded DAPO denominator. Tool responses remain charged to the trajectory
budget while being excluded from the policy loss.

Across those assistant tokens, absolute trainer-versus-sampler log-probability
differences have median 0.000009, P95 0.120480 and P99 0.248429; maximum 2.924984.
Per-episode mean absolute differences range from 0.015443 to 0.021787. Their
correlation with assistant-token length is -0.258, with no positive growth
pattern in this small diagnostic sample. The installed importance-sampling
ceiling is 3.0; no active token reaches it. Recomputed correction factors match
within 1.45e-7. These checks support arithmetic consistency of the captured
batch, not universal numerical equivalence between HF and vLLM.

## Provenance and limits

The initial runs were captured under source digest
`b7f362019a670e561383d292d2b9f7816dfe83e28eec6bd37281c48214707bd2`.
Their compatibility check preserves every training/evaluation dependency and
the recorder AST. Only the assessor function and explicit addition of the
replay module to the source manifest may differ. The assessor pins the reviewed
replay hash, so a later unapproved replay edit also fails. Runtime package,
OpenEnv, platform and executable identity remain exact checks.

The local project gate is executed on the Mac and transferred with its source
digest; the deployed controller rejects it if that digest differs. It runs
with the exact interpreter invocation recorded during training.
The final reviewed source digest is
`9cb8ac80183cb2329b2ec789d61d3349110b8e1af823ed4f2eb6439436a062cd`;
the pinned replay digest is
`4413746a4cea64346b1ea1e2b9b15e11a5a4080bdd508c89ce5f03c6813c619b`.
The full local gate passed. The replay regression tests and the otherwise
skipped parser, callback and adapter checks also passed on the installed box
stack. E3 artifacts and supplemental reports were harvested without checkpoints.
Commit review also caught a replay exception path that could leave an old
controller report after a crash. Native runtime failures, malformed replay
inputs and replay-output I/O failures now produce an explicit failed contract
with the exception type and message. Token arrays are checked for aligned
rows before replay; regression checks cover malformed rows and exception paths.
The replay pin is enforced for both legacy and new captures. A prior replay
report is removed before the new attempt, so a failed replay or output write
does not retain its predecessor's success evidence.

Tiny eval counts were E1 1/4, E2 3/4 and E3 1/4. They prove the lifecycle only.
Three updates also shorten relative warmup and cannot establish long-run
stability. Gate 4 must still predeclare its E0 calibration reference, pilot
horizon, checkpoint evaluations and feasibility criteria before launch.
