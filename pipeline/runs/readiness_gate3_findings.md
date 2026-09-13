# Gate 3 E2 readiness capture - passed

Date: 2026-09-13. Run: `readiness-g3-e2-s4001-r2`, the pinned E2
cosine-length arm at lambda 1.0. This is diagnostic evidence, not a thesis
result.

## Verdict

E2 passed. The final deployed-runtime controller passed Gate 1, E1, and E2,
then admitted only `run_e3:readiness-g3-e3-s4001-r2`. Its overall status is
`not_tested` because E3 has not run. E3 was not launched.

The fresh local gate passed after the local and deployed semantic-source digest
matched: `b7f362019a670e561383d292d2b9f7816dfe83e28eec6bd37281c48214707bd2`.
The deployed controller used the same interpreter path recorded during training.
This preserves the full readiness-recorder fingerprint while checking package,
OpenEnv, platform, and executable identity exactly.

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

## E3 execution - blocked on missing capture variation

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

The controller reported `gate4_e1_pilot`, but Gate 3 remains blocked under the
stricter scientific criterion. Use the plan's labelled E3 integration replay to
exercise budget exhaustion deterministically rather than rerunning until random
sampling produces it. The 1/4 evaluation result is lifecycle evidence only.
