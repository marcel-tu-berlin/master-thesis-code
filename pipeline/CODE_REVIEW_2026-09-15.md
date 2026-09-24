# Feasibility setup review - 2026-09-15

## Verdict

**2026-09-16 follow-up:** both launch-safety findings below are fixed and verified.
The completed trajectory audit subsequently confirmed that native training
continues after environment completion while evaluation stops. This supersedes
the earlier no-confirmed-measurement-defect assessment for episode boundaries.
Evidence and scientific implications are in
[`runs/gate4_e1_findings.md`](runs/gate4_e1_findings.md). The accepted correction
in decision 0014 passed fresh E1/E2/E3 requalification on 2026-09-17; see
[`runs/termination_readiness_findings.md`](runs/termination_readiness_findings.md).
The model-free harder-family audit is complete; read-table-2 remains the
screening candidate. The assessment below records the original pilot review.

**2026-09-17 follow-up:** the sampled table screen and a real invalid-action
replay confirmed that native BrowserGym errors can be hidden from policy
feedback. The old adapter handled only the wrapper's exception-error field.
Evidence and the resulting qualification stop are in
[`runs/family_screen_findings.md`](runs/family_screen_findings.md). The user
approved the shared repair and read-table-2 requalification in decision 0015.
The repair's native controls and fresh model qualification are recorded in
[`runs/feedback_requalification_findings.md`](runs/feedback_requalification_findings.md).
The original recipe assessment does not establish observation completeness.

Continue the current Gate 4 pilot unchanged. No new measurement or training
logic defect was confirmed in its active path. The actual recipe matches the
accepted decisions and qualified runtime. Two reproducible launch-safety gaps
need correction before the next training launch; neither was triggered by this
pilot. This is a setup review, not a Gate 4 pass or proof of learning headroom.

The review covers configuration and seed mapping, model/LoRA loading, batch
geometry, reward construction, TRL integration, environment adapters and server
lifecycle, observation, checkpoint dispatch, evaluation budgets, metrics,
paired analysis and provenance. A separate native `codex review` pass found no
confirmed defect in the active Gate 4 paths. Its sandbox could not run pytest;
the main session ran the full local gate successfully.

## Confirmed findings

1. **Unfinished-run configuration can be overwritten (P1, next-launch safety).**
   `training/train.py:208-223` refuses an existing `checkpoint-final/`, but does
   not refuse an existing frozen `config.yaml`. Calling training again with the
   same experiment ID after a failure, or while it is still running, replaces
   that config before model loading. Partial checkpoints and logs can then be
   attributed to the new configuration. Reproduced by invoking the actual
   `main()` with a temporary run, a changed description and a stub that stops
   before model loading: the original config changed without `--overwrite`.
   A completed-run negative control raised `FileExistsError`. The current E1
   was launched into a fresh run, and its frozen config still exactly matches
   the approved config. Fix the common entry-point guard, retain the ability to
   pre-create a directory for launch metadata, and test the batch retry path.

2. **Startup failure bypasses server cleanup (P2, operational).**
   `training/grpo_runner.py:309-320` starts and waits for the server before
   entering its cleanup `try/finally`. A readiness timeout or factory error
   therefore skips `server.stop()`. `EnvServerProcess.__enter__` has the same
   failed-entry cleanup gap. Replayed the actual runner method with a server
   stub whose readiness call raises: events were `start, wait`, with no `stop`.
   A real still-starting process can remain and obstruct the next launch. The
   occupied-port guard fails loudly, so this does not silently substitute a
   different server. Current startup completed normally. Put acquired-server
   cleanup around the whole startup/use path and test the failure boundary.

Reproduction used no model, GPU, real server or research artifacts. The temporary
script was `/private/tmp/thesis-review-repros.py`; its observed output was:

```json
{
  "unfinished_run_config_overwritten": true,
  "completed_run_refused": true,
  "startup_timeout_events": ["start", "wait"],
  "server_stop_called": false,
  "models_loaded": false,
  "real_servers_started": false
}
```

Both findings were fixed on 2026-09-16 after the pilot finished. The frozen
configuration is now protected, including exclusive creation for concurrent
launches. Startup and factory failures are inside cleanup in training and
evaluation; failed context-manager entry also stops its server. Five failing
regression cases passed after the fixes, alongside checks preserving metadata-only
directories, explicit overwrite and batch retry authorization. The full local
gate passed, and native review found no remaining concrete launch-safety issue.
Reward, optimizer and successful episode semantics were not changed. The original
pilot source snapshots and qualifications remain unchanged historical evidence.

## Recipe assessment

| Part | Checked configuration and assessment |
|---|---|
| Policy | Pinned Qwen3-1.7B model/tokenizer, bf16, LoRA rank 16 and alpha 32. Same lineage and precision across future compared arms. |
| Exposure | Four prompt groups x eight rollouts, micro-batch 1, accumulation 32, one update per rollout batch. 300 updates over 500 training instances means 1,200 prompt presentations, about 2.4 passes. This is coherent; high training reward alone does not establish held-out success. |
| Optimization | Learning rate 5e-5, fused AdamW, 30 warm-up updates followed by constant learning rate, weight decay 0.1, KL coefficient 0. These match decision 0010 and the earlier stability checks. No new evidence justifies reopening them. |
| Objective | Task reward only in E1; `naive_sum`, `scale_rewards: none`, DAPO, no Liger. Centering remains per prompt group. Disabling standard-deviation scaling preserves the intended lambda dose for future shaped arms; it does not remove GRPO centering. |
| Sampling and limits | Temperature 1 for training, greedy held-out evaluation, eight tool turns, a 4,096-token whole-trajectory allowance inside the 8,192 context. Tool feedback consumes trajectory budget but is excluded from the policy loss and assistant-only efficiency measure. |
| Evaluation | Checkpoints 100/200/300, with 300 the predeclared feasibility endpoint. E0 and E1 use the same four calibration splits and seeds. Final-test offsets remain reserved. Thresholds are fixed to the E0 reference; comparisons use paired instances. |

Both active configs validate and exactly match their frozen run copies. All
local qualification source hashes match, as did the approved remote manifest at
the live check. Earlier native Gate 3 captures verify the installed trainer's
32-way accumulation, reward centering, masks, full-batch DAPO denominator and
gradients. Observer parity was separately qualified before E1; this review
checked those artifacts and source identity, rather than launching another
expensive duplicate experiment.

## Remaining scientific limits

The live risk is family suitability: training saturation can remove informative
task-reward groups, while E2 length opportunity and E3 budget exhaustion may
behave differently. Decide from the final held-out evaluations and late sampled
groups, using the declared Gate 4 rules. Do not replace the family merely
because training reward is high, or select an earlier checkpoint that passes.

The off-target labels still require the planned blinded, task-grounded audit.
Action count is not proof of verification and an early stop is not automatically
harmful. This is an existing interpretation gate, not a newly found code bug.
The existing token re-encoding approximation and HF/vLLM numerical differences
remain documented limitations covered by prior checks, not reasons to retune
the running pilot. A new family needs its own reward/oracle and observation
audit, as described in the conditional environment-options plan.

The backlog's claim that a success drop cannot be detected from a 100% baseline
was corrected. The declared below-90% qualification rule itself was preserved.

## Verification

`./.claude/check.sh` completed successfully:

```text
72 files already formatted
All checks passed!
Success: no issues found in 72 source files
400 passed, 5 skipped in 22.06s
17 passed, 0 failed (setup harness)
```

The CPU skips require optional/native packages; the earlier Gate 3 record holds
their installed-stack checks. No new GPU diagnostic, restart, environment
upgrade, commit or push was performed for this review.
