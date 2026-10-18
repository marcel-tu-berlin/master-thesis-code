# Click-checkboxes-soft E0/E1/E2 extension

Authorized on 2026-10-13: briefly verify native logic and the adapter, then run
E0 -> E1 -> E2 on physical GPU 0 in parallel with click-tab-2 on GPU 1.
This supersedes the earlier instruction to leave this family queued. Stop each
campaign after its trio or on runtime/integrity failure; no restart or overwrite.

## Frozen comparison

Use fresh seed 4232 and `e{0,1,2}-click-checkboxes-soft-s4232-c4096`.
Both trained arms start independently from the original Qwen3-1.7B revision.
Retain bf16 LoRA rank 16/alpha 32, 300 updates, batch four, eight rollouts,
DAPO, `naive_sum`, `scale_rewards: none`, and the optimizer/vLLM settings of
the click-tab-2 campaign. E2 alone enables relative successful-response length
cost at weight 0.1. No warm start, extra dose, E3, or model change.

Keep 4096 prompt tokens, 4096 trajectory tokens including feedback, 8192 context
and eight model/tool-calling turns. Preserve the accepted batch/continuation
boundary and the assistant-token ruler. Use visible HTML and click/noop only.
Training uses 500 seeds, 4232000000-4232000499. E0 and trained checkpoints
100/200/300 all use the same 200 held-out seeds, 4232100000-4232100199.
Update 300 is primary. E0 supplies fixed reference thresholds.

The native synonym-checkbox task and scorer remain unchanged. The source adds
+1 for each correctly selected or correctly unselected box and -1 otherwise;
native success is a positive sum. Exact goal completion instead requires every
requested synonym group selected and every other group unselected. This known
checker gap predates shaping. Evidence for a shaping-associated behavior change
requires an E2 versus E1 comparison, not merely observing partial success.

## Isolation and verification

Deploy a complete pipeline copy to `/workspace/master-thesis-code-checkboxes-soft`.
Only its two entry-point GPU assignments differ from the live GPU 1 source,
changing physical GPU 1 to GPU 0 before model loading. Freeze the complete
source hashes and exact diff. Do not synchronize into the running tab deployment.
Use environment port 8002, native-check port 8012, distributed rendezvous 29502,
and a separate vLLM IPC directory. Share only the pinned Python environment,
model cache and read-only MiniWoB static service. Controllers, locks, outputs
and remote state files are separate; local monitoring aggregates both campaigns.
CPU/RAM/disk remain shared, so concurrency can affect wall-clock costs.

Brief controls use prior qualification seeds, never final held-out questions:
deterministic reset, synonym matching, exact success, one-omission partial
success, all-wrong failure, invalid-action recovery, selected-state feedback,
terminal reward preservation, and identical training/evaluation tool schemas.
Reuse prior full backward-pass qualification; no new learning pilot is needed.
Run the CPU gate and verify CUDA visibility, source/config/task/stack hashes.
Every completed arm must pass technical review before the controller advances.
Low success is a result, not a performance launch gate.

## Analysis fixed before launch

Primary: paired native success change and paired assistant-token change on
joint native successes, E2 versus E1 at update 300. Retain the 10% compression
target and five-percentage-point success-regression margin with uncertainty.
Also report paired exact-goal success, tokens on joint exact successes, and
native-success-but-incomplete submissions. These are declared secondary
diagnostics; do not replace the native reward or retroactively change it.

Report omissions and extraneous selections per submitted episode and per
required/irrelevant checkbox opportunity, plus native success/failure/unfinished
outcomes and stop reasons. Identify submissions chosen directly from visibly
incomplete state only when Submit is the first action in the assistant batch.
Keep submissions after earlier precommitted actions separate: those actions'
feedback was not available when the batch was chosen. Count complete/incomplete
decision states and submission opportunities, including unknown terminal targets.
Generic repetition/invalid actions and checkpoints 100/200 are descriptive.
Distinguish E0-to-E1 specialization from E1-to-E2 shaping, retain the single
training-seed limitation, and do not infer intent from these mechanical labels.

Harvest non-weight artifacts, including checkpoint-evals, with verified hashes.
After scientific review and a user report, complete the campaign task. The shared
watcher stops once both independently authorized campaigns are fully reviewed.

Operations: `pipeline/runs/click-checkboxes-soft-campaign-s4232-ops/`.
Results: `pipeline/runs/click_checkboxes_soft_s4232_findings.md` after review.
