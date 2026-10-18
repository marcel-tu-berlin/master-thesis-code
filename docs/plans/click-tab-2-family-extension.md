# Click-tab-2 E0/E1/E2 extension

Authorized on 2026-10-13: briefly verify native logic and the adapter, then run
E0 -> E1 -> E2 automatically with the existing budgets. Use physical GPU 1.
Stop after this trio or a runtime/integrity failure. Click-checkboxes-soft
remains open; parallel execution on GPU 0 is a separate pending choice because
its last use could not be established from recorded GPU accounting.

## Frozen comparison

Use fresh seed 4231 and `e{0,1,2}-click-tab-2-s4231-c4096`. No warm start.
Qwen3-1.7B revision, bf16 LoRA rank 16/alpha 32, 300 updates, batch size four,
eight rollouts, optimizer, learning rate, DAPO, `naive_sum`, `scale_rewards: none`
and vLLM settings match the completed tic-tac-toe campaign. E1 and E2 start
independently from the base model. Only enabling relative successful-response
length cost at weight 0.1 distinguishes E2 from E1, apart from metadata.

Retain 4096 prompt tokens, 4096 trajectory tokens including feedback, 8192
context tokens and eight model/tool-calling turns. Preserve the implemented
batch/continuation boundary accepted in the tic-tac-toe review: precommitted
calls run in order until environment completion, and the allowance governs
continuation. It is not a strict interruption between calls or a hard cap on
retained terminal feedback. No token ruler or training/evaluation source changes.

Use full visible HTML with click/noop. Fill, scrolling and additional interaction
tools stay disabled. Native task source and scorer remain unchanged. The task
has three tabs; clicking any link with exactly the requested text succeeds,
including duplicate occurrences. A different link terminates with failure.
Tab changes and noops do not themselves satisfy the goal.

Training uses 500 seeds, 4231000000-4231000499. All seven evaluations use the same
200 held-out seeds, 4231100000-4231100199. E0 evaluates the base once. E1/E2
evaluate at updates 100, 200 and 300; update 300 remains primary. E0 supplies
fixed reference thresholds. Brief scripted controls reuse separate qualification
seeds and do not inspect final held-out questions or create policy checkpoints.

## Verification and automatic advancement

Check deterministic reset, visible tab navigation, exact-target success,
wrong-link failure, invalid-action recovery, terminal reward preservation and
identical training/evaluation tool schemas. Reuse the completed 128-rollout
qualification and full forward/backward captures; do not run another learning
pilot. A brief native check does not guarantee every future batch fits memory.

Run the CPU project gate and freeze source, config, task, environment and host
driver hashes. The GPU controller admits only the three declared arms. Every
completed arm must pass config/provenance, metric-recomputation, checkpoint,
cost-ledger and reward-arithmetic checks before advancement. A task-grounded
auditor compares attributable visible-link selections with native scores and
reports any terminal action whose clicked element is not an identified link
separately; it must not invent a label for an unobserved click target.

Never restart, overwrite, change budgets or automatically repair a failed run.
Low success is a result, not a new launch gate. Harvest non-weight artifacts,
including checkpoint-evals, with matching hashes. Record live state with box time.
After the trio, conduct and report the scientific comparison, then stop monitors.

## Analysis fixed before launch

E2 versus E1 at update 300 is primary. Use paired success changes on the same
held-out seeds and paired assistant-token change on joint successes. Retain
the declared 10% compression target and 5-percentage-point success-regression
margin, uncertainty and single-training-seed caveat. Earlier checkpoints are
descriptive. E0 versus E1 measures specialization, not shaping.

Report native success/failure/unfinished outcomes and all stop reasons. Separate
initially visible and initially hidden targets. Retain generic invalid/repeated
action indicators, and exploratory task-grounded counts of wrong visible-link
selection, tab changes while the target is visible, and stopping with a visible
target. Report opportunities as well as events. Attribute a decision-boundary
label only to actions chosen with that observation available; later batch actions
cannot be labelled as ignoring unseen feedback. These labels do not establish
intent or a causal mechanism of reward-bias substitution.

Operations: `pipeline/runs/click-tab-2-campaign-s4231-ops/`.
Results: `pipeline/runs/click_tab_2_s4231_findings.md` after review.
