# Tic-tac-toe E0/E1/E2 extension

Authorized: 2026-10-07. The user requested a brief logic verification followed
by automatic E0 -> E1 -> E2 advancement for tic-tac-toe. Stop after this trio.
Click-tab-2 and then click-checkboxes-soft remain open follow-ups, not part of
this automatic batch. The earlier proposal for a separate learning pilot is
superseded by this explicit campaign authorization.

## Fixed comparison

Use seed 4230 and fresh runs `e0-tic-tac-toe-s4230-c4096`,
`e1-tic-tac-toe-s4230-c4096`, and `e2-tic-tac-toe-s4230-c4096` on GPU 1.
The configs are `pipeline/configs/e{0,1,2}-tic-tac-toe.yaml`.
E0 evaluates the original base. E1 and E2 independently train new LoRA adapters
from that same base; no qualification checkpoint or warm start is used.
Only enabling relative successful-response cost at weight 0.1 distinguishes
E2 from E1, apart from metadata. Other doses, placebo, replication and E3 remain
deferred. Low E0 success and policy errors are outcomes, not new launch gates.

Retain the corrected inbox recipe: pinned Qwen3-1.7B, bf16, rank 16/alpha 32,
300 updates, four questions by eight rollouts, micro-batch one, learning rate
0.00005, 10% warmup, constant schedule after warmup, DAPO, KL zero, weight decay
0.1, `naive_sum`, `scale_rewards: none`, vLLM reservation 0.24, and no Liger or
vLLM sleep. Budgets remain 4,096 prompt tokens, 4,096 whole-trajectory tokens
including feedback, 8,192 context tokens and eight turns in every phase.
Assistant-token reward measurement is unchanged.

Use the qualified full visible-HTML interface with click/noop; fill, scrolling
and additional interaction tools are disabled. Native task code and scoring are
unchanged. A win is success; both a draw and a loss are native failures. Clicking
an occupied cell can silently leave the board unchanged, so the generic invalid
action counter alone does not measure all ineffective moves.

## Questions and observations

- Training: 500 reset seeds, 4230000000-4230000499.
- Held-out: 200 reset seeds, 4230100000-4230100199, split `held_out_tic_tac_toe`.
- E0: one greedy evaluation. E1/E2: greedy evaluations at updates 100, 200 and
  300 on the same held-out seeds. Update 300 is primary.
- E0 supplies fixed token thresholds to every trained-policy evaluation.
- Native logic checks reuse separate qualification seeds; they do not inspect
  the final held-out seeds or supply weights.

These blocks are separate from the completed qualification. Tic-tac-toe has few
possible initial boards; different reset seeds can show the same initial board
and instruction. Disjoint seeds are not a claim of novel board configurations.
The opponent's seeded randomness and action-dependent responses generate later
states. Paired arms share initial observations and reset seeds; different policy
moves naturally lead to different subsequent states.

## Brief verification and advancement

Reuse the existing tactical self-checks, retained native outcome audits and
four completed full forward/backward qualification captures. Run the eight
scripted native game/recovery controls against the current source. These are
browser checks, not another model screen or a long training pilot. Run the
project check and verify frozen E1/E2 config equality outside the shaped term.

The GPU-side controller runs each complete arm, verifies its artifacts and
native-board outcome labels, then admits the next arm. The source/stack/config
manifest is checked before every launch. A nonzero exit, missing or inconsistent
artifacts, source drift, nonfinite training values or native outcome mismatch
stops advancement with evidence. Never overwrite or automatically retry a failed
run, change budgets, switch families or alter the reward to make it proceed.
All completed phases remain marked done/unharvested until copied and verified.
The controller works independently of this Mac; a local mirror harvests results
and live state when the SSH connection is available.

Scientific interpretation remains a review of the completed comparison, not an
automatic claim of harmlessness. Record success, compression, all stop reasons,
runtime costs and task-grounded behavior for each checkpoint. Unexpected
scientific issues that invalidate the declared comparison require a decision.

## Analysis fixed before launch

Carry forward the initial campaign's paired analysis: E2 versus E1 at update
300 is primary; earlier checkpoints are descriptive. Use identical held-out
seeds and initial observations, joint-success assistant-token change, the
declared 10% compression target, and the 5-percentage-point success-regression
margin. Report sampling uncertainty and the single-training-seed limitation.
E0-to-E1 changes describe specialization and cannot be credited to shaping.

Retain the generic off-target panel and add the already qualified board labels:
known-occupied clicks at a decision boundary, missed immediately available wins,
and unblocked single threats. Report episode rates plus opportunity denominators
and raw events. The last label is risk, not proof of suboptimal expected-win play
against the partly random opponent. Calls precommitted in one assistant turn
must not be labelled as ignoring feedback that was unavailable when chosen.
Changed state visitation can change the available tactical opportunities.

Preserve win, draw, loss and unfinished counts separately. Neither shorter
reasoning nor fewer preceding actions alone establishes harmful substitution.
The task-specific panel is exploratory across several indicators; do not choose
its strongest result after evaluation and label it a confirmatory primary test.
No shifted family is silently added to this initial comparison.

Operations and launch/review receipts live in
`pipeline/runs/tic-tac-toe-campaign-s4230-ops/`; live state is in `RUNNING.md`.
