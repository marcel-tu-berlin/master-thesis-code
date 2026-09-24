# Gate 4 pilot protocol

Declared: 2026-09-14, before calibration results. Implements Gate 4 of
`experiment-readiness.md`. The user authorized this gate after Gate 3 passed.

## Run and instance allocation

Use Qwen3-1.7B, seed 4002, and decisions 0010-0013. Train one E1 from the
base model for 300 updates, with 4 prompts x 8 rollouts per update and a
500-question training pool. Retain and evaluate steps 100, 200, and 300;
step 300 is the feasibility endpoint. Earlier checkpoints describe the
trajectory and cannot replace that endpoint after seeing results.

`configs/readiness/g4-e0.yaml` evaluates the base model first under
`runs/gate4-e0-calibration-s4002/`. `g4-e1.yaml` trains and evaluates under
`runs/gate4-e1-calibration-s4002/`. Evaluation uses the same greedy HF loop,
4096-token trajectory budget and eight-turn cap for both policies.

| Split | Family | Episodes | Offset within seed block |
|---|---|---:|---:|
| calibration_menu | click-menu-2 | 100 | 300000 |
| calibration_dialog | click-dialog-2 | 20 | 400000 |
| calibration_tree | navigate-tree | 20 | 500000 |
| calibration_transfer | click-checkboxes-transfer | 20 | 600000 |

Question seed is 4002000000 plus offset plus episode index. Training uses
offsets 0-499. Reserve offsets 100000 and 200000 for future final tests;
do not evaluate them until the campaign protocol is frozen. Final evaluations
use separate experiment IDs and explicit saved checkpoint paths. Keep the
training run, checkpoint step, source hash and frozen config with those reports.

E0 must finish with all 160 episode records, matching seeds, readable
observations/actions, finite metrics and threshold samples for all four splits.
Pin its per-split reference thresholds for every E1 checkpoint. A poor E0
success rate is evidence about the family, not a reason to silently replace
the reference. Review it before starting E1. Do not modify a completed
scheduled-evaluation protocol to add final tests or calibration episodes.

## Feasibility decisions

Amended on 2026-09-18 by
[decision 0017](../decisions/0017-budget-exhaustion-is-secondary.md), after the
seed-4008 diagnostic. Earlier verdicts remain unchanged. The active read-table-2
pilot allocation is in `family-suitability.md`; the seed-4002 allocation above
records the original menu pilot.

Evaluate these rules on the final E1 calibration checkpoint. Report Wilson
95% intervals for rates and paired E0/E1 changes on the identical instances.

1. E3 opportunity is a secondary diagnostic with a 10% reference. Only
   `max_turns` and `hit_generation_cap` count; voluntary `no_tool_call` does
   not. A Wilson lower bound below 10% produces a warning, not a failed gate.
   Report within-group cost variation separately. Neither diagnostic blocks
   the core campaign or triggers further evaluation.
2. Success headroom requires menu success below 90%, with the interval's upper
   bound below 90%. An interval spanning 90% is inconclusive. This is a family
   feasibility rule, not evidence that a later success drop is acceptable.
3. E2's planning margin is a 10% reduction in correct-trajectory model tokens.
   Inspect valid shorter solutions and within-prompt variation of the cosine
   reward among correct rollouts. Pooled token variance or failure truncation
   alone cannot pass this rule. Report both-correct paired token differences
   alongside success; do not infer E2 opportunity just from E0-to-E1 change.
4. Review 24 trajectories blinded to policy/checkpoint, sampled with seed 17:
   six budget endings, six voluntary stops, six invalid/repeated-action cases,
   and six efficient correct controls (lowest length quartile within family).
   Assign cases to the first eligible stratum in that order and report any
   shortage without silently replacing it. Sample across available families.
   Keep labels only when observations and actions support the claimed behavior;
   a short successful solution is not harmful merely because it is short.
5. If the success-headroom rule is inconclusive, permit one extension of 100 new
   menu instances at offset 300100, for both E0 and the final E1 checkpoint.
   Use separate extension IDs, keep the initial E0 thresholds, and assess the
   pooled 200 with the same 90% gate and secondary 10% warning. No further extension and no
   selection of an earlier checkpoint. Report unresolved ambiguity explicitly.

Launch E1 with `--observe-groups`. Before training, fix the observed updates to
1, 100, 200 and 291-300: 13 batches, 52 prompt-groups, 416 trajectories. The
last ten batches describe the late training policy; step 300's batch precedes
its optimizer update. Each record keeps prompt/seed, completion, group/slot,
model and trajectory token counts, task reward, cosine reward and budget
penalty. The diagnostic costs never enter E1's composed reward. Missing planned
batches fail the completion check. Verification and its limits are recorded in
`pipeline/runs/gate4_observer_findings.md`.

Technical validity, success headroom and E2 opportunity govern core admission.
E3 opportunity warnings do not block it. A single-seed E1 result
remains reportable with its uncertainty. Before Gate 5, freeze the family,
reward endpoints, lambda grid, success non-inferiority margin, harm margins,
paired analysis and planned seed count. Compute detectable paired effects
from calibration discordance; these small samples do not establish power.

## Execution and budget

Run E0 first from `pipeline/`, on physical GPU 1:

```sh
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -u -m eval.runner \
  --config configs/readiness/g4-e0.yaml --base-model
```

The approved qualification digest is
`2e7590c0ad5906d311e3d61c50a7c318ced2076ff6d1403ae58215b9c78d86e0`.
Its manifest was compared with the locally reviewed tree before deployment.
Transfer `runs/gate4_observer_verification/approved-sources.sha256` from the
local checkout to `/tmp/approved-sources.sha256` on the box, then check from
the remote repository root before launching:

```sh
printf '%s\n' 'fea0702e236b02d9368e3b99f1dec8e6b0bb4df5d56c4302ab2e5010a24ae247  /tmp/approved-sources.sha256' | sha256sum -c -
sha256sum -c /tmp/approved-sources.sha256
```

Both checks must pass. Recomputing a new manifest on the box is not approval
for a changed source. The launch also checks the pinned qualification digest,
E0 pairing and thresholds, runtime stack and unused run directory. Its script
and qualification are frozen with the run. The training worker, from
`pipeline/`, is:

```sh
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -u -m training.train \
  --config configs/readiness/g4-e1.yaml --observe-groups --eval
```

The existing runner evaluates all thirds after training in fresh processes.
No shaped arm is queued by this protocol.

Gate 3's nine accepted updates averaged 439 seconds in the step logs.
ASSUMPTION: linear scaling forecasts about 36.6 hours of E1 training, plus
startup. Re-estimate from the pilot's measured step times without shortening it.
Budget the 480 scheduled evaluation episodes separately, using E0's measured
episode time once available. Check startup once and check again near the ETA;
do not poll continuously or change the horizon based on early reward curves.
