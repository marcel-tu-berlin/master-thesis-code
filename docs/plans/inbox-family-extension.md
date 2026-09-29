# Proposed mixed-inbox E0/E1/E2 extension

Status: prepared for review, not authorized to launch. The user must approve this
extension before any of its three conditions runs. Qualification is recorded in
[`family_extension_s4019_findings.md`](../../pipeline/runs/family_extension_s4019_findings.md).
Decision [0024](../decisions/0024-qualify-stateful-families-before-expansion.md)
preserves the first read-table result and this approval boundary.

## Question and scope

Use `email-inbox-noscroll` to study an inbox specialist across reply, forward,
delete and mark-important requests. The longer workflows require finding the
sender, opening the message, choosing the operation and entering the requested
content. The native generator also includes simpler one-action requests. Report
these operation types separately so a change in easy-task success cannot hide
failed replies or forwarding.

This is a broader workflow experiment, not a demonstrated cure for rapid
interface learning. Later control IDs remain stable, and a trained policy may
learn them quickly. The informative question is whether compression preserves
the requested recipient, content, operation and completion across page states.
Correct, efficient batching is allowed. Shorter traces, missing intermediate
observations and interface memorization alone are not harmful substitution.
The simulator checks the requested recipient name as an exact string; invented
email addresses are wrong. Its score is not evidence of real email delivery.

Popup login and tree navigation remain diagnostic candidates. Their completed
development screens contain no within-question task-reward variation; aggregate
accuracy conceals a split between uniformly easy and uniformly failed questions.
Adding them to the training mixture now would not establish learnable complexity.

The final development screen has 11/32 successes over four questions and task
signal in three groups. This falls below the earlier 40-80% point-estimate
preference; it is not a claim that the preferred success band was met. The
proposal favors observed within-question learning signal over aggregate accuracy
alone. It remains a small-sample, provisional choice requiring the user's review.

## Proposed conditions

| Condition | Config | Initialization | Reward |
| --- | --- | --- | --- |
| E0 | `pipeline/configs/e0-inbox-proposed.yaml` | Original base, no adapter | Evaluation only |
| E1 | `pipeline/configs/e1-inbox-proposed.yaml` | Original base, new LoRA | Native task success |
| E2 | `pipeline/configs/e2-inbox-proposed.yaml` | Original base, new LoRA | Task success minus 0.1 times relative successful-response cost |

Use fresh seed 4021. Training uses question seeds 4021000000-4021000499;
`held_out_inbox` uses 200 questions, 4021100000-4021100199. These are disjoint
from qualification and the retained/deferred campaigns. E0 supplies the fixed
threshold reference for both trained arms. Each trained arm is evaluated at
updates 100, 200 and 300; update 300 is primary. Do not choose a checkpoint or
extend the sample after inspecting outcomes.

Keep the original pinned Qwen3-1.7B revision, bf16, rank-16/alpha-32 LoRA, 300
updates, 4 questions by 8 rollouts, micro-batch 1, DAPO, learning rate 0.00005,
constant schedule with 10% warmup, `naive_sum`, and `scale_rewards: none`.
E1 and E2 differ only in length-cost enablement, apart from their names and
descriptions. No warm start, Liger, vLLM sleep, placebo or E3.

The new protocol uses full visible HTML, the documented ordered-tool-batch prompt,
click/fill/noop, 12 turns, 5,120 whole-trajectory tokens and a 9,216-token context
with 4,096 prompt allowance. Tool observations count against the trajectory
budget; only assistant tokens count toward the length reward. vLLM reservation
is 0.24. Qualification and proposed execution use physical GPU 1 only. The
8,192- and 6,144-token settings failed native training memory checks; their
saved samples are separate protocols, not pooled evidence for this setting.

## Review and interpretation

Carry forward the initial plan's paired success/compression analysis, fixed
margins, bootstrap procedure, costs and explicit uncertainty about one training
seed. Compare new E2 against new E1 on identical questions and observations.
The old read-table trio remains first evidence; differences between studies
cannot be attributed solely to family because observation, prompt and budgets
also changed.

Alongside the pooled result, report reply/forward/delete/important counts and
paired outcomes. The fixed sample was not sized for four separate equivalence
claims. Inspect all success regressions and a declared sample of efficient
correct controls against the requested operation, sender, text and visible
states. Distinguish failed tool calls, ignored feedback, incorrect sends,
voluntary stopping and budget exhaustion. A triggering action error is not by
itself failed recovery. Existing verification-depth and short-response flags
remain descriptive until task-grounded labels are validated.

Track within-question reward variation, active groups and early observed
training behavior. Length shaping can activate gradients where task-only GRPO
has none; the comparison does not isolate this effect from cost assignment.
Qualification does not establish the trained learning curve or durable headroom.
If both arms learn the workflow rapidly, report that limitation rather than
calling the family a solution to interface saturation.

## Execution boundary

After explicit approval, sync the whole pipeline, record source/config hashes
and GPU allocation, then run E0, E1 and E2 using the existing CLI and RUNNING.md
discipline. E0 must finish before the reference-dependent evaluations. Harvest
and review each phase. Stop after this trio; other families, doses, seeds and
non-termination training require a further decision. No new watcher or queued
campaign has been installed for this proposal.
