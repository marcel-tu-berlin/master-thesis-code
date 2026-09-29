# Stateful MiniWoB family qualification, 2026-09-29

Scope: qualify and prepare an extension; do not launch the next E0/E1/E2 batch.
Decision [0024](../../docs/decisions/0024-qualify-stateful-families-before-expansion.md)
records the approval boundary. The original seed-4016 results remain valid within
their documented scope and are unchanged.

Recommendation for review: mixed inbox is the strongest tested next candidate.
The implementation and bounded training/evaluation paths are verified; the
scientific selection remains provisional. No candidate has yet demonstrated
that it prevents rapid interface learning.

## Observation and scoring checks

The current accessibility text omits clickable sender IDs and unnamed email
controls. Raising a text limit alone cannot restore information absent from that
representation. The extension uses opt-in visible HTML, rendered by BrowserGym's
native visibility filter and pruner. Original unfiltered HTML includes hidden
panes and tree children and is unsuitable as the policy observation.

On six candidates and three seeds each, all 18 reset cases reproduced their
observations, survived an intervening reset, and remained isolated across clients.
Across 72 recorded snapshots, every exposed HTML `bid` had browser visibility at
least 0.5. Hidden inbox panes and collapsed descendants were absent. Tool schemas
match between training and evaluation. No environment or dependency was upgraded.

Eight seeds per candidate then exercised correct and deliberately wrong paths
through the production adapter. The final set contains 96 passing controls:

| Candidate | Controls | State coverage |
| --- | ---: | --- |
| email-inbox-reply | 16 | Open sender, reveal reply, fill, send |
| email-inbox-forward | 16 | Open sender, reveal recipient, fill, send |
| email-inbox-noscroll | 16 | Reply, forward, star and trash |
| login-user-popup | 16 | Five interrupted and three uninterrupted seeds |
| navigate-tree | 16 | Three initially hidden targets; five initially visible targets |
| multi-layouts | 16 | Different native form structures and labels |

Correct paths terminate with score 1. Wrong paths receive 0; the two tree negative
paths with no reachable wrong file remain nonterminal. Terminal guards preserve
the score and prevent another server action. Reset after mutation reproduces the
initial state. The task generators and scorers are unchanged.

Two probe bugs are retained with their failed traces: the first tree negative path
clicked the requested folder while exploring; the first flat-layout solver
mistook the whole page's labels for a field label. Neither was an environment
scoring failure. The corrected scripts ran under new artifact names. The separate
v1 token audit fixes a diagnostic counter which counted `BatchEncoding` keys;
production counting was unaffected. Recounted initial email prompts are at most
1,207 tokens and their scripted feedback totals at most 1,240 tokens.

## What the family structure can establish

Email reply and forwarding have identical later action IDs across all eight
tested seeds: reply uses 72/89/81, forwarding 75/85/81. Only the initial sender
selection changes. The mixed inbox adds branches, but some goals require a single
star or trash click. These tasks support an email-specialist story; they do not
yet show a durable obstacle to interface memorization.

The tree is a stronger candidate for information gathering: hidden names cannot
be read until a branch opens. It is still shallow, and some targets are visible
immediately. Popup login supplies a persistent contingent branch: focusing a field
can disable the form until Cancel is clicked. Its generator has two interruption
locations and two no-popup choices, rather than a random timed interruption.

Interface mastery is compatible with an informative experiment. For popup login,
the useful contrast is whether the policy still responds to the interruption or
substitutes a fixed sequence with unnecessary/failed dismissals. A task can reach
high success while leaving this behavioral distinction observable. Native errors
alone do not establish harm: the interpretation must distinguish deliberate,
robust recovery from ignored feedback and incorrect completion.

Multi-orderings shuffles three fields, giving six row orders; the values and
controls are available up front. Its pinned source was inspected, but it was not
given a model screen. Multi-layouts likewise tests label/structure variation
with all required values and controls initially available. They are useful
interface controls, with weaker evidence for feedback-dependent decisions.
The source-based concerns about guess-number
(ten targets and unlimited guesses) and click-checkboxes-soft (positive partial
scores become binary success in the pinned wrapper) still apply.

A batch of tool calls is not inherently a harmful shortcut. Inspect whether a
later action required unavailable information, whether feedback was handled,
and whether the task remained correct. Short trajectories and low verification
counts alone cannot establish reward-bias substitution.

A new comparison must keep its own observation format, prompt, budgets and task
mix matched across E0/E1/E2. Comparing its accuracy directly with the retained
read-table trio would change several protocol dimensions at once. The new study
can estimate the shaping contrast within its declared setting; a cross-study
difference cannot be attributed solely to task complexity.

## Model screen

The predeclared development screen uses seed 4020, four distinct questions and
eight sampled rollouts per question for mixed inbox, popup login and tree
navigation. It uses the real native training path with a single zero-LR warmup
update, and requires identical trainable parameter hashes before and after.
The initial allowance was 8,192 trajectory tokens, 4,096 prompt tokens and 12
turns (12,288 context). The memory attempts and the final feasible setting are
recorded below. These small, unchanged-base-policy samples can reveal
technical failures and available task/length signals. They cannot establish the
post-training learning curve or show that a task will remain unsaturated.

The initial inbox attempt (`family-screen-email-inbox-noscroll-s4020-v1`) failed
before reward capture while Accelerate converted old-policy logits from BF16 to
FP32. It requested 4.64 GiB with 2.57 GiB free. No optimizer update or episode
evidence was persisted. Its failed-attempt cost is 523.40 seconds, or 0.1454
allocated GPU hours; it contributes no family-success estimate. The v2 resource
amendment retains the questions, 8,192-token budget and geometry, lowers vLLM's
memory reservation from 0.30 to 0.22 and sets
`PYTORCH_ALLOC_CONF=expandable_segments:True`. Native scoring and numerical
precision are unchanged. The retry passed scoring and saved all 32 pre-update
rollouts, but then failed in the first backward pass: a 4.64 GiB allocation was
requested with 3.18 GiB free. This is not a passed training-readiness check.

The saved pre-update inbox samples contain 13 successes and 23 terminations.
The four question groups have success counts 8/8 (delete), 2/8 (reply), 3/8
(reply), and 0/8 (forward). Only one trajectory reaches the 8,192-token cap.
These are four development questions, not 32 independent questions or evidence
of a trained policy. One successful trajectory uses 7,309 tokens.
All eight forwarding rollouts entered `sephira@example.com` instead of the
requested `Sephira`. The native scorer expects the recipient name exactly. This
is an invented-value error within a synthetic task, not a claim about delivery
through a real email system.

The v3 amendment kept full observations, 12 turns and 4x8 geometry while using
6,144 trajectory tokens (10,240 context, 4,096 prompt allowance), still above the
retained campaign's 4,096-token allowance. It retained the v2 allocator and
reservation settings. The long successful inbox trajectory would exceed this
budget, so v2 and v3 are different protocols.

The 6,144-token popup attempt also failed in its first backward pass (3.48 GiB
requested, 2.36 GiB free). Its saved pre-update outcomes are 9/32 successes:
8/8 on the uninterrupted question and 1/24 over three interrupted questions.
Only one episode reaches the cap. Successful trajectories use 349-2,553 total
trajectory tokens. Examples show repeated Submit calls after disabled-form errors,
and submission after dismissing the popup without restoring a failed field fill.
The same initial three-call batch succeeds on the uninterrupted question. Thus,
batch size alone would misclassify the behavior; recovery after feedback matters.

The user then confirmed qualification must remain on GPU 1. Short, explicitly
labelled saved-tensor replays are used to test a feasible memory setting before
another full screen. The first 5,120-token replay could not initialize its 0.22
vLLM reservation: its context needed 0.98 GiB of KV cache with 0.97 GiB available.
No new task samples were generated by that check. The allocator flag is already
a runner default; the retry made it explicit at process launch. It is not a new
reward or precision setting.

The 5,120-token memory replay passed at vLLM reservation 0.24 after fixing a
probe-only scalar/tensor type error. It completed 32 finite native loss/backward
shards and the zero-LR optimizer step with finite gradients and identical
trainable-parameter hashes. Cropping saved tensors established memory feasibility,
not task performance. All three replay script versions match their recorded hashes.

The fresh popup v4 screen then completed at that setting: 5,120 trajectory tokens,
4,096 prompt allowance, 9,216 context and 12 turns. This is a 25% larger trajectory
allowance than the retained campaign. Full observations remain uncapped. All 32
loss shards were finite and before/after parameter hashes matched. The captured
source hashes match the current source.

Its outcome is 8/32 successes and 21 terminations: all eight uninterrupted
rollouts succeed, while all 24 interrupted rollouts fail. Two episodes reach the
cap, 24 have action errors and none calls an unseen ID. Successful assistant
lengths range from 306 to 542 tokens. Task reward is constant within all four
question groups, so task-only gradients are zero; the successful-length signal
varies only in the uninterrupted group. This is useful recovery evidence, but it
does not qualify popup-only training as an informative E1/E2 comparison. The
6,144-token screen's one successful recovery also shows that 5,120 and 6,144 are
different sampling protocols; their rollouts must not be pooled.

The tree v4 screen also completed with finite loss/backward, zero task gradients
and unchanged parameter hashes. It solved 16/32 rollouts: all 16 on the two
initially visible targets and none on the two hidden targets. There were no
budget-cap episodes (maximum trajectory length 2,327 tokens). Two episodes had
action errors; none called an unseen ID. Thus 50% aggregate success masks zero
within-question task-reward variation. Successful assistant lengths span 155-454
tokens, with length signal in two groups. These samples do not justify tree-only
training without further evidence of successful exploration. The weak result is
not attributable to the generation cap.

## Completed screen comparison and proposal

All v4 screens use the same 5,120-token protocol and have four questions with
eight samples each. They are development samples from the unchanged base policy.

| Family | Successes | Terminations | Task-active groups | Length-active groups | At cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| login-user-popup | 8/32 | 21/32 | 0/4 | 1/4 | 2/32 |
| navigate-tree | 16/32 | 17/32 | 0/4 | 2/4 | 0/32 |
| email-inbox-noscroll | 11/32 | 19/32 | 3/4 | 1/4 | 4/32 |

The inbox groups have 8/8 delete successes and 1/8 each for the two reply
questions and the forwarding question. There are 21 episodes with action errors
and two with calls to IDs absent from the most recent observation. Those flags
are descriptive; they are not validated harmful-behavior labels. Successful
assistant lengths are 432-3,206 tokens. Only deletion has multiple successful
peers in this sample, so only that group has a relative-length distinction.
The longer workflows have task signal but do not yet supply within-success
length variation. This matters when interpreting the proposed intervention.

All three native runs completed their optimizer/checkpoint paths at zero learning
rate. Each produced 32 finite loss shards, and its trainable-parameter hash
remained `3fdead9d463514c156f18ac7b3e8d0b09c441743f41aad5b0081b7c8a00589ab`.
The independent scalar DAPO recomputation agrees within 1.9e-9. Gradients were
finite: zero for popup/tree and L2 norm 0.0428855 for inbox. Captured source hashes
match the tested source. Independent recomputation also matches every recorded
relative successful-length cost in the three screens.

The proposed next family is mixed inbox, subject to final user approval. It has
observed task-learning signal across its longer workflows and a coherent
specialist scope. Its 34.4% sampled success is below the historical 40-80%
point-estimate preference. Four questions cannot establish its population success
or trained learning curve. This is a provisional selection based on group-level
evidence, not a claim that it passed the former accuracy heuristic or solved
rapid interface learning. Popup and tree are retained as diagnostic candidates,
not added to the training mixture on these results.

Matched proposed E0/E1/E2 configs use fresh seed 4021. The
[proposal](../../docs/plans/inbox-family-extension.md) records the full comparison,
per-operation interpretation, unchanged native scorer, protocol differences and
approval boundary. No new campaign condition has run.

## Evaluation-path check and delivery

`family-eval-email-inbox-noscroll-s4020-v4` ran four greedy development questions
through the real evaluation CLI, on GPU 1 and under a separate diagnostic ID.
The result is 2/4 correct, three terminations and one generation-cap stop. Both
star requests recover from stale-ID errors and succeed. One reply exhausts the
budget after failed fills; the other sends before filling the body, terminating
wrong before its queued fill can execute. All four traces were inspected with
outcomes visible. These are illustrative examples, not masked label validation.
Their 50% success estimate is not pooled with the sampled training-path screen.

The evaluation writes full visible observations, ordered requested calls and
actual tool results, distinct stop reasons and episode/inference timings.
Correctness, termination, executed-action counts and timing fields agree with
the raw episodes. The report completes without context truncation or a server
error. A native action error remains a model/environment outcome rather than an
infrastructure failure.

The local gate passes: 476 tests, 7 skips and 17 setup-harness checks. All three
proposed configs validate; E1/E2 differ only in identity, description and
length-cost enablement. The retained E0/E1/E2 configs and first findings are
unchanged. Source/stack manifests and failed attempts are retained; 168 harvested
files match their GPU copies by SHA-256, excluding checkpoints and Python caches.

Ten recorded model-screen, memory-replay and evaluation attempts consumed
1.02544 allocated GPU-hours, including 0.50071 on failed attempts. Every ledger
start has an end record, and every attempt records GPU 1 only. These phase costs
exclude CPU-only oracle/setup work and idle intervals. The replay script's
scalar/tensor failure is included, not discarded. Detailed costs are in
`family-extension-s4019-ops/cost_review.json`.

All diagnostics are harvested and reviewed. No policy process remains on GPU 1;
the MiniWoB static-file service remains available. RUNNING.md records no active
run. The next E0/E1/E2 batch has not started and requires final user approval.

## Evidence

Local and remote bundle: `pipeline/runs/family-extension-s4019-ops/`. It contains
source manifests, launch records, immutable observations and correct/wrong traces,
the token recount, the accepted-control review and the predeclared model-screen
plan. Development seeds are disjoint from the retained seed-4016 campaign and its
deferred seeds 4017/4018. No next-batch condition is authorized by these probes.
