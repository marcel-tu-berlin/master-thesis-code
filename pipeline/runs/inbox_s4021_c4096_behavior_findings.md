# Inbox E1 -> E2: behavior, interpretation and visual guide

Offline analysis: 2026-10-05. This supplements the sealed
[initial comparison](inbox_s4021_c4096_findings.md); it does not change its results,
endpoint, protocol or launch decision. No new experiment was run.

## Assessment

**This is useful evidence for the thesis question.** Successful-response
compression accompanies a large change in tool behavior, and the intermediate
checkpoints expose a temporary task-quality deficit. The final policy achieves
better task outcomes while making more invalid actions. These observations
support a behavioral trade-off worth investigating. They do not yet establish
harmful reward-bias substitution or deliberate exploitation of the scorer.

E1 and E2 were trained independently from the same base; E2 is not a continuation
of E1. They share the original 4,096 trajectory-token, 4,096 prompt-token,
8,192 context-token and eight-turn limits. The only experimental reward change
is enabling relative successful-response length cost at weight 0.1.

## Visuals: start with 01 and 04

All figures are available together in the
[five-page PDF](inbox-campaign-s4021-c4096-ops/deep-review-20261005/inbox_e1_e2_behavior.pdf).
Each figure also has a separate PDF beside its PNG.

| Figure | What it explains |
| --- | --- |
| [01: learning and behavior](inbox-campaign-s4021-c4096-ops/deep-review-20261005/01_learning_and_behavior.png) | Success, paired compression, invalid actions and repeated actions across updates 100/200/300. |
| [02: tokens, actions and turns](inbox-campaign-s4021-c4096-ops/deep-review-20261005/02_tokens_actions_and_turns.png) | Where token savings came from on the same 182 jointly solved questions. |
| [03: operation breakdown](inbox-campaign-s4021-c4096-ops/deep-review-20261005/03_operation_breakdown.png) | Reply and forwarding account for most of the intermediate quality deficit. |
| [04: one task, two strategies](inbox-campaign-s4021-c4096-ops/deep-review-20261005/04_same_task_different_strategy.png) | The exact order of actions, page changes and feedback in an illustrative trace. |
| [05: training signal](inbox-campaign-s4021-c4096-ops/deep-review-20261005/05_training_signal.png) | Length shaping also increases the fraction of groups with a learning signal. |

## 1. What changed at the declared final endpoint?

| Measure | E1 update 300 | E2 update 300 |
| --- | ---: | ---: |
| Successful tasks | 183/200 (91.5%) | 199/200 (99.5%) |
| Mean assistant tokens, same jointly correct tasks | 1,472.7 | 119.8 |
| Mean parsed reasoning tokens, same jointly correct tasks | 1,385.7 | 17.1 |
| Episodes with an invalid action | 10/200 (5%) | 200/200 (100%) |
| Episodes with an exact repeated action | 1/200 (0.5%) | 66/200 (33%) |
| Mean attempted actions, all tasks | 3.15 | 4.18 |
| Episodes that do not terminate | 5/200 | 0/200 |
| Mean total episode time on this hardware | 69.22 seconds | 11.73 seconds |

The paired efficiency endpoint is a **91.96% median reduction** in assistant
tokens, with bootstrap 95% CI [91.57%, 92.36%], on 182 questions both policies
solve. Every one of those 182 E2 responses is shorter. Of those questions, E2
uses more actions in 170 and the same number in 12; none uses fewer. The number
of model decision turns is unchanged in 148, higher in 27 and lower in seven.

The decomposition makes the main source of savings concrete: generated
reasoning text almost disappears. The residual containing tool-call JSON and
template framing increases from 80.2 to 102.7 tokens. Parsed visible content
falls from 6.8 to zero. The categories use existing saved token fields; they
are not a new token-counting definition or a measurement of internal reasoning.

Thus the policy produces a much more compact representation of its workflow.
It also makes additional unsuccessful calls. The observed latency improvement
shows that these calls do not erase the token savings on this simulator and
hardware. Their cost in another application remains unmeasured.

## 2. The recurring action pattern

Every final E2 episode begins with a valid email-opening click followed by an
invalid action in the same model-generated batch. In 197 episodes these are
exactly the two first-turn calls; three episodes contain additional calls.
All 206 final action errors leave the saved visible page unchanged.

In the illustrative episode 0, the request is to mark Floris's message important:

- E1 opens the message, processes the resulting page, and clicks its detail star:
  two actions, two turns, 912 tokens.
- E2 opens the message and also attempts the now-hidden inbox-list star within
  the same batch. After receiving the page and error, it clicks the detail star:
  three actions, two turns, 81 tokens.

The longer reply workflow shows the same contrast. In episode 5, both policies
correctly reply to Rene with `Sed nibh faucibus.`. E1 takes four turns and four
actions, using 3,144 tokens. E2 takes four turns and five actions, using 161
tokens. Its extra first-turn call tries to fill the now-hidden sender element;
the later reply, fill and send actions complete the task correctly.

**Interpretation:** the traces are consistent with a compact, stereotyped action
sequence that handles page transitions imperfectly and then recovers. This is
a mechanism hypothesis, not a demonstrated internal strategy. The evidence does
not show that the invalid call helps compression or that the model seeks errors.
The next valid call could depend on the page, the error, memorized controls, or
some combination; saved traces cannot identify that dependence.

## 3. Why checkpoint 200 matters

| Checkpoint | E1 success | E2 success | Paired E2 token reduction | E2 invalid-action episodes |
| --- | ---: | ---: | ---: | ---: |
| 100 | 64% | 77% | 60.79% | 41% |
| 200 | 85% | 75% | 81.99% | 94% |
| 300 | 91.5% | 99.5% | 91.96% | 100% |

At update 200, E2 loses 38 questions E1 solves and gains 18 that E1 misses. The
largest operation deficit is reply: 26/54 versus E1's 40/54. Forwarding is also
lower, 42/60 versus 46/60. Reviewed failures include wrong recipients, changed
forward bodies, wrong operations and repeated invalid fills. Some episodes
exhaust the trajectory budget through tool feedback despite short model output.

Between E2 updates 200 and 300, 50 questions recover and one regresses. Final
forwarding is 60/60 and reply is 53/54. The remaining failure sends `Feugiat`
instead of `Feugiat.`. The period matters to the specified task. The final
endpoint therefore supports successful compression and improved task outcomes;
checkpoint 200 records a temporary quality deficit, not the final treatment
effect. The 77% to 75% change between E2 checkpoints alone is too small to call
a demonstrated population-level decline.

### Repetition does not mean the same thing at every checkpoint

At E2 update 200, 57 repeated-action events occur across 31 episodes. All repeats
occur in later model turns, after feedback was available. At update 300, all 66
repeat events occur within the same first-turn batch: 57 forwards and nine
deletions repeat a click on a now-hidden email-list element.

The final batch was chosen before the model could process the first call's
result. Calling this "ignored feedback" would be incorrect. Likewise, the
aggregate repeated-action rate rising from 15.5% to 33% does not establish that
failure to recover became worse. The timing and task outcome distinguish these
events. This is a concrete reason to validate behavior labels against traces.

## 4. Does this support reward-bias substitution?

The motivating literature describes optimization moving toward another proxy
while the targeted bias measure improves; it argues for observing multiple
behaviors on policy-generated outputs. See
[Lamparth et al., Reward Bias Substitution](https://arxiv.org/abs/2605.27996).
Our environment uses native task success plus an explicit token cost, so an
analogy to reward-model bias substitution requires a task-grounded argument.

Three claims have different levels of support:

1. **Targeted compression succeeded:** strongly supported in this seed. Both the
   declared compression and success-preservation criteria pass.
2. **Tool behavior changed outside the target:** strongly supported. The declared
   invalid/repeated-action margins fail, including on jointly successful tasks.
3. **Optimization substituted a harmful proxy for the real task objective:**
   not established. Final task success improves, and the native-state review
   found no success-scoring bypass. The action errors have no observed state
   effect. Their causal role and robustness cost require a controlled test.

The defensible thesis statement is: **In this run, token-cost shaping sharply
compressed successful trajectories and improved final task success, while
increasing invalid tool use and producing a temporary intermediate quality
deficit. Task success and token count alone would miss that behavioral change.**

This is a more informative workflow than the earlier table task, but final E2
again reaches the ceiling. The inbox family has not solved durable interface
complexity. Keep both studies as evidence; do not discard a valid study because
its outcomes differ from the hoped-for mechanism.

One further limitation matters: E2 has varying composed reward in 92.17% of
training groups, versus 49.67% for E1. The added reward also supplies more learning
signal. One seed cannot establish reproducibility or separate the cost preference
from this activation effect. Episode-level confidence intervals do not measure
training-seed uncertainty.

## Recommended next steps, requiring a new decision

1. **First, test the specific robustness hypothesis with frozen checkpoints.**
   Propose a small paired evaluation on fresh held-out tasks with ordinary and
   consistently remapped element IDs across page states. Preserve task content,
   visible labels, scoring, model weights and all original budgets. Qualify the
   remapping with a scripted oracle before model evaluation. Compare the change
   in success, invalid batches, later-turn recovery, exact recipients/content
   and total cost for E1 and E2. Predeclare the sample and criteria. This tests
   dependence on stable interface IDs; it does not isolate the effect of token
   cost during training. It is a new evaluation protocol, kept separate from
   the original result.
2. **Then replicate the unchanged training comparison on fresh seeds.** Retain
   all checkpoints and timing-aware behavior labels. Replication is required
   before describing the shift as a stable effect of the shaping method.
3. **Use a placebo if the causal claim needs it.** The already deferred placebo
   direction can help distinguish cost assignment from extra gradient
   availability. Its design must verify that it actually matches the relevant
   reward variation. A weight sweep alone would not resolve that issue.

My priority is the focused robustness evaluation, followed by replication.
Further family searches or reward changes should follow what that test reveals.
Nothing in this recommendation authorizes a launch. The campaign remains
`initial comparison reviewed; decision required`.

## Reproduction and verification

The [analysis script](inbox-campaign-s4021-c4096-ops/deep-review-20261005/analyze.py)
reads all seven saved evaluations, existing audits and training logs. It verifies
95 unique SHA256-bound inputs from the three original review receipts, checks
the paired seeds and initial observations, and reconciles action counts with
their saved results. Pending calls after an environment terminal action are
excluded exactly as in the existing evaluation loop.

New counts are in [diagnostics.json](inbox-campaign-s4021-c4096-ops/deep-review-20261005/diagnostics.json).
The [analysis receipt](inbox-campaign-s4021-c4096-ops/deep-review-20261005/analysis_receipt.json)
binds the figure inputs, script, style and generated outputs. None of the sealed
run artifacts or reports is edited. The analysis script, diagnostics, receipt
and figures are committed at the user's request. Their sealed source evidence
remains in ignored run directories and must be retained locally for reproduction.
Hashes identify the exported files; re-exporting can change PDF metadata and
hashes without changing the numbers or rendered content. Reproduction is checked
through the script's assertions and rendered figures, not byte equality with a
previous export. All five rendered figures and local report links were inspected. The
repository gate passed: Ruff, mypy, 485 tests (seven skipped), and all 17 setup
harness checks. The offline helper also passes its explicit Ruff checks.

From the repository root:

```sh
XDG_CACHE_HOME=/tmp/inbox-cache MPLCONFIGDIR=/tmp/inbox-mpl .venv-test/bin/python pipeline/runs/inbox-campaign-s4021-c4096-ops/deep-review-20261005/analyze.py
```
