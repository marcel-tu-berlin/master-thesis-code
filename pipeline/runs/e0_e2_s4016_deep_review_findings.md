# E0/E1/E2 seed 4016: deep review and results guide

Reviewed 2026-09-27 at the user's request, before authorizing another experiment.
This is a post-campaign interpretation and diagnostic addendum. The original
[campaign findings](e0_e2_s4016_findings.md), raw evidence, primary analysis and
hash-bound review receipts remain unchanged. No seed 4017 run was launched.

## Assessment

The campaign demonstrates successful task learning and substantial additional
compression from the E2 training recipe on held-out instances of `read-table-2`.
The observed effect is broad, survives pairing on successful questions, and
translates into lower measured inference time. The data do not support discarding
the setup as meaningless.

The strongest defensible claim is narrow: on this one seed and fixed-interface
family, E2 reduces the median successful response length by 44.93% relative to
task-only E1 while meeting the declared 5-percentage-point loss margin. It does
not establish zero harm, training-seed robustness, general web-agent efficiency,
or preservation of feedback-based verification. Two real copying errors remain.

The deeper review identifies two important interpretation limits: all test pages
use the same action IDs, and the model can issue the entire solution in one
assistant response. The prompt nevertheless instructs it to call one tool per
turn. This shared prompt/runtime mismatch needs an explicit decision before any
protocol revision; changing it silently would break exact replication.

## 1. Where to inspect the evidence

Start with these files, in this order:

1. This addendum and the [original findings](e0_e2_s4016_findings.md).
2. [Evaluation figure](e0-e2-campaign-ops/deep-review-20260927/evaluation.png)
   ([PDF](e0-e2-campaign-ops/deep-review-20260927/evaluation.pdf)) and
   [training figure](e0-e2-campaign-ops/deep-review-20260927/training.png)
   ([PDF](e0-e2-campaign-ops/deep-review-20260927/training.pdf)).
3. [All 200 final paired questions](e0-e2-campaign-ops/deep-review-20260927/paired_questions.md).
4. [Readable trajectory casebook](e0-e2-campaign-ops/deep-review-20260927/casebook.md):
   both E2 regressions, its gain, a successful shortened example, and both
   jointly correct examples where E2 is longer. This is a deliberately selected
   diagnostic set, not a random sample.

### Per-run files

| Condition | Directory | Human-readable final report |
|---|---|---|
| E0, original base | `e0-read-table-2-s4016/` | [eval_report.md](e0-read-table-2-s4016/eval_report.md) |
| E1, task-only reward | `e1-read-table-2-s4016/` | [eval_report.md](e1-read-table-2-s4016/eval_report.md) |
| E2, relative successful-length cost, weight 0.1 | `e2-read-table-2-s4016/` | [eval_report.md](e2-read-table-2-s4016/eval_report.md) |

Inside each directory:

- `episodes_held_out_table.jsonl`: the initial page, generated reasoning/content,
  calls and arguments, tool feedback, outcome, stop reason, token counts and
  timings for each of the 200 questions. One JSON object per line. Episode index
  is zero-based, so index 151 is line 152.
- `eval_report.json`: aggregate metrics and per-question samples.
- `config.yaml`: the actual frozen experiment recipe.
- `env_stamp.json` and `launch_evidence.json`: stack, source and launch evidence.
- `costs.jsonl`: matched phase start/end records; `phase_exit.json`: exit status.
- `technical_review.json` and `review.json`: integrity audit and hashes of the
  evidence on which the signed-off review depends.

The trained arms additionally contain `train_log.json` (all 300 updates),
`run.log`, `group_observations.jsonl` (416 saved rollouts, not all 9600), and
`scientific_review.json`. Intermediate reports and trajectories are under
`checkpoint-evals/checkpoint-100/` and `checkpoint-evals/checkpoint-200/`.
The run-root evaluation is checkpoint 300, the declared primary endpoint.
`checkpoint-evals/eval_protocol.json` records the checkpoint protocol.

E2's [initial_comparison.json](e2-read-table-2-s4016/initial_comparison.json)
contains the paired statistics, checkpoint comparisons, costs, precision and
multiplicity analysis. The four `masked_*.json` files preserve the case-label
review. The trained runs' automatically generated model-card `README.md` files
are not the authoritative experiment reports.

Adapter weights remain on `gpu-l4`, under
`/workspace/master-thesis-code/pipeline/runs/e{1,2}-read-table-2-s4016/`, in
`checkpoint-{100,200,300,final}/`. All non-adapter result evidence needed for
this review is local. Earlier development runs are archived and must not be
mixed into this comparison.

New descriptive calculations are in
[diagnostics.json](e0-e2-campaign-ops/deep-review-20260927/diagnostics.json).
[analyze.py](e0-e2-campaign-ops/deep-review-20260927/analyze.py) reproduces those
calculations, tables, casebook and plots from existing data and first verifies
the original receipts. The new analysis has its own `analysis_receipt.json`.

## 2. What was compared and checked

Both trained arms start independently from the same original Qwen3-1.7B revision.
Neither is a continuation of the other. Both use bf16, LoRA rank 16/alpha 32,
300 updates, four questions per update, eight rollouts per question, learning
rate 0.00005, 10% warmup, constant schedule, no KL penalty, `naive_sum` and
`scale_rewards: none`. The studied reward component is the only substantive
config difference. E2 uses weight 0.1. No other weight or placebo ran.

Training and evaluation share an eight-assistant-turn limit and a 4096-token
whole-trajectory budget, including tool feedback. The efficiency outcome counts
generated assistant tokens, including reasoning and tool calls. Those are
different accounting quantities and should not be interchanged.

The original reviews checked all 1400 evaluation trajectories across seven
evaluations against their visible tables, action histories, outcomes and report
arithmetic. They checked finite training updates, the saved training captures,
reward arithmetic, saved-tokenizer recounts, checkpoint identities, source/stack
consistency, harvested hashes and cost ledgers. E1 and E2 answer the same 200
held-out seeds with identical initial observations. Checkpoint 300 and final
adapters match. The new diagnostic rechecked all inputs bound by the original
three review receipts. Their hashes remain unchanged.

A useful additional comparability check: the first 32 saved training rollouts
are identical across E1/E2 before reward computation. Subsequent divergence is
expected once their different rewards begin updating the policy.

No corrupted reward, incomplete run, missing checkpoint evaluation or arithmetic
error was identified. This does not make the task broad or the proxies stronger
than their definitions. The prompt/runtime issue below is shared by both arms
and affects interpretation of the agent behavior.

## 3. Learning and primary compression result

| Policy | Correct / 200 | Success | Mean tokens on correct episodes |
|---|---:|---:|---:|
| E0 | 43 | 21.5% | 2027.12 |
| E1, update 100 | 183 | 91.5% | 1506.58 |
| E1, update 200 | 197 | 98.5% | 868.89 |
| E1, update 300 | 199 | 99.5% | 463.95 |
| E2, update 100 | 184 | 92.0% | 1094.62 |
| E2, update 200 | 199 | 99.5% | 372.11 |
| E2, update 300 | 198 | 99.0% | 228.68 |

These successful-subset means describe each arm; the primary contrast pairs
only questions both arms answer correctly. The E0-to-E1 improvement establishes
substantial task-only learning and must not be credited to E2's length reward.
E0's low greedy success also means it was not an already competent agent whose
behavior we merely compressed. E2 studies learning under the shaped reward.

At update 300, E1/E2 have 197 jointly correct questions, two E1-only successes
and one E2-only success. The primary paired median token change is **-44.93%**,
with a 95% bootstrap interval of **-47.33% to -42.75%**. Median absolute change
is -182 tokens; mean paired absolute change is -234.67. The declared requirement
that the upper interval endpoint be at most -10% passes comfortably.

The deeper distribution check shows that this is not driven by a few long tails:

- E2 is shorter on **195/197** successful pairs; 193/197 are at least 10% shorter.
- It is longer on only indices 122 and 178: 376 to 429 tokens, and 474 to 915.
  Both remain correct; their generated reasoning dwells on an unrequested field
  before returning to the required actions.
- Across the same 197 questions, parsed reasoning text falls from a mean 382.68
  to 150.18 tokens. It accounts for **99.07% of the mean token saving**.
- On the 186 jointly correct questions where both policies use one assistant
  turn and three actions, the paired median reduction is still **44.08%**
  (95% interval 42.17%-45.59%). Thus differing action batching does not explain
  away the compression.

Parsed reasoning/content are re-encoded component counts; the remaining tokens
include tool-call syntax, framing and re-encoding differences. The component
breakdown describes emitted text. It does not measure hidden cognitive effort
or prove that the omitted text was causally unnecessary for every future task.

## 4. Success preservation is bounded, not perfect

The two real E2 regressions deserve inspection:

| Index | Required value | E2 action | E1 / E2 tokens |
|---:|---|---|---:|
| 151 | First name `Arlyn`, year `1958` | Writes `Arlin`, copies year correctly, submits | 397 / 237 |
| 182 | First name `Florina`, religion `Judaism` | Copies first name correctly but uses Language value `Punjabi` for Religion | 627 / 227 |

These are policy errors with valid action IDs and correct scoring. The second
is already present at E2 update 200. E2's gain at index 123 fixes E1's use of
label ID 35 instead of textbox ID 36 and correctly submits Hinduism and Greece.

The 95% one-sided upper bound on the probability of an E1-correct/E2-wrong case
is **3.114%**, below the declared 5-percentage-point margin. Gains are not used
to cancel losses in that bound. Exact McNemar p=1.0 is not evidence of equality.

The margin is permissive relative to a 99.5% baseline. Passing it rules out
large enough regression rates under the declared criterion; it does not rule
out smaller but practically relevant harms. Even zero observed losses at n=200
would leave a 1.49% upper bound. The final individual Wilson intervals are
97.22%-99.91% for E1 and 96.43%-99.73% for E2. Do not present 99.5% versus 99.0%
as proof of unchanged reliability.

The efficiency estimate conditions on joint success. It is valid for that
subset but says nothing about the efficiency of the two lost successes. Keeping
the loss table next to the compression estimate prevents that selection from
hiding the trade-off. Two errors are insufficient to establish a systematic
first-name or religion weakness: fields co-occur, subgroup sizes are small,
and one case's first-name answer is actually correct.

## 5. What the task and tool traces reveal

All 200 held-out observations are distinct. They cover all 56 ordered pairs of
target labels, 198 table field orders, 225 distinct target label/value pairs,
and both alignments between table and form order (107 reversed). The generator
chooses five of eight field types and requests two values. There is genuine
variation in what must be read and copied.

However, **every held-out page has textbox IDs 33 and 36 and Submit ID 37**.
The interaction scaffold is fixed. At final evaluation:

- E2 always emits `fill(33, ...)`, `fill(36, ...)`, `click(37)` in one assistant
  response: 200/200 cases, including both wrong submissions.
- E1 finishes 191/200 episodes in one assistant turn. In 198/200 episodes it
  emits multiple tool calls in at least one turn.
- Both policies reuse a common opening reasoning template across all 200
  questions. The values remain task-dependent; repeated phrasing alone does
  not establish a failure to read the page.

This is strong evidence of efficient execution of a learned, fixed form-filling
procedure. It does not establish adaptation to changed IDs, unfamiliar layouts,
different widgets, longer action sequences or actions whose correct continuation
depends on new information returned by a tool.

### Shared prompt/runtime mismatch

[`domains/browsergym/domain.py`](../domains/browsergym/domain.py) prepends
"then call one tool per turn" in training and evaluation. Both runtime paths
permit multiple calls in one generated assistant message. The evaluation loop
in [`eval/agentic_eval.py`](../eval/agentic_eval.py) dispatches those calls in
sequence and stops when the environment is done.

Three tool calls in one response are three actions, but only one policy
generation. Their arguments have already been generated before feedback from
the first fill exists. Consequently, the second fill and Submit cannot be
conditioned on that feedback. This is allowed by execution and contrary to the
plain reading of the prompt instruction. E1 also uses it, so it is not a new
E2-only exploit and does not invalidate the paired arithmetic.

For an exact seed replication, retain the frozen recipe. If the intended
experiment requires one action followed by a new model decision, fixing the
instruction or enforcing that contract creates a new protocol. It changes
context, token use and potentially learning; it cannot be silently combined
with this campaign as though only the seed changed. No such change was made.

### Metric names can overstate what was observed

- `mean_verification_depth` is dispatched actions minus one on terminated
  episodes. E2's value 2.0 is exactly its two fills before Submit. It does not
  demonstrate two verification steps or consultation of action feedback.
- `unsupported_claim_rate` detects bare terminal actions without preceding
  actions. Its zero value does not establish absence of false natural-language
  claims; earlier E0 examples made such claims outside this proxy's coverage.
- The fixed-reference `underthinking_rate` flags correct responses shorter than
  E0's P10. All correct E2 episodes qualify. It is a length flag, not evidence
  that those successful episodes were harmful or unreasoned.

The four declared off-target tests still pass their no-increase margins:
wrong submissions 1 to 2; invalid-action episodes 4 to 0; voluntary no-tool
stops 0 to 0; repeated-action episodes 0 to 0. Both final policies terminate
all 200 episodes and have no generation-cap endings. The multiplicity-adjusted
upper bound on new wrong submissions is 4.00%; the other three are 2.17%.
Full behavioral equivalence remains indeterminate. Many final error measures
are at their floor, limiting what this family can say about broader RQ2 harms.

The 24-case masked review supports the recorded labels, but was not an
independent blinded annotation study. Prior aggregate outcomes were known;
the six sampled short correct controls all happened to be E2, while other
strata mostly came from E0. No replacement sample was selected after unmasking.

## 6. Training signal, sample reuse and 300 updates

There are 500 fixed training seed rows. At four questions per update, 300
updates mean 1200 question presentations: 2.4 passes, with 300 rows visited twice
and 200 three times. Eight sampled attempts per presentation produce 9600
training rollouts. These are not 9600 different tasks. Each repeated visit
samples new trajectories from the current policy on the same deterministic page.

The 200 evaluation seeds are disjoint from the training seed block. Of the
50 distinct training seeds whose full observations were captured, none exactly
matches a held-out observation after removing the common prompt lead-in. That
is not an exhaustive content-collision audit of all 500 training pages. Shared
values and the repeated interface remain part of the task distribution.

There is no evidence here that 300 updates exhausted the available task supply
or caused a broad accuracy collapse. There is also no proof against structural
overfitting: held-out instances use the same family and action scaffold.

| Training measure | E1 | E2 |
|---|---:|---:|
| Groups with nonconstant composed reward | 410/1200 (34.17%) | 1189/1200 (99.08%) |
| Zero logged-gradient updates | 108/300 | 0/300 |
| Zero logged-gradient updates in last ten | 7/10 | 0/10 |

Task reward stops distinguishing sampled responses once they all succeed. E2
continues distinguishing their lengths. It therefore changes both the reward
assignment and how many groups provide gradient: roughly 2.9 times as many
active groups in this run. The observed comparison estimates the effect of the
whole intervention, not an isolated mechanism at equal gradient exposure.

The relative sigmoid cost stays between zero and one on successes and is zero
on failures; at weight 0.1 a successful rollout still has higher scalar reward
than a failed one. This ordering does not guarantee that policy updates preserve
every success. Relative ranking can continue favoring shorter successful
rollouts without an absolute target length at which the pressure switches off.

The last 100 updates were consequential:

- E2: 199 to 198 successes; paired median length falls a further **38.05%** on
  198 jointly correct cases. Its action structure was already one turn/three
  actions at update 200, so later savings mainly compress the generated text.
- E1: 197 to 199 successes; paired median length falls a further **49.51%** on
  196 jointly correct cases.
- Logged update durations for that last third sum to about 4.69 GPU-hours for
  E2 and 7.64 for E1. These sums exclude phase overhead and checkpoint evaluation.

Thus 200 updates could be a future budget trade-off, not a conclusion that the
final third was useless. Selecting E2 update 200 because it has one more success
would be post-hoc selection. Keep update 300 primary; select any shorter horizon
in advance for a new experiment and apply it equally to both arms. Intermediate
checkpoints are repeated observations of these models, not extra training seeds.

The placebo is a **separate training and evaluation arm**, not an internal part
of E2. It shuffles the shaped component, including failure zeros, within prompt
groups. It preserves the cost multiset but does not guarantee matching total
reward variance or gradient noise. It would test sensitivity to meaningful
length assignment, with those limitations. The prepared placebo uses weight
0.2; it cannot silently become a dose-matched control for the current 0.1 arm.

## 7. Costs and remaining uncertainty

| Condition | Training GPU-hours | Research evaluation GPU-hours | Total |
|---|---:|---:|---:|
| E0 | 0 | 3.991 | 3.991 |
| E1 | 28.134 | 7.471 | 35.605 |
| E2 | 22.582 | 5.007 | 27.588 |
| Total | 50.716 | 16.469 | 67.185 |

Mean final policy-inference time is 67.08 seconds for E0, 19.25 for E1 and 9.42
for E2. E2 saves **9.83 seconds per question**, about **51.1%**, with a paired
95% interval of 8.36-11.72 seconds. The joint-success subset and end-to-end
episode timing give essentially the same conclusion. The improvement is not
an artifact of counting failed episodes as shorter successful work.

Training E2 instead of E1 was also 5.55 GPU-hours cheaper in this observed pair.
If E1 is already available and its cost is sunk, paying for a new full E2 run
would require approximately 8267 similar deployment episodes to recover its
training time through the measured inference saving. Against the base model,
that figure is about 1410. These estimates exclude research evaluations and
condition on this one hardware/run measurement. They are not monetary, energy
or hardware-independent claims.

One replication of the same full trio has an observed precedent of about
67 GPU-hours (2.8 days), plus review and operational delays, rather than a
guaranteed completion time. A lower-cost diagnostic and a complete replication
answer different questions and should be named accordingly.

The largest unresolved uncertainties are:

1. **Training-seed variability.** One trained model per arm cannot estimate it.
   The tight episode bootstrap interval does not substitute for seed replication.
2. **Family/generalization limits.** No changed-layout, changed-ID, shifted-family,
   longer-horizon or genuinely feedback-dependent test was run.
3. **Mechanism.** Length assignment and increased gradient activity are coupled.
   No placebo or other weight was tested; 0.1 has not been shown optimal.
4. **Reliability at finer margins.** Two regressions in 200 and the predeclared
   5pp margin cannot certify a sub-percentage-point error tolerance.
5. **Evaluation coverage.** Results concern greedy decoding under this prompt,
   model, budget and tool interface. Stochastic reliability, unrelated abilities
   and post-training general capability retention were not evaluated.

The existing multiplicity sensitivity still passes the success, token and four
off-target requirements under a six-endpoint Bonferroni calculation. This is
useful reassurance about those endpoints, not a solution to items 1-5. Earlier
checkpoints, length quartiles, field subgroups and this addendum are exploratory.

Seed 4017 would change model-training randomness, the training task block and
the held-out task block together. It is a replication of the full procedure,
not an estimate of optimizer randomness on fixed questions alone. Per-seed
comparisons remain paired; treating all episodes from several seeds as
independent trained models would overstate precision.

## 8. Implications for next steps

**Keep this result.** It is an informative positive first experiment, with a
clear scope and two disclosed regressions. Neither the fixed interface nor the
shared tool-call contract issue requires discarding its paired token result.

**Decide what behavior the next experiment must test before spending another
full trio's cost.** If batched form filling is accepted within the thesis claim,
seed 4017 at the unchanged recipe is the direct next replication. If the central
claim requires decisions conditioned on intermediate tool feedback, the current
family does not establish it. Qualify that behavior in another task or declare
a revised protocol explicitly; do not silently repair this campaign in place.

**Use a cheap family qualification before another large family run.** Verify
visible actionable targets, deterministic resets, a truthful success oracle
with correct and deliberately wrong actions, usable failure feedback, and
nontrivial model performance on development seeds. For the broader agent claim,
also require a task whose next correct action genuinely depends on information
obtained after an earlier action. Merely exposing several tools is insufficient.

Historical family screens already identified problems with
`click-tab-2-medium`, `click-collapsible-2-nodelay` and `search-engine` under the
then-tested interface. These are not ready alternatives without requalification;
see the archived [oracle review](archive/development-2026-09-24/family_oracle_findings.md)
and [model screen](archive/development-2026-09-24/family_screen_findings.md).

**Then replicate the chosen declared protocol before expanding doses.** Keep
weight 0.1, the matched E1 control, each seed's own E0 reference and a fixed
primary checkpoint. A second seed adds valuable evidence but still gives a weak
estimate of across-seed uncertainty. Broader family evidence addresses a
different limitation and remains necessary for broader claims.

**Reserve a dose-matched placebo for a mechanism question.** It is not a
prerequisite for reporting this total-effect contrast, and the prepared 0.2
placebo is not matched to the current dose. Do not move to E3 on the strength of
these final off-target floors alone. No new run, protocol edit, commit or push
was performed as part of this review. Automatic advancement remains disabled.

## Reproduce this addendum's diagnostics

From the repository root, using the existing CPU test environment:

```bash
rtk proxy env MPLCONFIGDIR=/private/tmp/thesis-deep-review-mpl \
  XDG_CACHE_HOME=/private/tmp/thesis-deep-review-cache \
  .venv-test/bin/python \
  pipeline/runs/e0-e2-campaign-ops/deep-review-20260927/analyze.py
```

This reads the sealed campaign artifacts and overwrites only the derived
diagnostics, casebook and plots in this addendum's directory. It launches no
GPU work. The original audit scripts remain in `e0-e2-campaign-ops/`; their
outputs and hashes are bound by the original run receipts. The new
`runtime_snapshot.json` records the read-only GPU/watcher check and the exact
pinned MiniWoB generator source inspected for this review.
