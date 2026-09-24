# Applied length reward audit, seed 4009

Reviewed 2026-09-23 after the first E1/E2 result. This is a post-hoc diagnostic,
not a changed endpoint or a new experiment. No runtime, reward, config, adapter,
or original review receipt was changed. The original comparison remains in
[the first-contrast findings](table_first_s4009_findings.md).

## Verdict

The configured cosine component was applied. No disabled component, inverted
correct-answer preference, omitted reasoning tokens, accidental placebo,
normalization cancellation, or disconnected reward-to-loss path was found.
There is evidence of a small brevity signal relative to correctness separation.
That is a plausible mechanism for the outcome, not a demonstrated sole cause.
This run does not establish a Qwen3-1.7B capacity limit.

## Config, implementation and actual training

- The frozen E2 config enables `token_length`, weight 0.4, max_len 4096,
  correct endpoints 1.0/0.5, wrong endpoints -1.0/-0.5, placebo false.
  Task reward has weight 1.0; E3 is disabled.
- Training used `naive_sum`, `scale_rewards: none`, DAPO, one update per
  rollout batch, 4 prompts x 8 rollouts, microbatch 1, gradient accumulation 32,
  learning rate 5e-5, KL 0, and token-level truncated vLLM importance correction.
  Liger and vLLM sleep were off. Train logs name both actual reward components.
- Model and training dictionaries match the E1 pilot whose adapters supply the
  comparison. E2 starts from the same base, not from the trained E1 adapter.
  The first 32 sampled completions and seeds are identical across both runs.
- Registry -> builder -> composer -> observation wrapper -> native TRL reward
  dispatch -> mean-centered advantages -> DAPO loss was inspected. The wrapper
  returns the original composed rewards. TRL applies no second component weight.
- All 38 admitted source files, 25 admission inputs and 30 final-review inputs
  match their canonical remote hashes. Local source/admission checks also pass.

The actual total reward, with L capped at 4096 for this formula, is:

```
correct:   R =  1.3 + 0.1 * cos(pi * L / 4096)
incorrect: R = -0.3 - 0.1 * cos(pi * L / 4096)
advantage = R - mean(R for the same prompt's eight rollouts)
```

This is a correctness-dependent reward, not an unconditional token tax. Shorter
correct responses receive more reward. Longer incorrect responses receive less
negative reward, intentionally. In mixed groups a relatively long correct
response can still have positive advantage because failures lower the group
mean. A large positive raw length-component contribution does not measure the
remaining brevity advantage after centering.

| Assistant content tokens | Correct total reward | Incorrect total reward |
|---:|---:|---:|
| 500 | 1.39274 | -0.39274 |
| 600 | 1.38960 | -0.38960 |
| 800 | 1.38176 | -0.38176 |
| 1000 | 1.37200 | -0.37200 |
| 4096 | 1.20000 | -0.20000 |

Saving 200 tokens from 800 to 600 buys 0.00784 reward. Correct versus incorrect
at 800 tokens differs by 1.76352. The slope becomes flatter near the short end
of the 4096-token cosine curve. Raising its coefficient also increases the
correctness gap and the incentive to make unsuccessful attempts longer; it
does not isolate a stronger brevity cost.

`scale_rewards: none` still subtracts the group mean. It retains these small
raw differences rather than inflating them to unit variance. This is intended
dose behavior, not evidence that the setting was ignored. The installed TRL
1.6.0 source confirms it; the [TRL documentation](https://huggingface.co/docs/trl/grpo_trainer#computing-the-advantage)
also distinguishes centering from variance scaling. Changing this setting would
change the protocol and would not isolate the reward's length dependence.

## Recorded evidence and replay

Both arms retain 300 update metric rows and 13 full rollout captures at updates
1, 100, 200 and 291-300: 32 rollouts per capture, 416 per arm. This is 4.33% of
9,600 sampled training trajectories per arm, with a deliberately late-heavy
schedule, not a random sample of the whole run.

For E2, all 416 captured token counts were recomputed using the pinned real
tokenizer. Raw cosine values and composed rewards matched independent formulas
to 1e-12. Every within-correctness ranking had the intended direction. Batch
means matched the actual native trainer logs. Visible-table checks also verified
correctness, terminal feedback and absence of messages after completion for all
416 captured E2 trajectories. The captures include 350,666 tokens stored in
`reasoning_content` out of 407,957 reward-counted model tokens; those tokens were
included, unlike the void historical campaign.

CPU replay through the installed TRL and actual configured reward builders
reproduced advantages for all 13 E2 batches within 1.92e-7. Native DAPO loss and
derivatives with respect to selected-token log probabilities matched the scalar
reference, including zero loss gradient for tool-feedback positions. Generation,
token arrays and log probabilities were fixtures; no policy was loaded. This
demonstrates reward arithmetic and loss connectivity, not a reconstruction of
every historical model-parameter gradient. Earlier readiness evidence covers the
same qualified loss implementation with real captured token arrays.

All 300 E2 updates have finite logs, nonzero gradient norms and zero reported
zero-variance groups. Its maximum gradient norm was 0.15494, below the resolved
clip threshold 1.0. PPO clipping stayed zero; mean vLLM importance ratios stayed
near 1. These aggregate checks give no indication that clipping removed the
reward signal, but do not identify individual component parameter gradients.
E1 has 109 zero-gradient update rows because task reward can saturate; E2 has
none. This difference is part of the treatment, not proof of better learning.

In the final ten captured updates:

| Prompt groups | Count | Mean absolute centered advantage |
|---|---:|---:|
| All eight rollouts correct | 36 | 0.00471 |
| Mixture of correct and incorrect | 4 | 0.38674 |

All-correct groups still have a median 383.5-token range but only a 0.01582
reward range. Their signal is real and small. Subtracting each correctness
class's mean reward within a group isolates within-class length differences;
the complementary class-mean term accounts for 99.877% of summed squared
advantages across these 40 groups. This is a reward-space diagnostic, not a
claim that 99.877% of actual parameter gradients came from correctness. Adam,
token weighting and shared parameters prevent that inference.

## What the observed trajectory supports

Mean correct held-out tokens at the saved observations were:

| Update | E1 | E2 |
|---:|---:|---:|
| 100 | 1331.75 | 1094.35 |
| 200 | 700.89 | 777.05 |
| 300 | 633.30 | 892.01 |

These means condition on each arm's own successes. The declared primary paired
estimate remains +47.04%, CI +39.80% to +56.38%, on 192 jointly correct questions.
Recomputing the paired median using the reward's re-encoded token ruler instead
gives +50.60%. That is an exploratory measurement check, not a replacement
endpoint: template accounting does not explain away the adverse result.
Exact eval tokens exceed reward-ruler tokens by means of 42.88 in E1 and 43.44
in E2. Correct-response reasoning means are 554.61 and 812.82 tokens, while both
have approximately one assistant turn and three actions. The extra length is
principally reasoning, not additional necessary tool use.

E2 learned successful behavior faster early on, became shorter, then E1 overtook
it and E2 length rose between 200 and 300. Training also shows the difference:
the last 50 updates average 530.76 versus 713.76 trajectory tokens for E1/E2.
Training trajectory lengths include tool bookkeeping and are not interchangeable
with the reward's assistant-only lengths or greedy held-out lengths.

The same 1.7B model already produces substantially shorter correct responses in
E1. Capacity remains a possible limit to further improvement, but it is not the
first explanation supported by this contrast. One seed, a correctness-coupled
objective, weak late brevity advantages and different learning trajectories
leave the causal explanation unresolved.

## Recommended next decision

Keep this adverse result intact. Before spending on a larger model or another
full run, declare a small development-only continuation diagnostic from the
existing competent E1 checkpoint, with a matched task-only continuation control.
Test a cost with a useful slope in the development response-length range and
separate its brevity effect from added correctness reward. A correct-only length
cost with constant reward on failures is one candidate; its success ordering
and failure behavior require offline checks before admission. Choose the formula,
coefficient and endpoint before new runs, using development data, and reserve
fresh confirmation data for any later claim. This is a proposed protocol change,
not an admitted experiment. Switching model size or normalizing groups now would
add another explanation before resolving this one.

For that diagnostic, retain actual advantages and assistant masks at fixed steps,
and log within-correct length spread and centered length-signal magnitude.
Current component raw means/stds and total gradient norms cannot tell us how much
of a parameter update comes from length versus correctness. A fixed-batch
component-gradient check would resolve that narrower question before training.

## Verification and limits

Evidence and runnable scripts are in
`table-first-s4009-ops/reward-audit-20260923/`: `recorded_audit.json`,
`native_audit.json`, `paired_analysis.json`, `analyze_rewards.py` and
`native_reward_check.py`. The latter runs CPU-only on the pinned box environment.
Three initial diagnostic-harness attempts failed due to CPU Accelerator state,
a missing fixture reset method, and a missing reference-loss settings field.
Their logs are retained. These were fixed in the new audit script; no training
implementation was changed. The final replay passed.

The installed stack emits a TRL support-range warning for vLLM 0.19.1 (declared
range ends at 0.19.0). Both arms share the qualified stack. No present evidence
connects this warning to the length outcome; it is a compatibility caveat, not
a demonstrated failure or grounds to silently upgrade the completed experiment.

The project check passed: 450 tests, six skips, formatting/lint/type checks and
17 setup checks. The audit scripts pass their direct assertions and lint checks.
No full-training per-token log-probability or component-gradient history exists,
so the logs cannot establish the unique cause of the adverse result.

Reporting corrections: the original `review.json` reason says four introduced
invalid-action episodes; the recomputed data show five (6.26% simultaneous upper
bound). The original receipt remains unchanged to preserve its bound hash. The
earlier findings' statement of eight records per capture was also inaccurate:
there are four groups of eight, 32 records per captured batch.
