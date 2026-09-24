# P2 settings analysis: do not enable all tested settings

Date: 2026-09-11. Status: complete; decisions recorded in ADRs 0010-0013.

The present evidence supports retaining the baseline settings. `num_iterations: 2`
is a credible speed candidate with an unresolved quality tradeoff. Liger and
sleep mode each expose a correctness problem in the installed stack, so neither
qualifies as a transparent performance optimization. The subsequent decisions
retain `num_iterations: 1`, disable Liger and sleep mode, and adopt the explicit
final recipe in decision 0010. Frozen run data remain unchanged.

## Results at a glance

All four arms completed 50 optimizer steps and saved their final adapters. The
batch took 21h27m. The raw logs contain no non-finite numeric values, traceback,
OOM, or ERROR entry. Completion establishes execution, not scientific validity.

| Arm | Mean seconds/update | Change in update time | Train runtime | Fresh batches | Raw mean training reward | Mean assistant tokens | TRL clipped fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 415.0 | reference | 5.88 h | 50 | 0.64875 | 2196 | 4.19% |
| Liger | 395.2 | -4.77% | 5.64 h | 50 | 0.66250 | 1816 | 3.19% |
| Two updates per rollout batch | 269.0 | -35.17% | 3.85 h | 25 | 0.58250 | 2316 | 8.63% |
| Sleep mode + memory utilization 0.6 | 426.6 | +2.80% | 6.04 h | 50 | 0.52438 | 2242 | 4.31% |

`step_time` sums the timed training micro-steps, including generation and backward
work. It excludes some outer-loop overhead; the separate `train_runtime` column
comes from the final trainer summary. Its reductions are 4.08% for Liger and
34.46% for reuse; sleep costs 2.77% more. Removing the first five updates leaves
the same ranking: 413.1 / 390.2 / 266.6 / 426.6 seconds respectively.

The reuse arm's raw reward must not be compared with the baseline's full-run
mean as a matched estimate. Its 25 generation batches cover only the first half
of the baseline's question sequence. The properly matched reward contrast is
given below.

## What was controlled, and what was not

Frozen configs confirm Qwen3-1.7B, seed 42, click-menu-2 only, 50 updates, four
prompt-groups per update, eight rollouts per group, micro-batch size one, an
8192 total context budget with 4096 reserved for the prompt, and eight tool
iterations. Rewards are env reward only, combined with `naive_sum`.

After excluding experiment names and descriptions, the Liger and reuse configs
each differ from the baseline in exactly their named setting. The sleep config
changes two fields: sleep mode is enabled and GPU memory utilization rises from
0.3 to 0.6. The runner also disables expandable allocator segments for sleep
compatibility. This arm tests that combined configuration; it cannot isolate
sleep mode's causal effect from the other changes.

All four environment stamps are identical: TRL 1.6.0, Liger 0.8.2, vLLM
0.19.1+cu130, PyTorch 2.10.0+cu130, transformers 5.12.0, PEFT 0.19.1, and
OpenEnv commit `024eedc90305cc8bd7a5b44f44d1b987102e957b`. The batch ran in the
fixed order base, iter2, Liger, sleep, on the same configured physical GPU.
Execution order was not randomized and hardware load/clock telemetry was not
recorded throughout, limiting fine-grained performance attribution.

The runtime recipe lines resolve defaults omitted by the frozen configs:
`adamw_torch_fused`, `constant_with_warmup`, learning rate 5e-5, KL beta zero,
and **`scale_rewards=group` in every arm**. The last setting matters: the
planned lambda campaign uses `scale_rewards: none`. `naive_sum` specifies how
components are combined; it does not disable TRL's subsequent group scaling.
These probes do not validate the final campaign recipe. Future configs should
state these load-bearing settings explicitly.

The installed GRPO loss default is `dapo`. It is shared by the arms, but the
two implementations do not apply the same denominator, as explained below.

## Paired evidence and uncertainty

Pair Liger and sleep on generation batch 1 through 50. Pair reuse batch k,
logged at update 2k-1, with baseline update k, for k=1 through 25. The dataset
construction and TRL RepeatSampler establish the shared seed order. The logs do
not contain episode IDs or full sampled trajectories, so pairing cannot be
independently reconstructed from per-episode records. Matching questions does
not imply matching sampled answers or matching policy states.

| Contrast | Matched batches | Mean reward difference | Positive / negative / tied | SE | Approximate 95% band |
|---|---:|---:|---:|---:|---:|
| Liger - baseline | 50 | +1.375 percentage points | 22 / 20 / 8 | 1.101 pp | -0.784 to +3.534 pp |
| Reuse - baseline first 25 | 25 | -1.625 percentage points | 9 / 11 / 5 | 1.532 pp | -4.628 to +1.378 pp |
| Sleep - baseline | 50 | -12.438 percentage points | 6 / 42 / 2 | 1.669 pp | -15.709 to -9.166 pp |

The existing `probes.p2_compare` results reproduce exactly. The bands are
mean +/- 1.96 SE across paired batches, not independent-run confidence
intervals. Training changes the policy over time, adjacent observations can
depend on one another, and there is only one seed. Neither a small mean nor a
band spanning zero establishes equivalence or non-inferiority. No acceptable
quality-loss margin was specified before these data were collected.

As an exploratory dependence check, 10,000 circular moving-block resamples
with blocks of five and ten batches give the following ten-batch sensitivity
bands: Liger reward -0.94 to +4.13 pp; reuse reward -4.25 to +1.63 pp; sleep
reward -17.69 to -7.44 pp. They preserve local dependence but still assume more
stationarity than a changing policy provides. They support the same cautious
reading, not formal campaign-level inference. Liger's time difference widens
to -49.3 to +8.6 seconds, illustrating why a 4.8% observed saving is not a
well-established kernel speedup.

## Liger: promising purpose, failed equivalence gate

The reward trajectory looks competitive: the mean paired gain is 1.38 pp,
with 22 positive versus 20 negative batches. That is no clear improvement and
does not establish equivalence. Meanwhile, length changes substantially:
the mean falls 17.3%, and the final ten-batch mean is 1264 tokens versus 2293
for the baseline, a 44.9% reduction.

| Updates | Baseline reward | Liger reward | Baseline tokens | Liger tokens | Baseline seconds/update | Liger seconds/update |
|---|---:|---:|---:|---:|---:|---:|
| 1-10 | 0.5844 | 0.5781 | 2228 | 2251 | 425.5 | 435.1 |
| 11-20 | 0.5969 | 0.5938 | 2038 | 2098 | 397.9 | 413.6 |
| 21-30 | 0.6250 | 0.6188 | 2011 | 1837 | 389.3 | 390.1 |
| 31-40 | 0.6375 | 0.6969 | 2412 | 1629 | 442.0 | 398.0 |
| 41-50 | 0.8000 | 0.8250 | 2293 | 1264 | 420.1 | 339.0 |

Liger is slower early and faster after its completions shorten. These data do
not separate time saved by changing the workload from time saved by the fused
kernel. The token figures also pool successes and failures; they are not the
thesis's held-out, both-correct efficiency measure.

The source audit identifies a concrete normalization mismatch. The baseline
DAPO path divides each micro-batch's loss numerator by the total model-token
count across the generation batch. `compute_liger_loss` does not pass
`num_items_in_batch` to Liger. Liger consequently falls back to the current
micro-batch's active-token count, and TRL divides that result by gradient
accumulation steps. At micro-batch size one this gives each trajectory equal
weight instead of giving each active token equal weight. Longer and shorter
trajectories therefore exert different relative influence between arms.

This is not just a displayed-loss discrepancy. For a simple batch with lengths
one and three, advantages +1 and -1, and probability ratios one, the baseline
loss is 0.5 and the Liger path's loss is 0.0. The derivative with respect to
the first trajectory's ratio is -0.25 versus -0.5. The reproduction script
checks both the missing argument in the archived installed source and this
algebraic counterexample. It does not execute a fused GPU gradient test.

The real first batch has identical logged reward and length summaries in the
two arms, yet loss is -0.014698 versus -0.000126 and gradient norm is 0.07175
versus 0.05302, a 26.1% difference. Over the run, Liger's mean gradient norm is
30.1% higher. These observations are consistent with materially different
updates; they are not a numerical parity check on identical saved tensors.

Tool masks and vLLM correction ratios are passed to the fused loss, so their
omission is not the problem found here. There is also a numerical-path
difference: Liger computes selected log probabilities in float32, while the
ordinary bf16 path supplies the old log probabilities. Liger logs a mean policy
clipping fraction of 0.513% with one update per batch, against exactly zero for
the baseline. Recheck log probabilities and gradients after addressing the
normalization mismatch; do not attribute all residual differences to one cause.

Liger does not log entropy, and its clipping key is `clip_ratio` rather than
`clip_ratio/region_mean`, which the current generic training overlay expects.
Missing telemetry is not a zero value.

Recommendation: do not adopt this Liger path as an equivalent optimization.
If the longer context budget is needed, prioritize correcting and verifying
the loss path, then run the 16k memory check. The current probe contains no peak
training-memory measurement and used a 4096 completion budget throughout. It
proves neither a specific memory saving nor 16k feasibility. The old-logprob
forward still uses the ordinary vocabulary-logit path, so even a corrected
fused backward does not remove every large allocation.

## Sleep mode: reject the tested configuration

Sleep mode plus memory utilization 0.6 is 2.8% slower and loses 12.44 pp in
paired training reward. It loses on 42 of 50 matched batches. The final ten
updates average 0.5625 reward versus the baseline's 0.8000. The observed clipped
fraction is almost unchanged, making a large increase in that diagnostic an
unconvincing explanation for the reward deficit.

The sampler/trainer discrepancy grows with training:

| Updates | Baseline mean absolute log-probability gap | Sleep gap | Sleep mean importance ratio |
|---|---:|---:|---:|
| 1-10 | 0.0179 | 0.0208 | 1.0000 |
| 11-20 | 0.0185 | 0.0504 | 0.9985 |
| 21-30 | 0.0185 | 0.0843 | 0.9936 |
| 31-40 | 0.0185 | 0.1219 | 0.9842 |
| 41-50 | 0.0189 | 0.1713 | 0.9708 |

The installed call sequence explains a serious failure mode. TRL first syncs
the trained, merged adapter weights to vLLM. With sleep enabled, `generate`
then calls `collective_rpc("reload_weights")` without supplying updated weights.
In installed vLLM, that call loads weights from the original model on disk.
The freshly synchronized policy is overwritten before sampling. In the
multi-turn loop, generation sleeps again after each turn, while synchronization
is guarded by optimizer-step number.

This is a source-backed explanation consistent with the growing probability
gap and flat reward trajectory. There is no saved per-turn weight checksum,
so this analysis does not claim a direct runtime measurement of which tensors
were loaded during the completed probe. The exact counterfactual improvement
after a fix is unknown. The trial does not establish that a correctly
implemented sleep mode intrinsically reduces quality.

Recommendation: leave sleep disabled. If later memory requirements justify
revisiting it, verify policy preservation across both optimizer updates and
every tool turn. Simply deleting the reload call is not a validated repair:
level-2 sleep discards weight memory, and the next turn still needs valid
trained weights. Retest sleep with memory utilization held fixed before
separately probing a larger cache.

## Two updates per batch: real update throughput, unresolved value

The 35.2% reduction in time per optimizer update is real for this trial.
Generation updates average 438.5 seconds; the intervening reuse updates average
99.5 seconds. Thus the second update adds about 22.7% of a generation update's
time, larger than the earlier smoke-test estimate. Full trainer runtime falls
by about 2.03 hours.

However, 50 updates now use 25 fresh batches: 100 distinct prompt seeds and
800 generated rollouts, versus 200 seeds and 1600 rollouts for the baseline.
At equal fresh-question coverage, reuse's 50 updates take 32.8% more timed
work than the baseline's first 25 updates, buying twice the updates on those
questions. The experiment measures cheaper updates, not faster acquisition of
fresh experience or proven faster learning.

Matched by the 25 fresh batches, reward is 0.58250 versus 0.59875, a -1.625 pp
difference with an SE of 1.532 pp. There is no demonstrated quality gain from
the additional updates. This comparison also evaluates different policy
states and warmup exposure at each matched question batch; it cannot replace
evaluation of the final checkpoints.

On the same matched batches, mean length rises from 2090 to 2316 tokens
(+10.8%) and the TRL clipped fraction rises from 2.88% to 8.63% (+5.75 pp).
These are warning signals, not proof of a persistent generalization penalty.
Dependence-sensitive bands are broad, and the length increase is concentrated
in the later portion of this short trial.

There is no obvious optimizer instability: mean gradient norm is 0.0793,
maximum 0.1268, importance ratios remain close to one, and the mean clipped
policy-ratio fraction on reuse updates is 0.558%. The two zero-gradient updates
come from one reused batch whose prompt-groups all have constant rewards.

Recommendation: keep `num_iterations: 1` as the conservative setting. If saving
training time is a priority, retain two-update reuse as the next quality-versus-
time candidate after the final recipe is fixed. Use a held-out comparison with
a predeclared acceptable success loss, and compare both equal update count and
equal wall-clock budget. A final-checkpoint eval of these two existing arms is
cheaper than retraining and answers the limited question under group scaling;
it would not settle performance under the planned `scale_rewards: none` recipe.

## Limits that apply to every recommendation

Only one seed and one short run per arm were tested. Training batches are not
independent experiment replications. There are no held-out eval reports for
these four runs, no saved training trajectories, no both-correct token
comparison, and no trained-policy non-termination panel. The raw training
reward is not a final model accuracy estimate.

The logged `completions/mean_length` counts model-generated tokens, excluding
tool responses. `completions/clipped_ratio` flags sequences whose final token
is neither EOS nor padding. It is a training diagnostic, not a direct
environment-non-termination rate or a complete stop-reason audit. Its
`mean_terminated_length` companion refers to token endings, not environment
success. None of these numbers alone answers the E2/E3 pilot criteria.

All runs emit the already-known warning that vLLM 0.19.1 is outside TRL's
advertised supported range ending at 0.19.0, plus an NCCL teardown warning.
Those shared warnings do not isolate the arm differences or prove that a
version downgrade fixes them. The inspected source files match their installed
wheel RECORD hashes; no ad hoc source modifications were found in these seven
files. This was checked after the runs, not recorded as a source hash at launch.

No combined Liger + reuse + sleep arm was tested. Their effects cannot be added
or multiplied to predict an all-settings configuration, especially given the
two source-level correctness findings.

## Adopted decisions

1. Decision 0010 freezes the new campaign recipe, including
   `scale_rewards: none`.
2. Decision 0011 disables Liger until identical-input loss and gradient parity
   are established.
3. Decision 0012 keeps one optimizer update per fresh rollout batch.
4. Decision 0013 disables vLLM sleep mode and retains GPU memory utilization
   0.3.

No new GPU run was launched as part of the analysis or decision cleanup.

## Reproduction and source evidence

From `pipeline/`:

```bash
../.venv-test/bin/python -m probes.p2_compare runs/probe-p2-base runs/probe-p2-iter2 runs/probe-p2-liger runs/probe-p2-sleep
../.venv-test/bin/python -m runs.probe_p2_analysis
```

`probe_p2_analysis.json` stores the readout, raw-file SHA-256 hashes, all metric
summaries, block sensitivity results, and validation results.
`probe_p2_source_audit.json` stores the inspected installed functions, source
line numbers, package versions, file hashes, and wheel-record checks. Those
installed sources are the primary evidence for the two implementation findings.

Related upstream references: [TRL v1.6.0 trainer](https://github.com/huggingface/trl/blob/v1.6.0/trl/trainer/grpo_trainer.py)
and [Liger v0.8.2 normalization](https://github.com/linkedin/Liger-Kernel/blob/v0.8.2/src/liger_kernel/chunked_loss/fused_linear_ppo.py)
document the distinct loss paths and optional batch denominator.
[TRL v1.6.0 generation](https://github.com/huggingface/trl/blob/v1.6.0/trl/generation/vllm_generation.py)
and [vLLM v0.19.1 weight loading](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/worker/gpu_model_runner.py)
provide the corresponding synchronization/reload implementations. Installed
wheel line numbers differ from the Git tags, so use the archived excerpts for
the precise code audited here.
