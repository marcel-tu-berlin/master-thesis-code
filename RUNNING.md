# Running

What is executing on the GPU box (`ssh gpu-l4`). Rows leave the table once the
results are harvested. Update rules are in CLAUDE.md.

Updated: 2026-08-06 11:54 UTC (box time)

| run | phase | pid | started | ETA |
|---|---|---|---|---|
| e24bs4 + e25bs4 poly pair | train 150 steps each, then eval, sequential | 2268241 (batch) / 2268243 (train) | 11:54 UTC Aug 6 | ~36h, rate-dependent |

## The poly cosine pair, re-run unconfounded

`batch_size: 4`, `max_steps: 150`, otherwise identical to e24/e25 - the `diff`
is three lines each. Log `/workspace/e24_e25_bs4.log`, configs
`/workspace/e2{4,5}bs4-*.yaml`.

This is the first cosine test that is clean on both known defects at once:

- **Token counting.** e24/e25 ran 2026-07-29 with `model_token_count` reading 8%
  of the real completion, because Qwen3 puts the think block in
  `reasoning_content`. Fixed since, verified on the box before the e28 launch.
- **The batch-size confound below.** At `batch_size: 4` a step is dead only if all
  four groups saturate (0.46^4 = 4.5%), so the env-only baseline gets a gradient
  on roughly 95% of steps instead of 43%, and stops being trained 300x less than
  the arm it is compared against.

150 steps at batch_size 4 is 600 prompts per arm, double the 300 the old runs
saw, with far better conditioned updates.

Cost, corrected: poly is **not** cheaper per step than browsergym. The stored
logs show e24 took 9h50m and e25 8h17m at 95-118 s/it, the same rate. The earlier
"~6h/arm" came from a 2048-budget run and was wrong. Poly wins only because the
pair is two arms rather than three: about 36h against 84h for browsergym.

ASSUMPTION: `gpu_memory_utilization: 0.23` leaves vLLM roughly 5 GB, which cannot
hold 32 sequences at 5120 tokens, so it schedules within budget and the run is
slower than 4x rather than failing. Same trade as the browsergym probe, where it
held. Watching the step rate.

## batch_size 1 confounded every treatment-vs-baseline comparison ever run

The batch-size probe passed, and checking whether the defect reached other runs
turned up something larger. **No config in this repo has ever set
`batch_size`**, so every experiment trained one prompt per optimizer step. That
does not hit all arms equally:

```
                                       gradient   grad_norm median
ENV-ONLY arms (baselines)
  e10  poly env-only                      41%        2.51e-04
  e12  poly env-only hibudget             41%        2.25e-04
  e24  poly env-only 4k                   43%        2.32e-04
  e20  countdown env-only                 50%        5.85e-03
  e27  browsergym env-only (x3)         50-54%    2.5e-03..9.5e-03
  e26  finqa                               3%        1.47e-04

COSINE / ENTROPY arms (treatments)
  e11  poly cosine hibudget                74%        8.70e-02
  e13  poly cosine w4                      79%        7.89e-02
  e17  poly cosine w16                     71%        7.63e-02
  e18  poly cosine w16 naive_sum           73%        8.16e-02
  e25  poly cosine 4k                      77%        5.11e-02
  e21  countdown cosine                    68%        4.44e-02
  e19  poly entropy                       100%        1.22e-01
```

Three matched pairs - e12/e11, e24/e25, e20/e21 - and the split is clean along
env-only versus shaped-reward. Every treatment arm got roughly twice the steps
with a gradient, at a magnitude about 300x larger.

**The mechanism is not a bug, which is why it went unseen.** `env_reward` is
binary, so a prompt-group whose 8 rollouts share a correctness label has zero
within-group variance and contributes nothing after z-scoring. The cosine reward
is continuous in length and always has variance. At batch_size 1, where a step is
one group, that difference *is* the training signal.

So the e9-e25 campaign never compared two versions of one training run. It
compared a baseline that barely trained against a treatment arm that did, and any
difference between them mixes the reward's shape with 300x more gradient. e28
against e27 would have reproduced it exactly.

**Raising batch_size removes the confound as a side effect.** Not because
`env_reward` gains variance across prompts - it cannot help, because TRL centres
on the per-group mean (`grpo_trainer.py:2177`, `scale_rewards` defaults to
`"group"`), so cross-group differences never reach an advantage. It works because
an optimizer step accumulates gradient over every group in the batch, and is dead
only when *all* of them are internally uniform:

```
                        steps with gradient    grad_norm median   min        max
e27bs8probe (bs=8)         40/40 = 100%           0.0398        0.0103    0.0944
e27 run1    (bs=1)         20/40 =  50%           0.0005        0.00e+00  0.4231
e27 run2    (bs=1)         19/40 =  48%           0.0007        0.00e+00  0.8264
```

The baseline moves from 300x below the cosine arms to within 2x of them.
`frac_reward_zero_std` stayed at 0.419, confirming that individual prompts
saturate exactly as often as before - a step is simply no longer one prompt.
`tools/failure_frequency` was 0.000 across all 40 steps at 64 concurrent
browsers, and clipping averaged 0.025.

Measured cost: 674 s/it for 64 completions, so 300 steps is 56h. `batch_size: 4`
leaves 0.46^4 = 4.5% dead steps, still about 95% coverage, at roughly half the
rate - near 28h for 300 steps, which looks like the better point on the curve.

40 steps cannot show whether training now *learns*; the reward curve is still
flat (0.725 / 0.648 / 0.631 / 0.713 by quarter). The probe answers the gradient
question only.

Also worth recording: e26 finqa trained on 3% of its steps. Its disqualification
stands - 1% per-rollout accuracy means nearly every group is uniformly wrong -
but the stated cause should be that the task is too hard to produce variance, not
that the model cannot learn it.

## The superseded probe writeup

`batch_size: 8`, `max_steps: 40`, everything else identical to e27 - the `diff`
is three lines. Train only: 40 steps cannot produce a good policy, so an eval
would burn 3h measuring nothing. The question is whether the gradient signal
works, and that is answered from `train_log.json`.

**The right yardstick is "steps with any gradient", not
`frac_reward_zero_std`.** That metric is the fraction of prompt-*groups* with
zero variance, so at batch_size 8 it should stay near 0.47 - roughly half of
individual prompts saturate either way. What must change is that a step now
draws on 8 groups instead of 1, so a step is dead only if all 8 saturate at
once: expected to go from ~50% of steps to ~99%.

Pass looks like: steps-with-gradient near 100%, and a reward curve that acquires
a trend instead of wandering. Fail looks like either unchanged, which would mean
the batch size was never the constraint.

Resource check before launch: 188 GB RAM with 174 free, 48 CPUs, GPU empty, so
64 concurrent playwright browsers are affordable. `max_concurrent=64` confirmed
in the launch banner.

ASSUMPTION: `gpu_memory_utilization: 0.30` leaves vLLM about 7 GB of KV cache,
which cannot hold 64 sequences at 4096 tokens simultaneously. vLLM schedules
within its budget rather than failing, so this costs step rate, not correctness.
The step rate is being watched to confirm; if it scales linearly with completions
(about 750 s/it) the run takes 8.4h, and if vLLM batches well it is far less.

## ROOT CAUSE: every optimizer step trains on a single prompt

`batch_size` is not set in any browsergym config, and `grpo_runner.py:105`
defaults it to 1:

```
total_completions = batch_size(1) * n_rollouts(8) = 8
micro = 1  ->  grad_accum = 8  ->  one optimizer step = ONE prompt-group
```

When all 8 rollouts of that one prompt share a correctness label the group has
zero advantage variance and the step contributes nothing. Measured across three
runs, that is about half of them:

```
        reward by third              frac_reward_zero_std   steps with ANY gradient
run1    0.746  0.650  0.708                0.473                158/300
run2    0.750  0.674  0.723                0.500                150/300
old     0.631  0.677  0.728                0.460                162/300
```

So each run is roughly 150 gradient updates, each derived from a single
question. That explains all three symptoms at once and needs no other cause: no
learning trend in any run, half the steps dead, and a parameter trajectory
dominated by which few prompts happened to produce signal.

**e27rep2 measured the consequence.** Identical config, identical seed 42,
identical question set - `diff` against the run that produced 0.650 is one line,
`experiment_id`:

```
held_out    click-menu-2     0.400 -> 0.280   lost 7 gained 1   p=0.070
            click-dialog-2   0.900 -> 0.840   lost 3 gained 0   p=0.250
            OVERALL          0.650 -> 0.560
shifted     OVERALL          0.820 -> 0.800
```

Run nondeterminism alone moves held_out 9pp. The question draw was never the
main cause, and one seed per arm is not defensible as things stand.

**Token noise is smaller than accuracy noise**, which matters because token
efficiency is the actual outcome:

```
correct-episode median tokens, two identical runs        n
  click-dialog-2               244 ->  278   +14%       42
  click-checkboxes-transfer    581 ->  513   -12%       45
  navigate-tree                312 ->  312     0%       34
  click-menu-2                1290 -> 1636   +27%       13  (small n)
```

About +/-13% on the well-populated families, against the -24% the e24/e25 poly
pair measured for a cosine arm. Roughly 2x noise: marginal at one seed,
workable at three - but only once training does something at all.

**Nothing should launch until batch_size is decided.** Raising it to 4-8 makes
zero-variance steps nearly impossible (every prompt in the batch would have to
saturate) and cuts gradient variance proportionally, but multiplies generation
cost per step. At a fixed ~8h budget the trade is 300 single-prompt steps
against roughly 37 eight-prompt steps, and 37 optimizer steps at LoRA LR 5e-6 is
probably too few. Doing this properly looks like 20-60h per run, which is a
change to the compute budget of the whole thesis, not a config tweak.

## ANSWERED TWICE: no bug. e27 at n=1 is not a usable baseline.

The zero point landed. All three on identical seeds under post-fix code:

```
click-menu-2 (n=50)     acc     non-term   correct-tok med   vs base
  base (e0)            0.600     17/50         2106           -
  old policy           0.660     11/50         1861           lost 0 gained 3  p=0.250
  new policy           0.480     18/50         1262           lost 6 gained 0  p=0.031

click-dialog-2 (n=50)
  base 0.840 / old 0.840 / new 0.760          base->old: 5 lost 5 gained, p=1.000

OVERALL held_out    base 0.720   old 0.750   new 0.620
```

The retrained policy is significantly worse than **not training at all** on
click-menu-2. The old run's entire benefit was termination (non-term 17/50 ->
11/50); the new run failed to learn it (17/50 -> 18/50).

The shifted split shows nothing at all - base is as good as both policies:

```
shifted (n=50 each)          base    old policy   new policy
  navigate-tree              0.660     0.620        0.640     both p>=0.625 vs base
  click-checkboxes-transfer  0.940     0.940        0.960     both p=1.000  vs base
  OVERALL                    0.800     0.780        0.800
```

Put together, 300 steps of task-success GRPO on this domain does nothing on the
untrained families and, on the trained ones, ranges from nothing (old run: 0
lost, 3 gained on click-menu-2, p=0.250) to significant harm (new run: 6 lost, 0
gained, p=0.031). The harvested e27 finding said training bought no measurable
accuracy; with the base model measured under the same code on the same seeds,
the stronger statement holds.

**The cause is not a bug.** Every training-path change in `8a01cc8` was traced:

- `max_tool_calling_iterations` reads as the obvious suspect from the commit
  message, but the old code was `if max_turns > 0: kwargs[...] = max_turns` and
  browsergym sets `max_turns: 8`. Both versions pass 8. The fix only changes
  domains that leave `max_turns` unset.
- `token_entropy` was deleted and `cosine_length` changed; e27 uses neither.
  `env_reward` is untouched.
- The rest is log dumping, smoke handling and checkpoint marking - no gradient
  path.
- The two frozen `config.yaml` files differ by one eval-only key
  (`reference_report`).

One difference remains: `seed_base=seed_block(seed)` changed which 500 questions
seed 42 trains on. Same code, same hyperparameters, different draw.

**So this is run-to-run variance, and it is large enough to invalidate the
experimental design as it stands.** One run of this config landed at +0.03
against base, another at -0.10 and significantly so. e28 and e29 at one seed each
cannot be read against an e27 that swings 13pp on question draw alone - the
noise is bigger than the token-efficiency effect they exist to measure.

Note for the design: under the old scheme seeds 42/43/44 shared 499 of 500
questions, so a seed sweep would have measured training noise only. Under the
fixed scheme a seed sweep measures question draw and training noise together,
which is the honest quantity but a larger one.

## Superseded observation, kept for the record

`e27retrain-oldseeds` landed. Paired on identical seeds, post-fix code on both
sides, so the policy is the only difference:

```
held_out                     old policy -> new policy   flips        p       non-term
  click-menu-2                  0.660  ->  0.480        lost 9 gain 0  0.004   11/50 -> 18/50
  click-dialog-2                0.840  ->  0.760        lost 7 gain 3  0.344    0/50 ->  1/50
  OVERALL                       0.750  ->  0.620

shifted
  navigate-tree                 0.620  ->  0.640        lost 0 gain 1  1.000   19/50 -> 17/50
  click-checkboxes-transfer     0.940  ->  0.960        lost 0 gain 1  1.000    3/50 ->  2/50
  OVERALL                       0.780  ->  0.800
```

Nine losses and zero gains on click-menu-2. Both untrained shifted families are
strictly non-worse - zero losses, one gain each. So the damage is confined to
click-menu-2, the harder of the two families the run trains on.

The earlier unpaired 0.660 -> 0.400 splits into both causes, policy dominating:

```
0.660  old policy, old seeds
0.480  new policy, old seeds   <- policy  -0.18  (significant)
0.400  new policy, new seeds   <- seeds   -0.08
```

The failure mode is termination, not wrong answers. click-menu-2 non-termination
went 11/50 to 18/50, with five `hit_generation_cap` where the old policy had
none, while correct-episode tokens *fell* from 1861 to 1262.

**Zero point running.** Neither policy tells us whether the retrain fell below
where training started, because e0's only report is pre-fix code on pre-fix
seeds. `e0-oldseeds` runs the base model with no adapter over the same question
set under post-fix code. Config `/workspace/e0-oldseeds.yaml`, log
`/workspace/e0_oldseeds.log`. Base near 0.66 on click-menu-2 means this retrain
actively damaged the policy; base near 0.48 means training never helped there and
the old run's 0.66 was the outlier.

Two candidates once the zero point lands, in order: the seed-block fix moved
training onto different click-menu-2 instances, or one of the other fifteen
training-path fixes changed the gradient. `checkpoint-100/200/300` are on disk,
so the regression can be bisected across training without retraining anything.

**e28/e29 stay held.**

## Original observation, kept for the record

e27 retrained cleanly - 300 steps in 7h50m, training curve indistinguishable from
the pre-fix run (both wander 0.62-0.75, both end at 0.756 over the last 50 steps,
`frac_reward_zero_std` 0.41-0.53 in both). Then the eval came back split:

```
held_out              NEW      OLD pre-fix   OLD post-fix(old seeds)
  click-menu-2       0.400       0.700           0.660      <- trained on
  click-dialog-2     0.900       0.860           0.840      <- trained on
  overall            0.650       0.780           0.750

shifted
  navigate-tree      0.680       0.620           0.620      <- never trained
  click-checkbox-tr  0.960       0.960           0.940      <- never trained
  overall            0.820       0.790           0.780
```

Better on both untrained families, worse only on click-menu-2. Wilson intervals
on click-menu-2 do not overlap: 0.400 [0.267, 0.549] against 0.660 [0.521,
0.777]. Its non-terminating episodes went 10/50 to 15/50, including four
`hit_generation_cap` where both older runs had none.

The task mix is not the explanation, checked rather than assumed:
`adapter.py:122` is `self._tasks[s % len(self._tasks)]` and both seed bases are
even, so every run above is exactly 50/50.

**The comparison is unpaired, which is the whole problem.** The seed-block fix
moved the eval to 42100000 / 42200000 while the older reports scored 100042 /
200042 - disjoint question sets, so the difference mixes the new policy with a
fresh draw. Two live candidates: the retrain regressed on click-menu-2, or the
new click-menu-2 instances are harder.

`e27retrain-oldseeds` separates them by running the retrained checkpoint over the
old question set, so the policy is the only thing that differs from
`e27paired-oldseeds`. Config at `/workspace/e27retrain-oldseeds.yaml`, log at
`/workspace/e27retrain_oldseeds.log`. click-menu-2 near 0.66 means the new seeds
are harder; near 0.40 means the retrain regressed.

**e28/e29 are held until this resolves.** They use e27 as their baseline, and
20 GPU-hours of arms against a baseline that cannot be explained is how the
e9-e21 sweep became uninterpretable.

**Also still true: the seed-block fix breaks the e0 pairing.** Training mean
reward ran 0.744 against the old run's 0.634 over the same steps - same config,
so that is seed 42 landing on different questions.
`runs/e0-browsergym-base-qwen3-1_7b/episodes_held_out.jsonl` holds seeds
100042..100141 while the retrained e27 scored 42100000..42100099, so the paired
McNemar in `e27_e1_baseline_findings.md` - what made the shifted transfer loss
significant at p=0.021 - cannot be recomputed against the new e27.

**Step-6 and step-98 health checks both passed** and are recorded here because
they rule out the cheap explanations for the drop. Gradient signal was live
(`frac_reward_zero_std` 0.47 at mean reward 0.744, against the 0.70 at 0.931 that
got attempt 1 killed for saturation). `tools/failure_frequency` was 0.000 on
every logged step, so the adapter was never talking to a stale server. Clipping
came in *below* the pre-fix run over the same 98 steps (mean 0.022 vs 0.026, max
0.375 both, longest completion 3842 against a 4096 cap), so e28's `max_len: 4096`
stands and the step-2 spike of 0.375 was noise.

One re-eval fixes both open problems, because e0's token figures also carry the
~400/episode multi-call inflation that the "+247 median tokens" result rests on.
Preserve the existing directory first, the same way e27's was, since
`--base-model` writes into `runs/<experiment_id>/` regardless:

```
mv runs/e0-browsergym-base-qwen3-1_7b runs/e0-PREFIXCODE-browsergym-base-qwen3-1_7b
cd /workspace/master-thesis-code/pipeline && setsid nohup ../.venv/bin/python -m eval.runner \
  --config configs/e0-browsergym-base-qwen3-1_7b.yaml --base-model \
  > /workspace/e0_reeval.log 2>&1 < /dev/null &
```

Log at `/workspace/e27_retrain.log`. The pre-fix-code run is preserved at
`runs/e27-PREFIXCODE-browsergym-e1-baseline-qwen3-1_7b/` rather than overwritten,
because it holds `eval_report_pre_fix.json` and the episode files the paired
probe was measured against, plus its own checkpoints. The retrain therefore
writes into a clean `runs/e27-browsergym-e1-baseline-qwen3-1_7b/` and the
overwrite guard stays armed.

## The eval dispatched one tool call per turn. TRL dispatches all of them.

Found 2026-08-04 by the paired probe, which is the only reason it was found: the
unpaired re-eval showed the same drop and blamed it on the sample.

The first paired run (kept at `runs/e27paired-oldseeds-MULTICALLBUG-qwen3-1_7b/`)
came back held_out 0.780 -> 0.750 and shifted 0.790 -> **0.600**, with **19
losses and 0 gains** on shifted. One-directional, so not noise.

Replaying the worst seed (200051, `python /workspace/debug_turn.py 200051 6`)
showed what happens. The model emits four clicks in one turn - bb, Gppj, Cxrb,
Submit, exactly the task. The loop dispatched the first and appended the parsed
message advertising all four. The next turn's prompt therefore held four
`<tool_call>` blocks answered by a single `<tool_response>`, and the model read
the three unanswered ones as having succeeded: *"The task is complete. The
checkboxes have been selected, and the Submit button has been clicked."* No tool
call, episode over, scored wrong.

Both eval behaviours were wrong, in different ways:

- **Before the review fixes**, the loop appended a stub rebuilt from the one
  dispatched call. Self-consistent, so the model just re-requested the dropped
  clicks over later turns and got there eventually - at 4 steps and 2848 tokens.
  Divergent from training, and it inflated tokens and steps on every multi-click
  task.
- **After them**, appending the parsed message while still dispatching one call
  made the transcript inconsistent, which is what cost the 19 episodes.

`trl/trainer/grpo_trainer.py` ("Call the tools, and build the new prompt")
appends the full assistant message and then loops over `tool_call_list`,
executing every call. Training has always done this. So the fix is to dispatch
all of them and append one tool response per call, which is now what
`_run_multiturn_episodes` does (`_first_tool_call` became `_tool_calls`).

Verified on the same seed: four calls dispatched, env done, reward 1.0, **915
tokens in one turn** against 2848 in four. 253 tests on the box, 240 locally,
including four new ones in `test_multiturn_eval.py` that pin the multi-call
transcript shape.

**Consequences for numbers already written up.** Training is unaffected - TRL
did this correctly all along, so e27's policy is fine. Every eval number ever
produced by this pipeline was measured one-call-per-turn, which on multi-click
families overstates tokens and steps by roughly 3x. e0-vs-e27 in
`e27_e1_baseline_findings.md` is still internally valid, since both arms were
measured the same way, but its absolute token figures are not comparable to
anything measured after today and the "+247 median tokens" result there needs
re-deriving before it is cited.

## The paired probe: the fixes are clean, and the eval got harder on purpose

`e27paired-oldseeds` finished 12:17 UTC, 200 episodes, post-fix code over the
pre-fix question set. Diff with `/workspace/paired_diff.py`.

```
                        held_out            shifted
accuracy         0.780 -> 0.750      0.790 -> 0.780
flips            3 lost / 0 gained   2 lost / 1 gained
both-correct     28 shorter/5 longer 34 shorter/0 longer
                 mean -319.4 tok     mean -271.2 tok

pooled n=200  McNemar lost 5 gained 1  p=0.219
token sign test  held_out p=6.6e-05   shifted p=1.2e-10
all episodes  mean -396.1 tokens
```

Accuracy does not move detectably. Token count does, hard, in one direction:
shifted has zero episodes that got *longer* out of 77 both-correct pairs. A
further 15 held_out and 10 shifted episodes were relabelled without changing
correctness, e.g. `no_tool_call/2st/3746tok -> max_turns/8st/2864tok` - same
failure, honest stop reason, 900 fewer phantom tokens. Mean steps rise (1.78 ->
2.33) because a step is now a dispatched call, so a four-click turn counts four.

**All five losses were replayed under both loop modes** (`debug_turn.py <seed>
10 [--first-only]`, where `--first-only` reproduces the pre-fix dispatch). Two
distinct mechanisms, both of them the old loop handing the model information
that training never gives it:

- **The pre-fix stub dropped `reasoning_content`.** Seed 100106, turn 0 is
  byte-identical in both modes. Turn 1's prompt is 1732 tokens post-fix against
  548 pre-fix, and the gap is the model's own 1204-token think block. Rebuilding
  a stub from `{name, arguments}` silently discarded it. With the think block in
  context the model clicks bid 44 and scores 0; without it, bid 41 and scores 1.
- **The pre-fix loop bought extra observations with the dropped calls.** Seed
  200063, the model asks for clicks 21/24/27/32 in one turn. Post-fix dispatches
  all four blind, the 4th bid is wrong, and the next turn declares the task
  complete. Pre-fix dispatched only 21, so the model saw three more page states
  and revised its way to 21/27/24/36, which is right.

**TRL concatenates, it does not re-render**, which settles which behaviour is
faithful: `grpo_trainer.py:1578` is commented "Build token IDs by concatenation:
prompt + completion + tool_suffix", and line 1590 does
`prompt_ids + completion_ids + suffix_ids`. The raw generated tokens carry the
think block forward verbatim, and every call in `tool_call_list` is dispatched.
The post-fix eval re-renders through `apply_chat_template` instead, but arrives
at the same content - 401 prompt + 1204 completion + ~127 tool and framing = the
1732 observed. So both flip mechanisms are the eval ceasing to be easier than
training. Pre-fix accuracy was optimistic by about 2pp (ns) and long by about 400
tokens.

## Queued

All 16 review fixes are applied and verified (`pipeline/FIX_PLAN.md`), and the
paired probe above clears them. Both the eval loop and the training path changed,
so nothing measured before today is comparable to anything measured after. Two
GPU jobs, in order:

1. **Running now:** retrain e27 + eval (~14h). Required because the seed-block fix
   changes which questions seed 42 trains on, and e28/e29 must share e27's scheme.
2. Re-eval e0 on the new seed scheme (~3.3h), command above. Blocks the
   e0-vs-e27 pairing and the "+247 median tokens" re-derivation, both of which
   `e27_e1_baseline_findings.md` currently rests on.
3. Launch e28/e29 (~20h).

`runs/e27-.../eval_report_pre_fix.json` is the pre-fix reference (100 episodes
per split, held_out 0.780, shifted 0.790). Do not delete it until step 1's diff
is written up. Its `.md` was removed on purpose - a re-eval rewrites it, and the
4-episode smoke numbers that briefly sat there were worse than nothing.

`--smoke` now refuses to write into a run dir holding a real report, so the
accident that produced that state cannot recur.

## e28/e29 stopped at step 83/300 - GPU handed to a colleague, not a failure

Stopped 10:40 UTC Aug 3 after 2h20m, by decision, to free the box for someone
else. The run was healthy: the cosine fix had already been confirmed live
(below), and there was no fault of any kind. Killed in order - batch parent
783675 first so it could not launch e29 as the trainer exited, then the watcher
786026, then train 783676; the browsergym env server 783870 orphaned rather than
following its parent and needed a separate SIGTERM. Verify with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` returning
empty, not just with `ps`. The MiniWoB static server on :8080 (pid 217225) was
left up on purpose - it is shared infrastructure and does not survive a restart.

**Nothing to harvest, and nothing to resume from.** The run directory holds
`config.yaml` and `batch_train.log` only. The per-step reward curves survive as
text in that log; `train_log.json` does not, because `_save_train_log` runs after
`trainer.train()` returns and a kill never reaches it.

Relaunch from scratch, both arms, no code change needed - the batch skip
predicate wants `checkpoint-final` to skip training, so the partial directory
will not be mistaken for a finished run:

```
cd /workspace/master-thesis-code/pipeline && setsid nohup ../.venv/bin/python -m training.batch \
  configs/e28-browsergym-e2-cosine-qwen3-1_7b.yaml \
  configs/e29-browsergym-e3-nontermination-qwen3-1_7b.yaml \
  --train --eval --force > /workspace/e28_e29_batch.log 2>&1 < /dev/null &
```

The venv is at the repo root, so it is `../.venv/bin/python` from `pipeline/`;
an earlier version of this command said `.venv/bin/python` and would have died on
the spot. `setsid` and `< /dev/null` are what let the job outlive the ssh drops.

Budget about 8.7h for e28 train at the observed 105 s/it, then its eval, then e29
in full - roughly 20h for the pair. Check :8080 and :8000 first, per the launch
checklist below.

**Checkpoints do get written every 100 steps, and the runner's docstring says
otherwise.** `grpo_runner.py:186` claims "save_strategy is off (we save only the
final LoRA)". It is not: `GRPOConfig.save_strategy` defaults to `STEPS`, the
runner passes `save_steps=100`, and e27's directory carries `checkpoint-100`,
`-200`, `-300` and `-final` at 681M total. So this kill at step 83 missed the
first checkpoint by 17 steps, about 30 minutes. Worth knowing for the next
handover: stopping just past a hundred-step boundary costs far less than stopping
just before one. It would still not be resumable as things stand -
`resume_from_checkpoint` appears nowhere in the pipeline and `runner.train()`
calls `trainer.train()` with no resume argument - but `save_only_model` is False,
so the optimizer state is on disk and wiring it is small if it ever matters.

**The fix was verified on the box before this launch**, because it had never run
there:

```
before: model_token_count [18, 1024]   vs completion_ids [1024, 1024]   ratio 0.018
after : model_token_count [880, 1024]  vs completion_ids [1024, 1024]   ratio 0.86 / 1.00
```

0.86 is correct rather than residual error - `completion_ids` includes the
injected 332-character page and the chat framing, which the reward deliberately
excludes. 880 assistant tokens plus about 144 of tool and framing is 1024.

**Early check at step 12, 09:00 UTC: PASSED.** The signature was
`reward/CosineLengthReward/raw_std` on steps where `reward/EnvReward/raw_std` is
0 - under the bug it read 0.0002 there, because the cosine's only variance came
from the correctness gate.

```
uniform-correctness steps (env_std = 0): 5 of 12
cosine raw_std on those: min 0.0647  median 0.0944  max 0.1196
```

About 470x the buggy value, and the sign structure is right as well: step 5 is
all-wrong at mean_len 2336 with cos_mean -0.71, step 8 all-correct at mean_len
1909 with +0.78. Correct-and-short rewarded, wrong-and-long penalised least -
the Wu/Yeo cell structure, which a reward reading 8% of the length could not
produce. Mean lengths run 1900-2700 with within-group spreads of 1200-3000, so
there is real length variance for it to bite on.

The ssh outage ran 07:55-08:30 UTC and recurred briefly around 09:20, both times
self-healing. Neither touched the run: uptime reads 81 days, tqdm is continuous
across both windows, and `/workspace/completion_dump.json` from 07:56 survived.

## e28 first attempt, KILLED at step 8/300 - the cosine was measuring the wrong quantity

The earlier kill the same morning, not the handover above. Killed 07:47 UTC (was
batch 769998, train 769999). Not inert - wrong.

Decoding the logged cosine values against TRL's own length counter:

| step | mean_len | cos_mean | cos_std | env_mean | implied n_tokens |
|---|---|---|---|---|---|
| 4 | 508 | +0.9999 | 3.4e-05 | 1.0 all correct | ~37 |
| 3 | 2728 | -0.9996 | 2.0e-04 | 0 all wrong | ~73 |
| 6 | 2202 | -0.9998 | 1.6e-04 | 0 | ~52 |

`model_token_count` is counting 1-3% of the real completion. Every bit of the
cosine's variance comes from the correct/wrong split: when all eight rollouts
share a correctness label its std is about 0.0002, so it carries no length
information at all. Steps 1 and 2, which had mixed correctness, show std 0.66
and 0.97 - that is the correctness gate talking, not length.

The step-30 check caught this at step 8 only because it was armed to print
`contrib_l1` per component. Without it the run would have finished in 12 hours
looking like a clean null.

**Root cause, dumped rather than inferred.** The real completion object TRL
hands a reward function:

```
roles          : ['assistant', 'tool', 'assistant']
msg keys       : [['content', 'reasoning_content', 'role', 'tool_calls'],
                  ['content', 'name', 'role'],
                  ['content', 'role']]
msg 0: content 0 chars, tool_calls True     <- think block is in reasoning_content
msg 2: content '<think>\nOkay, so the user...'  <- final tool-less msg keeps it inline
model_token_count: [18, 1024]    completion_ids: [1024, 1024]
```

transformers' chat parser splits a Qwen3 response so the think block lands in
**`reasoning_content`** on any assistant message carrying a tool call.
`model_token_count` read `content` and `tool_calls` only, so it counted 18
tokens of a 1024-token completion. For Qwen3 the think block is the generation;
everything else is a ten-token tool call.

Two paths were consistent with the symptom and reasoning from the source picked
the wrong one - TRL guards `parse_response` on `tokenizer.response_schema`, and a
freshly loaded `Qwen/Qwen3-1.7B` reports that as None, which predicts the plain
`batch_decode` fallback and correct counting. The dump settled it in one run.
Third time in this project that reasoning about an interface beat looking at it,
and the third time it was wrong.

**Fixed** in `training/rewards/utils.py`: `reasoning_content` is now summed
alongside `content`, with a regression test built from the dumped shape rather
than a synthetic one. The old tests passed throughout - they exercised the
function against messages this pipeline never produces.

**e9-e25 measured the same thing - checked, confirmed.** The poly cosine
campaign ran the same model through the same multi-turn path (reasoning_gym's
`answer` tool produces an assistant + tool pair, so `_is_multiturn` is true and
this counter was used). Decoding e25's `train_log.json` over its 166 non-capped
uniform-correctness steps: actual median length 859, implied 71, ratio 0.083,
pearson r +0.671 between implied and actual, cosine `raw_std` median 0.00017.
The whole campaign is void. Write-up in
`pipeline/runs/cosine_token_count_bug_findings.md`; never cite that null.

**e29 is unaffected** - `NonTerminationPenalty` reads `env.done` and never
touches token counts. It was killed only because it shared the batch parent, and
can relaunch as soon as the GPU is free.

**ssh proxy dropped 05:30-07:25 UTC**, `Connection refused` on the forwarded port
while the host answered ICMP at 0% loss. Same devpod failure as 12:54-14:58 on
Aug 2. e0 ran through it untouched and finished at 07:25. Do not relaunch on a
refused connection.

## e0 landed: the E1 baseline is weaker than it looked

Paired on identical seeds, McNemar exact:

```
held_out accuracy  0.750 -> 0.780   lost 4  gained 7   p=0.549
shifted  accuracy  0.870 -> 0.790   lost 9  gained 1   p=0.021
click-menu-2 tokens, 33 both-correct seeds: +247 median, 23 longer / 10 shorter, p=0.035
```

Three hundred steps of task-success-only GRPO bought no measurable accuracy
where it trained and cost a significant amount where it did not, while inflating
length on the one family that has length to inflate. That last part is headroom
for E2 rather than a problem.

The probe-based version overstated the transfer loss: navigate-tree probed 0.85
at n=20 but pairs against 0.74 here, so the drop is 0.12, not 0.23. Second time
a 20-episode browsergym estimate has moved by more than a tenth.

Full analysis, including why the pooled per-split token median is an artifact of
the bimodal mix and must not be reported, is in
`pipeline/runs/e27_e1_baseline_findings.md`.

## Before the E2 / E3 arms - two knobs checked against e27's real numbers

**E3 is live.** `NonTerminationPenalty` reads `env.done`, and
`BrowserGymEnvAdapter` sets it (`__init__`, `_act`, `reset`). The component's
docstring lists reasoning_gym / finqa / repl / textarena terminal tools and not
browsergym, so this was worth confirming rather than assuming: browsergym has no
terminal tool, the env reports `done` when the page task completes after a
`click`. E3 runs under `naive_sum`, matching e27.

**E2's `max_len` should be 4096.** Completion lengths over all 300 steps:

```
per-step mean : min 351  p25 557  med 1518  p75 2254  max 3091
global        : min 242                                max 3969
within-group spread: p25 500  med 1066  p75 1572  max 2834
steps whose longest rollout exceeds 1024 / 2048 / 3072 / 4096: 216 / 160 / 77 / 0
```

The e25 rule is that `max_len` must not sit far above where completions live,
because the cosine is flat near both endpoints and the whole distribution would
land in the dead zone. Here it does not: the observed maximum is 3969, 97% of
4096, so the distribution fills the range and the median lands at progress 0.37,
on the sloped part of the curve. 3072 would centre the median better but sends
the longest rollouts of 77 of 300 steps to the flat endpoint, where the reward
cannot tell 3072 from 3969 - and those are exactly the rollouts it exists to
push down.

The within-group spread is what the reward actually has to bite on, since GRPO
compares rollouts inside one prompt-group: median 1066 tokens, p75 1572. There
is real length variance to shape at fixed correctness.

**The cosine's weight is a structurally weak lever under naive_sum.** Raising
`w` scales the cosine's length term and its own correctness gating equally; only
env_reward's fixed 1.0 stays put. Computed numerically against the real
`CosineLengthReward` on e27's own within-group spread (985 to 2051 tokens,
max_len 4096), length's share of the total advantage range reads:

```
w      1      2      4      8     16     -> asymptote
share  0.063  0.077  0.086  0.091  0.094    0.097
```

A sixteenfold weight change buys a factor of 1.5, and no weight puts length past
a tenth of the range. This does *not* explain the e9-e21 flat dose-response -
that was the token-counting bug, and a weak lever and a blind reward are
different failures. It is only the reason e28 runs one weight (1.0) rather than
a sweep: a null there would not be answered by a bigger weight, it would be
answered by the cosine's endpoint spread, which is Wu/Yeo's parameterisation and
not a knob to quietly retune.

**Both conditions act on the click-menu-2 half only.** click-dialog-2 terminates
in 50 of 50 eval episodes at 1.02 steps and 313 median tokens: no
non-termination for E3 to remove, and a cosine spread of about 0.01 against
env_reward differences of 1.0. It is gradient insurance, not a treatment family.
E2 and E3 results have to be read against the click-menu-2 half. Detail in
`pipeline/runs/e27_e1_baseline_findings.md`.

## Before any browsergym launch

1. **MiniWoB static server must be up on :8080.** It does not survive a box
   restart. `cd /workspace/miniwob-plusplus/miniwob/html && python3 -m http.server 8080`
   (currently pid 217225, up since Aug 01). Check with
   `curl -o /dev/null -w "%{http_code}" http://localhost:8080/miniwob/click-option.html`.
2. **Port 8000 must be free.** `pgrep -af server.app` and kill by pid.
   `EnvServerProcess.start()` now raises instead of running on someone else's
   server, so this is loud rather than silent, but it still blocks the launch.

## Box state - read before the next run

- **browsergym is installed in the pipeline venv** (`browsergym-core`,
  `browsergym-miniwob`, `playwright==1.44.0`), not only in `/workspace/bgym-venv`.
  `EnvServerProcess` runs the env server with `sys.executable`, so the server
  subprocess needs it in the same venv as training. The only version move was
  `greenlet 3.5.2 -> 3.0.3` for playwright's pin.
- **The browsergym server takes its port from `BROWSERGYM_PORT`, not the `--port`
  argv** `EnvServerProcess` passes. It always binds 8000 by default, which is what
  the configs ask for. A different `training.env_server.port` would fail loud
  (nothing answers the client), not silent.
- **OpenEnv clone updated** `d372fab` -> `024eedc`. Rollback point is `d372fab`.
  The finqa patches survived the pull and are saved in `openenv-patches/`.
- **`openenv-core` must stay uninstalled.** It ships the same `openenv` import
  name as the repo (`0.4.2.dev0`) and shadows it; the skew fails at the first
  client `reset()` with a `metadata=` TypeError, not at import. `setup.sh` handles
  this now. Verify envs with a real HTTP reset via `pipeline/probes/env_check.py`,
  never with an import.
- **`ss` is not installed on the box.** `ss ... | grep` prints nothing whether or
  not the port is held, which reads as "port free". Use `pgrep -af server.app`.
- **Read eval progress from `runs/<exp>/episodes_<split>.jsonl`, not from the
  log.** Episodes are written and flushed one per line, but Python block-buffers
  stdout to a file, so the log sits frozen at "loading model" for the whole run
  and reads like a hang.
- **Never write a watcher as `while pgrep -f '<pattern>'; do sleep N; done`.** The
  watcher's own command line contains the pattern, so it matches itself and spins
  forever after the job it watches exits. Three have been killed this way (273631,
  297350, 908606). Poll the output file instead, or match on the pid.

## Harvested

**e27paired-oldseeds** - done, 200 episodes, harvested into the section above.
Diagnostic, not an arm: config at `/workspace/e27paired-oldseeds.yaml`, kept
outside the repo on purpose. `seed_block(0) == 0`, so `seed: 0` with
`seed_offset: 100042 / 200042` reproduces the pre-fix seed bases exactly. The
run with the multi-call bug is preserved at
`runs/e27paired-oldseeds-MULTICALLBUG-qwen3-1_7b/`.

**The unpaired post-fix re-eval in `runs/e27-.../eval_report.json` must not be
quoted as a measurement of the fixes.** The seed-block fix moved eval seeds from
`seed + offset` to `seed_block(seed) + offset`, so it scored 42100000..42100099
and 42200000..42200099 against the pre-fix run's 100042..100141 and
200042..200141. Zero overlap, so its accuracy change mixes the code change with a
fresh draw of 100 MiniWoB instances - which is exactly how the multi-call bug
nearly got dismissed as sample noise. `eval_report_pre_fix.json` is the pre-fix
reference (held_out 0.780, shifted 0.790); keep it.

**e0 browsergym base model** - done, 200 eval episodes, no adapter. held_out
0.750 / 0.860 termination, shifted 0.870 / 0.900. Paired against e27 in
`pipeline/runs/e27_e1_baseline_findings.md`.

**e27 browsergym E1 baseline** - done, 300 steps + 200 eval episodes. held_out
0.780 success / 0.900 termination, shifted 0.790 / 0.850, one generation-cap hit
in 200 episodes. Findings in `pipeline/runs/e27_e1_baseline_findings.md`.

**BrowserGym difficulty probes e27probe1-4** - all done, fifteen families
measured under the real adapter. Findings in
`pipeline/runs/browsergym_difficulty_correction.md`, which supersedes the
difficulty numbers in `browsergym_feasibility_findings.md`.

**e26 finqa E1 qualification** - done. finqa disqualified (0/60 eval, 1%
per-rollout training accuracy, gradient on 10 of 300 steps). Findings in
`pipeline/runs/e26_finqa_qualification_findings.md`.

**e24/e25 4k pair** - done. Reports and train logs in
`pipeline/runs/e24-poly-4k-env-only-qwen3-1_7b/` and
`pipeline/runs/e25-poly-4k-cosine-w16-qwen3-1_7b/`; findings in
`pipeline/runs/e24_e25_4k_pair_findings.md`.

## Killed

**e27 first attempt, step 21/300 (was pid 246083).** `frac_reward_zero_std` 0.70
with mean reward 0.931 - saturated, no gradient, and success and termination had
collapsed onto one axis. Cause was difficulty measured under the first probe's
five tools and prompt rather than the adapter's two. Nothing to harvest. The
relaunched run on corrected numbers is the harvested e27 above.

## Traps that have each cost real time

- **A leftover env server on the shared port silently serves the next run.** Every
  env's `server/app.py` binds one fixed port, so the new server dies on bind while
  the readiness probe passes against the old one. The first e27 smoke trained to
  completion against a day-old reasoning_gym server: every browsergym action came
  back `VALIDATION_ERROR`, giving `tools/failure_frequency` 1.0 and reward 0 with
  no traceback anywhere. `EnvServerProcess.start()` now refuses an occupied port.
- **Difficulty measured under conditions the run does not use has now cost three
  measurements**: the five-tool probe surface, the 2048 budget read per-turn when
  TRL applies it per-trajectory, and greedy decoding when training samples at
  temperature 1.0. Any future selection probe runs `do_sample: true` at the
  training temperature, through the real adapter, at the real budget.
- **`--smoke` overrides the budget geometry** to `max_seq_length=2048`,
  `max_prompt_length=1024`. A smoke's clipped_ratio does not transfer.
- **`pkill -f <pattern>` over ssh matches the remote shell running it** when the
  pattern appears in that command line, killing the session before the work runs.
  Kill by pid. The same aliasing makes `pgrep -af <pattern>` report your own shell
  as a hit - a "port busy" that is only your check.
- **`Connection refused` on ssh does not mean the box died, and it is never a
  local networking problem.** Three-command proof, no sudo:

  ```
  route -n get 130.149.248.103     # expect: interface utun8, gateway 130.149.212.110
  nc -z -v -G 4 -w 4 130.149.248.103 22       # expect: OPEN
  nc -z -v -G 4 -w 4 130.149.248.103 30236    # the devpod port
  ```

  A refused connection arrives as a TCP RST from the host in about 130 ms, which
  means the SYN crossed the tunnel, reached `siena04.cit.tu-berlin.de`, and its
  kernel refused because nothing is bound to that port. A NordVPN killswitch or
  a stale pf rule produces a *timeout* or "no route to host" instead, never a
  fast RST from the destination. Port 22 answering while 30236 refuses localises
  the fault precisely: the machine is up, the devpod workspace's sshd is not.
  Controls worth keeping in the same check: `www.tu-berlin.de:443` (tunnel) and
  `1.1.1.1:443` (general internet).

  Gateway `130.149.212.110` is TU Berlin, so `utun8` is the TUB VPN even when
  NordVPN's Shield extension is loaded and holds its own default route. Seeing
  NordVPN in `pgrep` is not evidence it is the cause - check the route.

  **The pod does not die, and neither do its processes.** Measured 2026-08-03,
  not inferred: the container's PID 1 (`sshd -D -e`) started Thu Jun 4 13:31:54
  and has 60 days of elapsed time, so it has not restarted through any outage,
  and the e27 re-eval ran to completion at 22:31 UTC while the connection was
  down. Second confirmed survival (2026-08-02 was the first). Still check
  `pgrep` on reconnect rather than trusting this, and never relaunch on a
  refused connection - you will be starting a second copy of a live job.

  **Do not wait for someone to restart the workspace.** There is nothing to
  restart. The previous version of this note said the pod was down and the
  DevPod workspace needed a manual restart; that was wrong, and it cost a run's
  worth of waiting.

  **Port 22 is not a recovery path.** It is the host's own sshd and it rejects
  `~/.ssh/tub` as user `dev` with `Permission denied (publickey)`. The key is
  scoped to the workspace, not to `siena04`. So kubelet and kube-proxy logs are
  out of reach and the fault cannot be confirmed from the cluster side; no devpod
  CLI is installed locally and kubectl has no context. Wait it out, or use the
  exalsius console.

  **Best hypothesis, not yet confirmed.** `hostname` inside the workspace has
  the `<name>-<replicaset>-<pod>` shape of a k8s Deployment pod, and it sits
  behind NodePort 30236. A fast RST on the
  NodePort while the pod runs is the signature of kube-proxy's REJECT rule,
  which it installs when the Service has no ready endpoints - a failing
  readiness probe or a node briefly NotReady would both produce it, with the pod
  untouched. The competing explanation is a TUB firewall rejecting high ports
  for some VPN client addresses. These are distinguished by whether the client
  VPN address changes across the outage: on 2026-08-03 it did not
  (130.149.214.224 both at 19:46 before and 23:33 after), which is evidence
  against the client-side story, but `last` only records successful logins so it
  cannot see the middle of the window.

  Both ends are now instrumented so the next outage is measured rather than
  argued about. `/workspace/netwatch_inside.log` in the pod records a per-minute
  heartbeat plus egress; the outside probe logs port 22, port 30236, two
  controls and the client VPN address every minute. Pair them: no gap inside
  while outside reads `30236 refused` localises the fault to the ingress path,
  and a VPN address change at the boundary localises it to the client.
- **`eval.runner --base-model` writes into `runs/<experiment_id>/`** regardless of
  whether an adapter was loaded (`runner.py:36,51`), so pointing it at a trained
  run's config overwrites that run's `eval_report.json` and episode files. A
  base-model eval needs its own config with its own `experiment_id`.
