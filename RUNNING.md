# Running

What is executing on the GPU box (`ssh gpu-l4`). Rows leave the table once the
results are harvested. Update rules are in CLAUDE.md.

Updated: 2026-08-04 16:49 UTC (box time)

| run | phase | pid | started | ETA |
|---|---|---|---|---|
| e27 retrain | train 300 steps, then eval 200 eps | 1008154 (batch) / 1008156 (train) | 14:07 UTC Aug 4 | ~04:00 UTC Aug 5 |

Rate settled at 115-131 s/it after a fast first step, so train ends about 00:45
UTC and the 200-episode eval adds roughly 3h.

**Step-6 health check passed.** `frac_reward_zero_std` averages 0.50 at mean
reward 0.708, against the 0.70 at mean reward 0.931 that got attempt 1 killed for
saturation, so half the steps carry gradient. `tools/failure_frequency` is 0.00
throughout, which is the check that the adapter is not talking to a stale server.
`tools/call_frequency` runs 1.0 to 4.25, so multi-call turns are as real in
training as the eval fix now assumes.

**Clipping recheck at step 98: closed, e28's `max_len: 4096` stands.** The step-2
`clipped_ratio` of 0.375 was noise. Over the same first 98 logged steps:

```
                 clipped_ratio             mean_len  max_len  reward  frac_zero_std
NEW  post-fix    mean 0.022  max 0.375  12/98   1448     3842    0.744    0.47
OLD  pre-fix     mean 0.026  max 0.375  15/98   1524     3951    0.634    0.48
```

Marginally less clipping than before, and the longest completion is still under
the cap. `tools/failure_frequency` is 0.000 on all 98 steps.

**The seed-block fix breaks the e0 pairing, and e0 must be re-evaluated.** Mean
reward runs 0.744 against the old run's 0.634 over the same steps - same config,
so that is seed 42 landing on different training questions. The same shift moves
the eval: `runs/e0-browsergym-base-qwen3-1_7b/episodes_held_out.jsonl` holds seeds
100042..100141, while the retrained e27 will score `seed_block(42) + 100000` =
42100000..42100099. Zero overlap, so the paired McNemar in
`e27_e1_baseline_findings.md` - which is what made the shifted transfer loss
significant at p=0.021 - cannot be recomputed against the new e27.

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
