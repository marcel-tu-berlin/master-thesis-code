# Running

What is executing on the GPU box (`ssh gpu-l4`). Rows leave the table once the
results are harvested. Update rules are in CLAUDE.md.

Updated: 2026-08-02 17:25 UTC (box time)

| Run | Phase | pid | Started | ETA | Notes |
|---|---|---|---|---|---|
| e27 browsergym E1 baseline | train, 300 steps | 343578 | 17:25 UTC | ~12.5h train, then eval 200 eps | Mix click-menu-2 + click-dialog-2. Env server 343758. Log `/workspace/e27.log`. 155 s/step. **On probation - decide at step 30, see below.** |

### e27 probation: greedy selection vs sampled training

First six steps, one prompt-group each (batch_size 1, so a step is one family):

| step | mean_length | tool calls | reward | frac_reward_zero_std |
|---|---|---|---|---|
| 1 | 2342 | 3.1 | 0.125 | 0 |
| 2 | 1940 | 2.5 | 0.375 | 0 |
| 3 | 2656 | 3.8 | 0 | 1 |
| 4 | 479 | 1.1 | 0.875 | 0 |
| 5 | 2475 | 3.3 | 0 | 1 |
| 6 | 2537 | 3.9 | 0 | 1 |

click-menu-2 (the ~2500-token steps) is running near 0.10 against the 0.60 the
probe measured. `tools/failure_frequency` is 0 and the lengths match the probe,
so this is neither a broken tool nor truncation - it is that **every selection
probe measured greedy accuracy while training samples at temperature 1.0**
(`grpo_runner.py:93`, unset in the config so it defaults to 1.0; the probes ran
`eval.temperature: 0.0`). GRPO's within-group variance depends on the sampled
rate, not the greedy one, and for click-menu-2 the two differ by about 6x.

This is the same error as the five-tool probe and the 2048 budget, in a third
place: difficulty measured under conditions the run does not use. Any future
selection probe has to run `do_sample: true` at the training temperature.

`frac_reward_zero_std` is 0.50 so far, and the zero-variance groups are all-ZERO
(too hard) rather than the all-one groups that killed the first attempt at 0.70.
Opposite failure mode, and the more recoverable one - half the steps still carry
gradient, against finqa's 10 of 300.

Not killed on six steps, because that is how the phantom e13 signal happened.

### Step-30 verdict: continue

```
zero_std  mean : 0.516   first/second half: [0.533, 0.500]
reward    mean : 0.560   first/second half: [0.375, 0.734]
clipped   mean : 0.036
length    mean : 1616
last 11 steps  : 1.0 1.0 0.75 1.0 1.0 0.5 0.875 0.5 0.625 1.0 1.0  (mean 0.84)
```

The policy is learning: reward 0.375 -> 0.734. Completion length fell 2500 ->
1616 and step time 155 -> 83 s/step **while reward rose**, so it is solving
faster rather than bailing out - premature stopping would have moved reward the
other way. Clipping at 0.036 confirms the 4096 budget is sized right, which
retires the tail worry from the launch note.

`frac_reward_zero_std` stays near 0.50, but its cause has inverted: the early
zero-variance groups were all-ZERO (too hard for click-menu-2 under sampling),
and the recent ones are all-ONE - six of the last eleven steps scored a clean
1.0. The run is heading toward saturation, the same end state that killed the
first attempt, reached by learning instead of by starting there.

That is not a reason to kill it. A strong E1 is a good baseline: E2 and E3 then
get compared on efficiency at matched task success, and substitution shows up as
a drop from a high ceiling rather than noise around a middling one. It does mean
the later steps may carry little gradient. **Step-100 checkpoint armed**
(`/workspace/e27_checkpoint100.txt`) to decide whether to stop at a saved
checkpoint - `save_steps` is 100, so 100 and 200 are both resumable - rather
than burn 200 near-gradient-free steps.

## e27probe4 - done, all seven splits

| family | acc | term | gap | median assistant tokens | steps |
|---|---|---|---|---|---|
| click-checkboxes-transfer | 1.00 | 1.00 | 0.00 | 687 | 1.9 |
| click-dialog-2 | 0.80 | 1.00 | 0.20 | 385 | 1.0 |
| click-menu-2 | 0.60 | 0.70 | 0.10 | 2569 | 2.6 |
| grid-coordinate | 0.10 | 0.50 | 0.40 | 5904 | 3.4 |
| click-tab-2-hard | 0.10 | 0.20 | 0.10 | 6550 | 5.7 |
| click-collapsible-2 | 0.05 | 0.30 | 0.25 | 5638 | 4.8 |
| click-pie | 0.00 | 0.00 | 0.00 | 8929 | 6.3 |

Fifteen families have now been measured under the real adapter. The boundary is
sharp and it is about episode length, not task family: at or under 2.6 mean
steps everything is solvable, at 3.4 and above everything is dead, and the dead
ones fail the same way - 5, 7, 10, 13 of 20 episodes looping until `max_turns`.
Difficulty and trajectory length are one variable for this model, which is why
every earlier probe found the interesting families untrainable.

Observation cost was measured rather than estimated, through the real adapter:
**45 tokens for a click-menu-2 page, 101 for click-dialog-2**, against the 500
that a 2000-char cap implies. That is what makes click-menu-2 trainable - its
real worst-case trajectory is about 4124 tokens, not the 5107 the cap-based
estimate gave.

**ssh was unavailable roughly 12:54-14:58 UTC and nothing was wrong.** The box
never rebooted (`uptime` reads 81 days) - only the devpod's ssh proxy dropped,
which presents as `Connection refused` on the forwarded port while the host
still answers ICMP. Every process kept running through it: e27probe3 finished
all four splits at 13:01:45 and the chain script launched e27probe4 ten seconds
later. Do not treat a refused connection here as a dead box; check `uptime` and
`pgrep` before concluding anything died, and never relaunch on the assumption
that it did.

## e27probe3 - done, all four splits

| family | acc | term | gap | median assistant tokens | steps |
|---|---|---|---|---|---|
| click-checkboxes-large | 0.60 | 0.75 | 0.15 | 6066 | 5.9 |
| click-checkboxes-soft | 0.85 | 0.90 | 0.05 | 3390 | 3.6 |
| navigate-tree | 0.85 | 0.85 | 0.00 | 331 | 1.6 |
| click-tab-2 | 0.05 | 0.35 | 0.30 | 5327 | 4.0 |

click-tab-2 came in at 0.05, far below the band, with 5 of 20 looping to
`max_turns` and a 5327-token median - disqualified on difficulty and on length.
Raising the budget did not rescue it, which is the counter-example that keeps
the click-checkboxes-large result from being "a bigger budget helps everything".

## e27 is blocked on a hardware ceiling, not a config choice

e27probe3 found a family that clears both selection bars and then found that it
cannot be trained on this box. Both halves matter.

**click-checkboxes-large clears the bars once the budget stops lying.** Paired
seeds, 2048 -> 4096 per-turn eval budget: success 0.45 -> 0.60, termination
0.45 -> 0.75, gap 0.00 -> 0.15, generation-cap hits 8/20 -> 1/20. Five of the
six seeds that flipped to correct had stopped on `hit_generation_cap` at 2048,
so the mechanism is attributed per seed rather than inferred from the aggregate.
Its 0.45 was mostly truncation, exactly as suspected.

**It is also untrainable here.** TRL applies `max_completion_length` to the
whole multi-turn trajectory, not per turn (`grpo_trainer.py:1605` drops tool
results once `len(pct) - len(prompt_ids[i])` passes it), and the loss tensor is
sized to it, so memory scales with it directly. The fp32 logits upcast puts this
L4 near a 4096-completion ceiling; 8192 would want roughly 12.5 GB against the
~12 GB left after the model and vLLM. click-checkboxes-large spends a median
6066 assistant tokens per episode before page observations count, so 16 of 20
episodes would train truncated - the e9-e21 artifact, moved from eval to
training.

**And the trainable family is saturated.** navigate-tree fits (331 median
tokens) but reads 0.85 success against 0.85 termination at 4096: all 17
terminated episodes correct, the two axes fully collapsed. Its apparent 0.15 gap
at 2048 was itself a truncation artifact.

So difficulty and trajectory length are entangled across every family probed so
far: trainable and saturated, or interesting and too long. e27probe4 searches
the remaining quadrant - families decided in one or two clicks whose difficulty
comes from reading the page rather than acting many times.

**Train/eval budget mismatch, worth fixing regardless of which family wins.**
`max_seq - max_prompt_length` feeds two different things: TRL's whole-trajectory
`max_completion_length` in training (`grpo_runner.py:78`) and `max_new_tokens`
per turn in eval (`agentic_eval.py:36`, generate called once per turn with no
total cap). For a single-turn domain these coincide, which is why it never
surfaced on reasoning_gym. On an n-turn domain they differ by a factor of n:
e27's original geometry gave training 2048 tokens for a whole 8-turn episode
while eval allowed 2048 on every turn. `eval.max_new_tokens` overrides the eval
side without a code change if the two need to be set apart.

## Why e27 has not relaunched yet

Two probes have now run, and each ruled out one explanation for e27's
saturation. What is left is a budget question, which e27probe3 is answering.

**e27probe (8 families, 2048 budget) - harvested.** Difficulty re-measured
through the real adapter. Everything previously probed came in ~0.45 higher:
click-option 1.00, click-checkboxes 0.95, click-widget 0.90, navigate-tree 0.80,
click-checkboxes-soft 0.75, click-checkboxes-large 0.45, click-tab-2 0.10,
click-link 0.10. Both the training mix and the shifted split were saturated.

**e27probe2 (hint removed, same 2 families, paired seeds) - harvested, NULL.**
click-option 1.00 -> 1.00, click-checkboxes 0.95 -> 1.00. The `_LEAD_IN`
sentence that stated the solution was not what inflated difficulty; the two-tool
adapter surface (click, noop) against the first probe's five tools was. The
hint-free lead-in stays in `domains/browsergym/domain.py` regardless - it is
correct not to hand over the plan - but it is not the fix, so the family set is
what has to change.

**Which leaves the budget.** The two families that would centre the 0.40-0.80
band are the two whose scores are mostly truncation, not failure:
click-checkboxes-large hit the generation cap in 8 of 20 episodes plus 3
`max_turns`, click-tab-2 in 11 of 20. Picking a training mix off those numbers
would repeat the e9-e21 artifact exactly. e27probe3 re-runs all four survivors
at 4096 on the same seeds, with the two barely-truncated families
(navigate-tree 0/20, click-checkboxes-soft 3/20) as controls that should not
move. Whatever budget it validates is the budget e27 then trains at.

## Killed

**e27 was killed at step 21/300 (was pid 246083).** `frac_reward_zero_std` 0.70
over 20 steps - 14 of 20 groups had all 8 rollouts score identically - with mean
reward 0.931 and `reward/EnvReward/raw_std` 0. Saturated: no gradient, and
success and termination have collapsed onto one axis, so the run could not have
answered RQ2. Nothing to harvest; the run dir holds 21 steps and no checkpoint.

The cause is that the first feasibility probe measured difficulty under
conditions the training run does not use - browsergym's five tools and the
probe's own prompt, against the adapter's two tools and a `_LEAD_IN` that states
the solution pattern. e27probe re-measures through `eval.runner --base-model`,
which drives the real adapter, so its numbers transfer by construction.

Smoke passed before the kill (`tools/failure_frequency` 0, `call_frequency` 2,
reward 1, `clipped_ratio` 0). Its eval numbers are not a signal - `--smoke`
forces `eval.max_new_tokens = 256` and all four held-out episodes read
`hit_generation_cap`. Real runs compute `8192 - 6144 = 2048` instead.

Note when tailing: the run's stdout is block-buffered into the log, so the
per-step metric dicts arrive in ~8 KB bursts. Only tqdm is live. An empty grep
for `'reward':` early in a run means "not flushed yet", not "no reward".

## Before any browsergym launch

1. **MiniWoB static server must be up on :8080.** It does not survive a box
   restart. `cd /workspace/miniwob-plusplus/miniwob/html && python3 -m http.server 8080`
   (currently pid 217225, up since Aug 01). Check with
   `curl -o /dev/null -w "%{http_code}" http://localhost:8080/miniwob/click-option.html`.
2. **Port 8000 must be free.** `pgrep -af server.app` and kill by pid.
   `EnvServerProcess.start()` now raises instead of running on someone else's
   server, so this is loud rather than silent, but it still blocks the launch.

## Box state - read before the next run

- **browsergym is now installed in the pipeline venv** (`browsergym-core`,
  `browsergym-miniwob`, `playwright==1.44.0`), not only in `/workspace/bgym-venv`.
  `EnvServerProcess` runs the env server with `sys.executable`, so the server
  subprocess needs it in the same venv as training. The only version move was
  `greenlet 3.5.2 -> 3.0.3` for playwright's pin.
- **The browsergym server takes its port from `BROWSERGYM_PORT`, not the `--port`
  argv** `EnvServerProcess` passes. It always binds 8000 by default, which is what
  the e27 config asks for. A different `training.env_server.port` would fail loud
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

## Harvested

**BrowserGym/MiniWoB feasibility probes** - both done. Training mix click-option +
click-checkboxes, shifted split click-widget + navigate-tree. Findings in
`pipeline/runs/browsergym_feasibility_findings.md`, probes in `pipeline/probes/`.

**e26 finqa E1 qualification** - done. finqa disqualified (0/60 eval, 1%
per-rollout training accuracy, gradient on 10 of 300 steps). Findings in
`pipeline/runs/e26_finqa_qualification_findings.md`.

**e24/e25 4k pair** - done. Reports and train logs in
`pipeline/runs/e24-poly-4k-env-only-qwen3-1_7b/` and
`pipeline/runs/e25-poly-4k-cosine-w16-qwen3-1_7b/`; findings in
`pipeline/runs/e24_e25_4k_pair_findings.md`.

## Traps that have each cost real time

- **A leftover env server on the shared port silently serves the next run.** Every
  env's `server/app.py` binds one fixed port, so the new server dies on bind while
  the readiness probe passes against the old one. The first e27 smoke trained to
  completion against a day-old reasoning_gym server: every browsergym action came
  back `VALIDATION_ERROR`, giving `tools/failure_frequency` 1.0 and reward 0 with
  no traceback anywhere. `EnvServerProcess.start()` now refuses an occupied port.
- **`--smoke` overrides the budget geometry** to `max_seq_length=2048`,
  `max_prompt_length=1024`. That is 1024 tokens per turn against the config's
  2048, and 1024 is inside the range where the probe measured pure truncation
  artifacts. A smoke's clipped_ratio does not transfer to the real run.
- **`pkill -f <pattern>` over ssh matches the remote shell running it** when the
  pattern appears in that command line, killing the session before the work runs.
  Kill by pid.
