# Running

What is executing on the GPU box (`ssh gpu-l4`). Rows leave the table once the
results are harvested. Update rules are in CLAUDE.md.

Updated: 2026-08-03 05:22 UTC (box time)

| Run | Phase | pid | Started | ETA | Notes |
|---|---|---|---|---|---|
| e0 browsergym base model | eval, 200 eps | 738769 | 05:18 UTC Aug 3 | ~07:10 UTC | `--base-model`, no adapter. Same two splits, seeds and 4096 budget as e27, so it pairs episode for episode. Log `/workspace/e0.log`. Watcher 739458 writes `/workspace/e0_final.txt` on exit (fires on crash too). |

## What e0 is for

e27 finished and its numbers raise one question that only a paired baseline can
answer. Against the n=20 probes the trained policy looks like it gained where it
trained (click-menu-2 0.60 -> 0.70, click-dialog-2 0.80 -> 0.86) and lost where
it did not (navigate-tree 0.85 -> 0.62, click-checkboxes-transfer 1.00 -> 0.96).

That comparison is unsound: the probes ran 20 episodes per family at
`seed_offset` 300000, the eval splits 50 per family at 100000 and 200000. e0
runs the identical splits at the identical seeds, which makes it paired. If
navigate-tree comes back near 0.85, task-success-only training cost 0.23
accuracy on an unseen family and every later condition is read on top of that.

Do not report a transfer claim before this lands.

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
- **Stale watcher loops from the probe runs are still resident** (pids 273631,
  297350) spinning on `pgrep -f "eval.runner --config configs/e27probe"`. Harmless
  now, but they would latch onto a new probe run with a matching command line.
  Kill by pid before the next `e27probe*` launch.

## Harvested

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
- **`Connection refused` on ssh does not mean the box died.** The devpod's ssh
  proxy drops while the host still answers ICMP and every process keeps running.
  Check `uptime` and `pgrep` before concluding anything died, and never relaunch
  on the assumption that it did.
- **`eval.runner --base-model` writes into `runs/<experiment_id>/`** regardless of
  whether an adapter was loaded (`runner.py:36,51`), so pointing it at a trained
  run's config overwrites that run's `eval_report.json` and episode files. A
  base-model eval needs its own config with its own `experiment_id`.
