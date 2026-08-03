# Running

What is executing on the GPU box (`ssh gpu-l4`). Rows leave the table once the
results are harvested. Update rules are in CLAUDE.md.

Updated: 2026-08-03 07:30 UTC (box time)

| Run | Phase | pid | Started | ETA | Notes |
|---|---|---|---|---|---|
| e28 (E2 cosine) + e29 (E3 non-termination) | batch: train + eval, both arms | 769998 (batch), 769999 (e28 train) | 07:28 UTC Aug 3 | e28 ~21:00 Aug 3, e29 ~10:30 Aug 4 | One `training.batch` over both configs. Log `/workspace/e28_e29_batch.log`, per-phase logs at `runs/<exp>/batch_{train,eval}.log`. Watcher 771563 writes `/workspace/e28_e29_final.txt` on exit; watcher 771562 writes `/workspace/e28_step30.txt` at step 30. |

**e28 step-30 check is armed and is the one to read first.** It reports
`reward/CosineLengthReward/contrib_l1` alongside the env component. If the
cosine line comes back MISSING or near zero the reward is inert and the run
should be stopped rather than left to burn 270 more steps - that is the failure
mode the poly campaign only caught after the fact.

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

**The cosine's weight is a structurally weak lever under naive_sum, and this
retro-explains the poly null.** Raising `w` scales the cosine's length term and
its own correctness gating equally; only env_reward's fixed 1.0 stays put.
Computed on e27's own within-group spread (985 to 2051 tokens, max_len 4096),
length's share of the total advantage range reads:

```
w      1      2      4      8     16     -> asymptote
share  0.063  0.077  0.086  0.091  0.094    0.097
```

A sixteenfold weight change buys a factor of 1.5, and no weight puts length past
a tenth of the range. The e9-e21 campaign read its flat w1-w16 dose-response as
a property of the task; a good part of it is this. e28 therefore runs one weight
(1.0) rather than a sweep - a null there is not answered by a bigger weight, it
is answered by the cosine's endpoint spread, which is Wu/Yeo's parameterisation
and not a knob to quietly retune.

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
- **`Connection refused` on ssh does not mean the box died.** The devpod's ssh
  proxy drops while the host still answers ICMP and every process keeps running.
  Check `uptime` and `pgrep` before concluding anything died, and never relaunch
  on the assumption that it did.
- **`eval.runner --base-model` writes into `runs/<experiment_id>/`** regardless of
  whether an adapter was loaded (`runner.py:36,51`), so pointing it at a trained
  run's config overwrites that run's `eval_report.json` and episode files. A
  base-model eval needs its own config with its own `experiment_id`.
