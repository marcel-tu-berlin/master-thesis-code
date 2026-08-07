# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

The project studies token-efficiency reward shaping in GRPO (a cosine length
reward, a non-termination penalty) in an agentic, multi-environment setting.
Training runs against live OpenEnv environments through TRL's
`environment_factory`; the policy is a tool-calling model (Qwen3-1.7B) rewarded by
the environment, not by grading an answer string. The pipeline is agentic-only.

## This is scientific work, so a regression is the worst outcome

Every number under `pipeline/runs/` cost GPU hours and will be cited in a thesis.
A silent behaviour change does not merely break a feature: it invalidates results
that were already collected, and it does so quietly, usually weeks before anyone
notices. Two campaigns have already been thrown away this way (see "Void results"
below). That is the failure mode to design against, ahead of speed and ahead of
elegance.

What it requires on every change:

- **Verify the assumption before writing code against it.** Read the function,
  the config key, the installed version. "It probably works like X" is how the
  cosine reward ended up shaping 8% of the tokens it claimed to shape.
- **Know what the change touches.** Grep every caller before editing a shared
  function. A fix that is right at one call site and wrong at its sibling is worse
  than no fix, because plausible numbers still come out the other end.
- **Run the tests.** `pipeline/tests/` is CPU-only and finishes in seconds. A
  change to training or eval semantics also needs a test that fails without it.
- **Never change a measurement silently.** Token counting, the correctness
  threshold, the seed mapping, budget accounting, metric definitions: changing any
  of these makes new runs incomparable with old ones. If the change is right
  anyway, say so explicitly, name the results it invalidates, and record it in
  `LAB_NOTES.md`.
- **A smoke test is not a validation.** It proves the code runs, not that it
  measures the intended quantity. The e9-e21 sweep smoke-tested clean for months
  while measuring the wrong thing.

When in doubt, stop and confirm rather than ship and let the next eval catch it.
The next eval is nine hours of GPU time.

## Void results, do not cite

Two errors invalidated whole campaigns. Both are written up in full in
`LAB_NOTES.md` and `pipeline/runs/cosine_token_count_bug_findings.md`; the short
form lives here so no argument gets rebuilt on them.

- **e9-e21 (the poly and countdown cosine sweeps).** The cosine reward counted
  only the visible answer text and not the `reasoning_content` that made up most
  of the completion. Separately, every wrong episode sat exactly at the token cap,
  so "wrong" and "truncated" were one label.
- **Everything trained before `a19b1ff`.** `batch_size` defaulted to 1, so an
  optimizer step saw a single prompt-group and 40-60% of steps carried no gradient
  at all. Every treatment-versus-baseline pair from that era is confounded by how
  much live gradient each arm happened to receive.

The 4096-cap pair (`e24bs4` / `e25bs4`) is the first comparison free of both. Its
configs were never tracked and existed only as frozen run copies; they are now in
`configs/` so the pair can be re-run and seed-replicated.

**Re-running a void-era config does not reproduce its void run.** 26 configs in
`configs/` predate `a19b1ff` and state no `batch_size`, so they trained at 1 but
would resolve to 4 today - a different experiment under the same
`experiment_id`. The overwrite refusal catches the collision, not the semantic
change. `e24` and `e25` carry a header pointing at their `bs4` successors because
the names invite the mistake; treat every other pre-`a19b1ff` config the same way.

## Git workflow

Work on `master`. Always. Do not create branches or git worktrees for this repo -
no feature branches, no `.claude/worktrees/` copies. Make changes directly on
`master` in the main checkout, and commit there only when asked. If a tool or skill
drops you onto any branch other than `master`, stop and move the work back onto
`master` before continuing.

## Where things are written down

| File | Holds |
|---|---|
| `RUNNING.md` | What is executing on the box right now. Nothing else. |
| `LAB_NOTES.md` | What was learned: box operations, launch checklists, traps, standing decisions, run history. |
| `pipeline/runs/*_findings.md` | Per-run and per-pair numbers, with the statistics and the caveats attached. |
| `BACKLOG.md` | Work that is still open and not started. Finished and abandoned items get deleted, not annotated. |
| `DECISIONS.md` | Why each OpenEnv environment was integrated or rejected. |
| `pipeline/CODE_REVIEW.md`, `pipeline/FIX_PLAN.md` | The 2026-08-03 review and how each finding was closed. |
| taskwarrior (`project:thesis`) | Open work and what comes next. |

Each fact belongs in exactly one of these. The failure mode is not hypothetical:
`RUNNING.md` once grew to 950 lines of rules and history, which buried the single
question it exists to answer.

### Run state (`RUNNING.md`)

`RUNNING.md` at the repo root records what is executing on the GPU box. Keeping it
current is a hard requirement for every run, not a courtesy:

- Add the row before you launch, and fill in the pid and the ETA once the launch
  is confirmed live.
- Update it at every phase change (train -> eval), kill, and crash.
- Delete the row once the phase is done and its results are harvested. A phase
  that finished but has not been read yet stays, marked `done, unharvested`.
  The file is live state, not a run history; `runs/` and git hold the record.
- Stamp `Updated:` with box time (`ssh gpu-l4 date`), never local time.
- When nothing is running or queued, write "Nothing running." A stale table is
  worse than an empty one.

A run missing from `RUNNING.md` is a run nobody can find after a context reset.

**Nothing else goes in this file.** It is a table of live runs, not a notebook.
Findings go to `pipeline/runs/*_findings.md`; everything else - operational rules,
launch checklists, traps, standing design decisions, run history - goes to
`LAB_NOTES.md`.

### Lab notes (`LAB_NOTES.md`)

`LAB_NOTES.md` at the repo root holds what was learned rather than what is running:
box operations, the browsergym launch checklist, traps that have already cost time,
standing decisions such as the `batch_size 4` rule, and the narrative history behind
them. Read the operations sections before launching anything on the box, and add to
it whenever something costs time twice.

### Backlog (`BACKLOG.md`)

Open work only. `BACKLOG.md` answers one question - what could someone pick up
next - and an item that is finished or abandoned answers nothing while still
costing a read.

**Delete an item the moment it is done or decided against. Do not mark it DONE or
DROPPED and leave it in place.** A file where most entries are closed trains the
next reader to skim, which is how the one live item gets missed. This is the same
failure `RUNNING.md` had at 950 lines.

If the reasoning behind a dropped item is worth keeping, move it to the file that
owns that fact before deleting - `DECISIONS.md` for environment choices,
`LAB_NOTES.md` for traps and standing decisions, `pipeline/runs/*_findings.md` for
numbers. Git holds whatever nobody moved, so a deletion loses nothing.

### Task tracking (taskwarrior)

Pipeline work is tracked in taskwarrior under the `thesis` context (`project:thesis`).
`RUNNING.md` says what the GPU is doing right now; taskwarrior says what work is
open and what comes next. Do not duplicate a run's live state into a task.

Always pass the context explicitly, so the command is correct no matter which
context happens to be active:

```bash
task rc.context=thesis add "run seed 43/44 replication of e24/e25"   # add
task rc.context=thesis list                                          # read open work
task rc.context=thesis <id> annotate "bootstrap CI crosses zero"     # record a finding
task rc.context=thesis <id> modify +blocked                          # retag / re-scope
task rc.context=thesis <id> done                                     # finished
task rc.context=thesis rc.confirmation=off <id> delete               # obsolete, never done
```

`delete` prompts for confirmation, which hangs a non-interactive shell - hence
`rc.confirmation=off`. Only ever pass it to `delete`.

When to touch it:

- **Add** when work is agreed but not started - a planned experiment, a fix that
  is out of scope for the current turn, a harvest that has to wait for a run.
  One task per unit of work someone could pick up cold.
- **Annotate** when a task's outcome or blocker becomes known but the task is not
  finished. Findings belong on the task; the numbers stay in `runs/`.
- **Done** as soon as the work lands and is verified, not when the code is
  written. A launched run is not a done task.
- **Delete** when a task is obsolete (premise refuted, superseded by another
  task). `delete` is not `done` - do not close abandoned work as completed.

Read the open list before proposing next steps, and reconcile it when a run is
harvested. A stale task list misdirects the next session exactly like a stale
`RUNNING.md` does.

## Where code runs

Edit and run tests on the Mac. Training and eval run on the GPU box
(`ssh gpu-l4`, an L4 with 24 GB), which keeps its own checkout at
`/workspace/master-thesis-code`.

```bash
# tests: CPU-only, a few seconds, no GPU stack required
cd pipeline && ../.venv-test/bin/python -m pytest tests/ -q

# push code to the box before any launch - the whole package, never one module
rsync -a --delete --exclude 'runs/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  pipeline/ gpu-l4:/workspace/master-thesis-code/pipeline/

# harvest results back; checkpoints stay on the box
rsync -a --exclude 'checkpoint*' \
  gpu-l4:/workspace/master-thesis-code/pipeline/runs/<experiment_id> pipeline/runs/
```

Sync the whole `pipeline/` tree. A single-module sync has produced a mixed-version
pipeline that ran to completion against stale code, which is the expensive version
of this mistake; an import error is the cheap version. The box copy is assumed
stale until a sync says otherwise.

Box operations, the browsergym launch checklist and the ssh-failure diagnosis are
in `LAB_NOTES.md`. Read them before launching.

## Environment Setup

```bash
./setup.sh
```

Creates `.venv` via `uv` with Python 3.12. Installs the GPU stack (`trl`, `peft`,
`bitsandbytes`, `accelerate`, a pinned `vllm` cu130 wheel,
`reasoning-gym`) and clones `meta-pytorch/OpenEnv` to `/workspace/OpenEnv` (its
env servers are not on PyPI). Hardware target: NVIDIA L4 (24 GB) or RTX 4090,
CUDA 13.0, Linux. `--torch-backend=auto` selects the torch variant.

The `openenv` core is installed **editable from that clone**, not from PyPI as
`openenv-core`. Both ship the same `openenv` import name, so installing both lets
whichever shadows the other decide the version - and a skew fails at runtime, not
import time: the repo's env clients pass `metadata=` to `StepResult`, which older
cores reject with a `TypeError` on the first `reset()`. After updating the clone,
re-verify the envs with a real reset, never with an import.

## Running the Pipeline (`pipeline/`)

The pipeline is the surface for systematic experimentation. Full docs in
`pipeline/README.md`. Run from `pipeline/`:

```bash
python -m training.train --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml --eval
python -m eval.runner --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml
python -m training.batch configs/e5-*.yaml --train --eval --seeds 42 43 44
```

Add `--smoke` to any command for a fast sanity check (3 steps, 10 eval episodes).

Results on disk are protected by three refusals, all of them there because the
overwrite already happened once: `training.train` refuses to clobber an existing
`runs/<experiment_id>/` (pass `--overwrite`, or change `experiment_id`; the frozen
`config.yaml` and `checkpoint-final/` are the trigger artifacts), a `--smoke` eval
refuses to write over a real `eval_report.json`, and `--base-model` refuses to
write into a directory holding a trained checkpoint. Give a throwaway run its own
`experiment_id` rather than working around any of them.

Outputs land in `runs/<experiment_id>/`: frozen config, LoRA checkpoint, and the
agentic eval report (`eval_report.json` / `eval_report.md`, keyed under the
`agentic` split).

### Experiment geometry: one knob differs, never two

Every arm of a comparison runs `batch_size: 4`, `n_rollouts: 8` and the same
`max_steps`. Fixing only one of the three rebuilds the confound: an arm that saw
more prompts, or more optimizer updates, than its baseline is not a reward
ablation. Two configs being compared should differ in exactly one `rewards:` key,
and the diff of their frozen `runs/<exp>/config.yaml` files is what proves it -
check that diff, not the config you intended to write. The reasoning behind the
value 4, and why it is not a memory knob, is in `LAB_NOTES.md`.

`max_steps` has to match across the arms of one comparison, not across campaigns.
The browsergym arms run 300 and the poly pair ran 150; those are different
questions on different environments and nobody compares a number across them.

**Write the geometry keys out, do not lean on the defaults.** Freezing copies the
config verbatim - it does not resolve defaults - so a key no config states is
absent from every frozen copy, and the diff above then proves nothing about it. It
reads as "equal" when it is really "unrecorded". That matters most for exactly the
key it already went wrong on: `batch_size` silently resolving to 1 is what voided
every run before `a19b1ff`, and `grpo_runner` now defaulting it to 4 fixes the
value without making it visible.

### Batch runner

`training.batch` queues many configs through training and eval as subprocesses
(one fresh Python process per phase, so GPU memory is released cleanly between
runs). Built for unattended ablation and seed sweeps on a single GPU.

```bash
python -m training.batch configs/e5-*.yaml --train --eval
```

Phase flags `--train` and `--eval` are independent and combinable; default when
neither is given is `--train --eval`. Other flags: `--seeds A B C` (replicate
each config across seeds, each into its own `<exp>-s<seed>` run dir), `--smoke`,
`--vllm`, `--force` (re-run even if outputs exist; passes `--overwrite` to train),
`--retries N`. Skip predicates are content-aware and resume-friendly: a
non-smoke `checkpoint-final/` skips train and a real `eval_report.json` skips
eval, so a re-run after a crash picks up where it stopped. Per-phase logs land at
`runs/<exp>/batch_{train,eval}.log`; an end-of-batch summary is written to
`runs/batch_summary_<timestamp>.md`.

## Architecture

### Domains

A domain wraps one OpenEnv environment. `EnvDomain` (`domains/env_base.py`) is the
interface: `make_env_factory` (a zero-arg callable that builds one env adapter
against the server `base_url`), `build_seed_dataset` (rows of `{prompt, seed}`,
one distinct question per seed), `episode_messages` (the eval prompt for a
question), `episode_reward` / `is_correct` (read the env score), and
`server_module` (the `python -m ...` server entry point the runner launches).

`build_domain(config)` (`domains/__init__.py`) maps `training.env` to a domain and
imports it lazily, so one environment's dependencies never block another. Two exist:
`reasoning_gym` (the reward-science domain, task families such as
`polynomial_equations`) and `browsergym` (the E1/E2/E3 domain, MiniWoB).

Three domains existed and were deleted. `finqa` after e26 disqualified it (0/60
held-out, 1% training accuracy, a gradient on 10 of 300 steps); `textarena` and
`repl` because neither backs a live experiment and both were carrying cost -
textarena a PyPI dependency plus an NLTK corpus download in `setup.sh`, repl a
deliberately trivial arithmetic task family. None of the three produced a number
anyone cites, so nothing is invalidated; `git show ace8954` (finqa, repl) and
`git show 2b26343` (textarena) restore them if a premise returns.

Two things they leave behind. The e26 findings stay at
`pipeline/runs/e26_finqa_qualification_findings.md` because they establish the
criterion every candidate environment is now judged against: base-model accuracy in
the 40-80% band, and at least two tools. And the server contract finqa's patches
encoded (`MAX_CONCURRENT_ENVS`, a deterministic `reset(seed=N)`) is in
`LAB_NOTES.md` and applies to any new env.

Each config seed owns a disjoint block of the seed-to-question mapping (question
seed = `config_seed * SEED_BLOCK + offset`, `SEED_BLOCK = 1_000_000` in
`training/config_schema.py`). Before that block existed, `--seeds 42 43` shared
roughly 99% of their questions, so a seed sweep measured nothing. Validation
rejects an eval `seed_offset` large enough to cross into the next seed's block.

### Agentic training loop

`training.mode: agentic` (the only supported mode) with `training.env: <domain>`
trains against a live OpenEnv server. The model is driven through its native
tool-calling template and rewarded by the environment. The walkthrough below uses
`reasoning_gym`; the other domains differ only in their adapter and server module.

- **Server lifecycle (runner-owned).** `EnvServerProcess` (`training/env_server.py`)
  launches the OpenEnv env as a local HTTP server subprocess (no Docker:
  `python -m reasoning_gym_env.server.app`), waits for the port, and stops it after
  training. The server code lives in a clone of `meta-pytorch/OpenEnv` at
  `training.env_server.repo_path` (default `/workspace/OpenEnv/envs`). One server
  serves every rollout-slot client; `MAX_CONCURRENT_ENVS` is sized to
  `batch_size * n_rollouts`.
- **Env-factory adapter.** `ReasoningGymEnvAdapter` (`domains/reasoning_gym/adapter.py`)
  wraps the OpenEnv sync client. TRL's `GRPOTrainer(environment_factory=...)` builds
  one adapter per rollout slot, calls `reset(**row)` (its return is appended to the
  prompt), and exposes every other public method as a tool. The adapter's public
  surface is exactly `{reset, answer}`, so the model sees one tool, `answer(answer:
  str)`; the tool docstring needs a Google-style `Args:` block (transformers builds
  the tool JSON schema from it). The adapter stores the env score on `self.reward`.
- **Dataset.** `build_seed_dataset` returns `{prompt, seed}` rows; each seed is a
  distinct, deterministic reasoning_gym question. TRL repeats each row
  `num_generations` times to form a GRPO group.
- **Rewards.** `EnvReward` reads `[e.reward for e in kwargs["environments"]]`.
  The efficiency rewards are the other live signals; `CosineLengthReward` takes
  correctness from `environments` (env reward >= `CORRECT_REWARD_THRESHOLD`,
  0.5 - reasoning_gym scorers are graded and hand partial credit to
  near-misses) instead of an answer column.

Validated end to end on an L4 (24 GB) with `Qwen/Qwen3-1.7B` + vLLM colocate: the
model calls the tool, env reward flows into the composer, and the LoRA saves.

### Reward Registry

Rewards are wired via `REWARD_REGISTRY` in `pipeline/training/rewards/__init__.py`.
Each entry maps a config key (under `rewards:`) to `(default_enabled,
default_weight, builder)`. Three exist: `env_reward` (task success), `token_length`
(cosine length), `non_termination`. All default off; configs enable what they
study. Adding a reward requires both a builder + registry entry and the matching
key in `_KNOWN_REWARD_KEYS` (`pipeline/training/config_schema.py`), or validation
rejects it. `train.build_reward_components` iterates the registry, so there are no
per-reward branches in `train.py`.

A token-entropy reward existed and was deleted. It rendered the prompt without
`tools=`, so it measured the entropy of a completion given a context the policy
never saw, and it is not among the expose's conditions (E0, E1, E2 length, E3
non-termination, optional E4 combined). Nothing in the code, the configs or the
schema refers to it any more, and a config naming `token_entropy` now fails
validation. Do not re-add it without fixing the conditioning first.

`NonTerminationPenalty` is the E3 signal: -1 for an episode whose env never
reported `done`, 0 otherwise, so a config's `weight` is the penalty coefficient
lambda of `R_task - lambda * C_target`. The sign lives in the component so every
registry weight stays positive. It is binary, so `advantage_weighted` z-scores it
to zero in any prompt-group where all rollouts terminate - run the lambda sweep
under `naive_sum`; `warn_inert_scalars` warns otherwise.

`CosineLengthReward` (Wu/Yeo 2025) is the single token-length reward: correct
completions are rewarded more when shorter, wrong completions penalized less when
longer, making wrong-and-short the most-penalized cell. The reward is non-linear
in length and gated by correctness, so it survives per-group z-scoring with real
structure.

Length is measured by `model_token_count` (`training/rewards/utils.py`) over every
assistant message, summing `content`, `reasoning_content` and serialized tool-call
arguments, and skipping the tool messages TRL interleaves. Both halves of that are
load-bearing and both were once wrong. Dropping `reasoning_content` measured 18
tokens of a 1024-token completion, because transformers puts the think block there
on any assistant message carrying a tool call. Preferring `len(completion_ids)`
branched on whether a tool message had been interleaved, which happens exactly when
the model called a tool, so the ruler itself became a proxy for correctness.
`tests/test_cosine_length_ids.py` and `tests/test_model_token_count.py` hold both
closed. Any change to how a completion is counted is a change to the experiment.

### Reward Composition

Components are combined via a composer selected by `rewards.compose_method`:

- **`advantage_weighted`** (default) - `AdvantageWeightedComposer`
  (`pipeline/training/rewards/compose.py`). Per-prompt-group z-scoring of each
  component's raw rewards before the weighted sum (DIET 3.2): raw variance differs
  across components, so a naive sum lets a high-variance signal dominate regardless
  of weight. A component with zero within-group variance contributes 0, by design.
- **`naive_sum`** - `NaiveSumComposer`. Plain weighted sum, no normalisation. The
  ablation baseline that isolates the advantage-weighting effect.

**Scale-invariance.** Because `advantage_weighted` z-scores each component per
prompt-group, it is invariant to any global positive scalar inside a component's
raw reward: a knob that multiplies every rollout in the group does nothing (a
negative scalar flips the sign, zero silences it). The live levers are the
component `weight`, the per-completion signal shape, and switching to `naive_sum`.
`build_reward_components` calls `warn_inert_scalars` and warns when a knob is inert
as configured; treat that warning as a sweep that will produce a flat line.

**Adding any component changes how much gradient the arm gets.** Under
`advantage_weighted` a prompt-group with zero within-group reward variance yields
zero advantage, so it trains on nothing. On a near-saturated task the env reward is
constant across a group most of the time, while a length reward varies whenever the
rollouts differ in length. Measured on the e24bs4 / e25bs4 pair: 40.7% of groups
live in the control arm against 99.5% in the cosine arm. The contrast that produces
is "reward plus more gradient", not the reward alone. Check
`frac_reward_zero_std` in both arms' training logs before reading any pair, and say
so in the write-up when they differ.

### Agentic Evaluation

`eval.runner` dispatches to `run_agentic_eval` (`pipeline/eval/agentic_eval.py`):
it loads the trained LoRA, launches the env server, runs N held-out episodes
(inside the run's own seed block, offset by `_EVAL_SEED_OFFSET = 100_000`, so they
are disjoint from that seed's training questions), generates greedily unless
`eval.do_sample`, parses the tool call, scores it via the env, and writes
`eval_report.json` / `.md` keyed by split name (`agentic` unless
`eval.agentic.splits` names others).

The generation budget is a **whole-trajectory** budget, matching what TRL applies
in training, and interleaved tool responses are charged against it too. Applying it
per turn let an episode generate `max_turns` times the training budget. It defaults
to the training budget (`max_seq - max_prompt_length`); a smaller eval cap silently
truncates completions before the tool call and tanks the success rate.

Metrics come from `eval/metrics.py`: accuracy with a Wilson 95% interval,
`mean_token_count_correct` (correct episodes only, with a bootstrap CI),
underthinking / overthinking rates, mean steps, and the off-target panel below.

Three things to hold onto when reading a report:

- **`mean_token_count_correct` is the efficiency number, not `mean_token_count`.**
  The pooled mean is dominated by failures, which run to the cap, so a change in
  failure rate is indistinguishable there from a change in length. That confound is
  part of what voided e9-e21.
- **Under/overthinking thresholds are per-run percentiles unless pinned.**
  Two arms scored on their own P10/P75 are measured against different rulers and
  their rates are not comparable. `load_reference_thresholds` pins them to a
  reference report; use it for any cross-arm claim.
- **A comparison is only paired if both arms answered the same questions.** Compare
  on the intersection and report the paired statistic. Unpaired means shift when
  the treatment arm solves questions the control missed, which are usually the long
  ones, and the mean then moves the wrong way while the paired median falls.

**Protocol splits.** `eval.agentic.splits` is a list of `{name, n_episodes,
env_config, seed_offset}`; `env_config` merges over the training one and each
split gets its own env server. This is how the held-out and shifted splits of the
evaluation protocol are expressed. `--base-model` evaluates the base model with no
adapter (E0) through the identical loop.

**Off-target panel (RQ2).** Every episode records `terminated`, `stop_reason`
(`env_done` / `no_tool_call` / `max_turns` / `hit_generation_cap`) and its ordered
`tool_calls`, persisted to `runs/<exp>/episodes_<split>.jsonl`. `compute_metrics`
derives non-termination rate, unsupported-claim rate (terminal tool called with
nothing before it) and mean verification depth, all Wilson-CI'd. Two rules this
enforces: an episode that never called the tool is no longer scored as merely a
wrong answer, and a completion that filled the token budget is never labelled the
same as one where the model stopped on its own - that conflation is what made the
e9-e21 sweep uninterpretable. The claim/verification numbers are degenerate in a
single-tool domain such as reasoning_gym and only carry information in a
multi-tool env.

**One turn cap.** `training.env_config.max_turns` is the only turn-cap key for
every multi-turn domain. Both readers go through `resolve_max_turns`
(`training/config_schema.py`): training passes the result to TRL as
`max_tool_calling_iterations` and the eval loop runs that many turns, so an unset
key means the same one-iteration episode on both sides instead of a silent
train/eval divergence. TRL treats an unset `max_tool_calling_iterations` as
`sys.maxsize`, so a per-domain alias leaves the training tool loop unbounded -
which is why the schema still rejects the aliases the deleted domains used
(`max_steps`, `max_iterations`). browsergym has no server-side step cap, so its
cap is client-side only and `server_env` maps nothing; a domain whose server does
cap steps maps `max_turns` onto that var there.

### LoRA Configuration

Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`. `lora_alpha = lora_r * 2`. The model loads in bf16 (or 4-bit nf4 when
the registry sets `load_in_4bit`), with vLLM colocate generation and micro-batch +
grad-accum to fit 24 GB.

### Outputs

- `runs/<experiment_id>/config.yaml` - frozen experiment config
- `runs/<experiment_id>/checkpoint-final/` - LoRA adapter + tokenizer
- `runs/<experiment_id>/eval_report.json` / `eval_report.md` - agentic episode
  metrics (success rate, token efficiency)
