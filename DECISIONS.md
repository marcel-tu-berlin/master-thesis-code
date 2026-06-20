# Adding OpenEnv environments: finqa and repl

Decisions and trade-offs from integrating two new agentic environments, for review.

## Summary

| Env | Status | One-line reason |
|-----|--------|-----------------|
| finqa | integrated | Real financial QA, automatic binary reward, deterministic after a small server patch. |
| repl | integrated | Code-execution env; ships no tasks, so we supply a deterministic task generator with an exact-match reward. |

Both smoke green (`--smoke --eval --vllm`, exit 0). They are wired like
reasoning_gym/textarena: a domain plus an adapter, one `build_domain` branch, one
`_KNOWN_ENV_CONFIG_KEYS` entry, one config. The eval loop needed a small
generalization (below).

A third env, a Google-Calendar-style MCP server, was prototyped and then removed.
Its reward is task-blind - the server returns a positive scalar for any tool call
that succeeds at the HTTP layer, never checking whether the goal was met - so it
cannot drive the reward-shaping study. Not worth carrying dead. The launch-override
hook it needed (`server_command`) was reverted with it, so `env_server.py` is back to
the plain `python -m <module> --port` path.

## Why each verdict

### finqa (integrated)

FinQA serves financial questions over SEC-filing tables. The model explores tables
with three read tools, then submits a number; the env scores it 1.0 (correct) or 0.0
(wrong) with a fuzzy numeric match. That is exactly the automatic scalar reward the
GRPO pipeline needs, and the task is multi-turn with real token slack, which is what
the cosine-length study wants.

Three things about the shipped env did not fit the pipeline, so we patch the OpenEnv
clone (see `patch_openenv.py`, applied on the box; re-run it after re-cloning OpenEnv):

1. Non-deterministic questions. The server picks questions with an unseeded shuffle,
   so `reset(seed)` ignored the seed. That breaks the seed -> question contract and,
   worse, GRPO's assumption that every rollout in a group solves the same problem (one
   server hands each concurrent rollout a different question). Patch: `reset(seed=N)`
   selects `questions[N % len]`.

2. Single session. `create_app` defaults `max_concurrent_envs=1`, and finqa never
   raised it, so the server closed every websocket after the first. The pipeline runs
   one server for many rollout-slot clients, so the second client's reset died with
   `ConnectionClosedOK`. reasoning_gym/textarena already pass the env var through.
   Patch: read `MAX_CONCURRENT_ENVS` in finqa's `app.py`, and mark the env class
   `SUPPORTS_CONCURRENT_SESSIONS = True` (safe: each session gets a fresh, isolated
   env instance).

3. The question never reached the client. finqa's reset puts the question in
   `Observation.metadata`, but the OpenEnv serializer drops `metadata` on the wire
   (reasoning_gym/textarena put their payload in observation fields instead, which
   survive). Rather than patch shared serializer code, the adapter reads the question
   locally from the same `finqa.csv` the server uses, keyed by the same seed. Display
   text and scoring stay consistent because both are `questions[seed % len]` over one
   file. The server reset still runs to arm the ground-truth answer.

Reward plumbing detail: the MCP client's `call_tool` returns only the tool's string
and drops reward/done, so the terminal `submit_answer` goes through
`step(CallToolAction(...))` to read the reward off the StepResult. The three
read tools use `call_tool`.

Data: the dataset is not in the repo. We download `snorkelai/finqa-data` (public, MIT
on HF) into the OpenEnv clone at `/workspace/OpenEnv/envs/finqa_env/data` and point
`FINQA_DATA_PATH` there. The env's own `download_data.sh` needs `huggingface-cli`,
which the box lacks; `hf download` works instead.

### repl (integrated)

A Python REPL: the model writes code, reads stdout, and prints `FINAL(answer)` to
finish. The env runs the code in an in-process sandbox (smolagents, restricted
imports) and scores the final answer by exact match. Good fit for the length study:
correct solutions vary in length, and the reward is automatic.

The catch: repl ships no tasks. It is a bare sandbox, and `reset` takes the
`context`/`task_prompt`/`expected_answer` from the caller. So we own the task
distribution. `domains/repl/tasks.py` mints a deterministic task per seed
(arithmetic over a list: sum/max/min). The adapter derives the task from the seed and
threads `expected_answer` into the server's rubric, so training and eval both call
`reset(seed=N)` and get the same task, symmetric with reasoning_gym. The arithmetic
family is deliberately simple; it gives a real reward and real token slack. Swapping
in a richer task source (for example reasoning_gym-minted question/answer pairs) is a
later change isolated to `tasks.py`.

Dependencies: repl needs `smolagents`, `gradio`, `pypdf` in the venv (the first is
load-bearing; the other two are imported by the server's web UI module, which
`app.py` imports unconditionally). Install them with uv, not pip - the venv is
uv-managed and has no pip (`uv pip install --python .venv/bin/python smolagents
gradio pypdf`). uv resolved them without disturbing the pinned torch/vllm/trl
stack. No server patch is needed: repl already passes `max_concurrent_envs` and
marks its env `SUPPORTS_CONCURRENT_SESSIONS=True`, and like textarena its `app.py`
binds port 8000 and ignores `--port`, so the config targets 8000. (The earlier
scout reported an import crash; that was the missing `smolagents` surfacing, not a
separate bug.)

## Shared changes

- `training/train.py`: two `build_domain` branches (finqa, repl). Also fixed the
  smoke override (`apply_smoke_overrides`): the old hard `max_seq_length=512` cap
  rejected finqa's ~700-token tool-rich prompt outright, so smoke now keeps the
  config context (capped at 2048 to fit the L4) and lowers vLLM colocate's
  `gpu_memory_utilization` to 0.45 so the policy-grad backward does not OOM
  against vLLM's KV pool. This was a latent bug for any tool-heavy config, not
  finqa-specific.
- `training/config_schema.py`: env_config keys `data_path`, `max_steps` (finqa) and
  `max_iterations` (repl).
- `eval/agentic_eval.py`: the multi-turn eval loop was hardcoded to textarena's `move`
  tool. It now dispatches by the parsed tool name, restricted to the domain's declared
  `eval_tools`, so one loop drives textarena (move), finqa (four tools), and repl
  (execute). Malformed model arguments become tool feedback instead of crashing the
  episode. reasoning_gym's single-step path is unchanged.
- `patch_openenv.py` (repo root): idempotent, re-appliable patches to the OpenEnv
  clone. Run `python patch_openenv.py [envs_path]` after cloning OpenEnv. It carries
  the three finqa patches above. The OpenEnv clone is not vendored here, so a re-clone
  reverts these and the script must be re-run.

## Configs

- `e14-agentic-finqa-env-only-qwen3-1_7b.yaml`: finqa, env reward only.
- `e15-agentic-repl-env-only-qwen3-1_7b.yaml`: repl, env reward only.

Both start as env-only baselines (cosine length off), matching the e12 pattern; their
cosine-length ablation arm is a later config.

## Validation (smoke results)

Both envs pass `python -m training.train --config <cfg> --smoke --eval --vllm` on the
L4 (3 training steps + LoRA save + eval report, exit 0):

- finqa (e14): train 68s, eval 10 episodes, report written. The reward path was
  also checked by driving the adapter against a live server: reset returns the
  question + company, the data tools return rows, and a wrong submit_answer scores
  0.0 (the env reward is binary 1.0/0.0). The 3-step smoke model emits no valid
  tool call at eval time (mean_steps 0), so the eval-side dispatch of finqa's four
  tools rests on that standalone check and the generic dispatch rather than the
  smoke eval itself; training rollouts did exercise the tools (completions ~700
  tokens with env reward flowing).
- repl (e15): train 32s, eval 4 episodes (mean_steps 0.25, so the model did call
  execute at eval). The reward path was verified directly: reset(seed=5) yields
  "minimum of [80, 33, 95, 46]", executing print("FINAL(33)") scores reward 1.0,
  and a wrong FINAL scores 0.0 - confirming expected_answer reaches the server
  rubric through the WS reset.

Smoke accuracy is 0.0 for both, as expected: a 3-step LoRA cannot solve the tasks. The
smoke checks the machinery, not learning.

Smoke tuning needed to fit the L4 (all in `apply_smoke_overrides`): the old hard
512-token cap rejected finqa's ~700-token prompt; the full 4096 OOMs the backward
under vLLM colocate. Settled on a 2048 context, vLLM `gpu_memory_utilization` 0.45,
eval `max_new_tokens` 256, and 4 eval episodes (a multi-turn eval runs many
generations). finqa tool results are also truncated to ~1200 chars in the adapter
so a verbose table dump cannot push a turn past the context.

## Open items for review

- The smoke runs at a 2048 context to fit the L4; real runs use the config's 4096.
  Watch that finqa episodes finish before the cap on real runs - a long table
  exploration plus tool results can approach the budget even at 4096.
- The finqa adapter reads `finqa.csv` directly. If the server's question order ever
  diverges from a plain `pd.read_csv` of that file, display and scoring would drift.
  They agree today because both read the same file in file order.
- repl's task family is arithmetic only. It validates the harness; it is not a
  research-grade task set. Upgrade `tasks.py` before drawing conclusions.
- repl's sandbox has no wall-clock timeout (only operation/loop caps), so a
  pathological generated snippet can stall a rollout slot.
- The two box-side prerequisites are not yet in `setup.sh`: the env deps
  (`uv pip install smolagents gradio pypdf`) and the finqa data download plus
  `python patch_openenv.py`. They are documented here; folding them into setup is a
  follow-up.
