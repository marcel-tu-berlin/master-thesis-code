# Probes

Standalone diagnostic scripts. Not part of the pipeline: they run outside
`training/` and `eval/`, answer one question each, and are kept because the
question recurs.

## `bg_probe.py` - BrowserGym/MiniWoB difficulty probe

Measures base-model success per MiniWoB task so an environment can be checked
against the two bars before anything is trained: base accuracy in 40-80% (so GRPO
gets within-group variance and the task-success baseline has headroom) and a
success/termination gap (so the off-target axis is not collinear with task
performance). Mirrors the agentic eval loop - native tool calling, greedy,
stop-reason bookkeeping - so the numbers transfer.

Results and interpretation: `pipeline/runs/browsergym_feasibility_findings.md`.

Runs in its own venv on the box, not the pipeline venv:

```bash
# MiniWoB HTML must be served first; browsergym-miniwob does not ship it.
cd /workspace/miniwob-plusplus/miniwob/html && python3 -m http.server 8080 &
/workspace/bgym-venv/bin/python bg_probe.py
```

`MAX_NEW` is the per-turn budget and is load-bearing. At 512 the probe reported
click-option at 0.10 with 17 of 20 episodes truncated; at 2048 the same task gives
0.50. Qwen3's thinking block costs about 300 tokens per turn and accumulates across
turns, so a budget that fits turn 1 will not fit turn 3.

## `env_check.py` - post-update env verification

Run after updating the OpenEnv clone. Starts each integrated env and does a real
`reset` over HTTP.

An import check is not sufficient: a core/env version skew imports cleanly and
fails at the first `reset` with
`TypeError: StepResult.__init__() got an unexpected keyword argument 'metadata'`.

Every env's `server/app.py` hardcodes port 8000 with no PORT variable, so servers
started back to back silently re-check whichever one already holds the port. This
script imports each `app` object and serves it on its own port instead of calling
`main()`.

```bash
/workspace/master-thesis-code/.venv/bin/python env_check.py
```
