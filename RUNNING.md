# Running

What is executing on the GPU box (`ssh gpu-l4`). Rows leave the table once the
results are harvested. Update rules are in CLAUDE.md.

Updated: 2026-08-01 17:30 UTC (box time)

Nothing running. GPU free.

## Harvested today

**BrowserGym/MiniWoB feasibility probe** - done, all 8 tasks. BrowserGym passes:
click-option 0.50 and enter-text 0.60 sit in the 40-80% band, click-checkboxes is
the one task where success (0.35) and termination (0.50) separate. Findings in
`pipeline/runs/browsergym_feasibility_findings.md`, probe kept at
`pipeline/probes/bg_probe.py`.

**e26 finqa E1 qualification** - done. finqa disqualified (0/60 eval, 1%
per-rollout training accuracy, gradient on 10 of 300 steps). Findings in
`pipeline/runs/e26_finqa_qualification_findings.md`.

**e24/e25 4k pair** - done. Reports and train logs in
`pipeline/runs/e24-poly-4k-env-only-qwen3-1_7b/` and
`pipeline/runs/e25-poly-4k-cosine-w16-qwen3-1_7b/`; findings in
`pipeline/runs/e24_e25_4k_pair_findings.md`; batch summary in
`pipeline/runs/batch_summary_20260730_042707.md`.

## Box state changed today - read before the next run

- **OpenEnv clone updated** `d372fab` -> `024eedc` (was 1695 commits behind).
  Rollback point is `d372fab`. Our finqa patches were stashed across the pull and
  reapplied cleanly; they are also saved in `openenv-patches/`.
- **The update broke the pipeline; the fix is applied.** The repo is now
  `openenv 0.4.2.dev0` and its env clients pass `metadata=` to `StepResult`, which
  `openenv-core==0.3.0` rejects with a `TypeError`. This hit reasoning_gym, not
  just browsergym. An import check does NOT catch it - the server module imports
  fine and the client fails at the first reset. Fix: `openenv-core` uninstalled
  from the pipeline venv (it shadows the `openenv` import name) and the repo
  installed editable. `setup.sh` now does this too. All four integrated envs
  re-verified by real HTTP reset via `pipeline/probes/env_check.py`.
- **browsergym is installed and working** in its own venv `/workspace/bgym-venv`
  (editable repo install, `openenv-core` removed). Server deps: browsergym-core,
  browsergym-miniwob, playwright 1.44 + chromium, torch + transformers for the
  probe. `playwright install-deps` fails on Ubuntu 24.04 - playwright 1.44 asks
  for `libasound2`, renamed by the t64 transition - so the libs were installed by
  hand.
- **MiniWoB HTML is not shipped with `browsergym-miniwob`.** Cloned to
  `/workspace/miniwob-plusplus` (130 task files), served by a detached
  `python -m http.server 8080` on `/workspace/miniwob-plusplus/miniwob/html` with
  `MINIWOB_URL=http://localhost:8080/miniwob/`. That static server must be up
  before any browsergym server starts, and it does not survive a box restart.

## Two traps that cost time today, both silent

- Every env's `server/app.py` hardcodes port 8000 with no PORT variable, so
  servers started back to back silently re-check whichever one already holds the
  port. `pipeline/probes/env_check.py` imports each `app` and serves it on its own
  port instead of calling `main()`.
- `pkill -f <pattern>` over ssh matches the remote shell running the command when
  the pattern appears in that command line, killing the session before the real
  work runs.
