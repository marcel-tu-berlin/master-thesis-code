# Standing rule: truncation is a reported outcome, not a confound to remove

Decided: 2026-08-07. Status: accepted. Moved verbatim from `LAB_NOTES.md` on 2026-09-04.


The 4096-token completion cap is the L4's hardware ceiling, not a design choice
(the memory arithmetic is in `e24bs4`'s config description; Liger is the only
route past it and is unverified, BACKLOG item 1). So some episodes will always end
by filling the budget rather than by finishing, and the decision as of 2026-08-07
is to measure that rather than engineer around it: every accuracy is reported
with its `stop_reason` breakdown beside it, and truncation becomes a signal of
the experiment rather than noise inside the accuracy number.

**Whether the cap corrupts accuracy is a per-environment fact, so check it per
environment.** Measured from the reports on disk, wrong episodes by `stop_reason`:

```
poly       e24bs4 agentic   env_done 86 | cap 14                        wrong: 14 cap,  0 terminated
           e25bs4 agentic   env_done 91 | cap  9                        wrong:  9 cap,  0 terminated
browsergym e27    held_out  env_done 85 | cap 4  turns 3  no_tool 8     wrong: 20 terminated, 15 other
           e27    shifted   env_done 82 | cap 0  turns 1  no_tool 17    wrong:  0 cap,  18 other
           e0     held_out  env_done 86 | cap 0            no_tool 14   wrong: 11 terminated, 14 no_tool
           e0     shifted   env_done 90 | cap 0            no_tool 10   wrong:  3 terminated, 10 no_tool
```

On polynomial_equations the two are the same measurement: every wrong episode is a
truncation and every terminated one is correct, so `accuracy == 1 - truncation
rate` to the episode. A length reward that shortens completions mechanically
lifts that accuracy, which is why the pair's accuracy gain is not independent
evidence about task success. `runs/e24bs4_e25bs4_pair_findings.md` says so at
length; do not quote a poly accuracy without it.

On browsergym they are not. Wrong-and-terminated is a populated cell (20, 11, 3),
truncation runs 0-4%, and the dominant off-target mode is `no_tool_call` at 8-17
per hundred. The live campaign's accuracy therefore carries information the cap
does not supply, and the interesting signal there is the whole breakdown, not
truncation on its own.

Already on disk in every report: `stop_reasons` as a raw per-split counter, and
`non_termination_rate` with a Wilson interval (`_offtarget_panel`,
`eval/metrics.py`). What does not exist yet is a per-reason rate with its own
interval. That is the open design step, not something the current numbers are
missing.
