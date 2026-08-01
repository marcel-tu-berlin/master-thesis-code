# BrowserGym / MiniWoB feasibility probe

Base `Qwen/Qwen3-1.7B`, no training, greedy, native tool calling, 20 episodes per
task, 2048-token per-turn budget, 10-turn cap. Probe script kept at
`pipeline/probes/bg_probe.py`. Run 2026-08-01 on the L4 box.

Purpose: decide whether BrowserGym replaces FinQA as the study environment, after
e26 disqualified FinQA (`e26_finqa_qualification_findings.md`). Two bars, both set
by that run: base accuracy in 40-80%, and at least two tools.

## Results

| Task | Success | Termination | Gap | Tokens | Steps | Stop reasons |
|---|---|---|---|---|---|---|
| click-test | 1.00 | 1.00 | 0 | 269 | 1.0 | 20 env_done |
| click-button | 1.00 | 1.00 | 0 | 226 | 1.0 | 20 env_done |
| click-dialog | 1.00 | 1.00 | 0 | 334 | 1.0 | 20 env_done |
| **enter-text** | **0.60** | 0.60 | 0 | 952 | 1.6 | 12 env_done, 8 no_tool_call |
| **click-option** | **0.50** | 0.50 | 0 | 1061 | 1.9 | 10 env_done, 9 no_tool_call, 1 cap |
| **click-checkboxes** | 0.35 | 0.50 | **0.15** | 1332 | 2.0 | 10 env_done, 10 no_tool_call |
| login-user | 0.05 | 0.10 | 0.05 | 1029 | 1.2 | 18 no_tool_call, 2 env_done |
| click-tab-2 | 0.00 | 0.20 | 0.20 | 2870 | 3.5 | 6 cap, 6 no_tool_call, 4 max_turns, 4 env_done |

MiniWoB spans saturated to unreachable for this model, and the band sits in the
middle where tasks can be selected into it. This is the property FinQA lacked: its
difficulty was fixed at 1% with no dial.

click-tab-2 is the only task that exercised all four stop reasons, including
`max_turns`. Its 0.00 success makes it unusable as a training task, but it is the
one probed task where a longer horizon and every failure mode are live at once.

## The selection criterion is not accuracy alone

On click-option and enter-text, success equals termination exactly: every episode
that terminated was correct. Those two axes are collinear there, which is fatal
for RQ2 - if reducing non-termination *is* increasing task success, a
non-termination penalty cannot be told apart from a task reward and there is no
substitution to observe. The protocol's task-performance and off-target dimensions
would be one dimension.

click-checkboxes separates them: 10 episodes terminated, 7 were correct, 3
submitted a wrong answer. Task performance and the off-target axis can move
independently.

So a task qualifies on **band membership AND a success/termination gap**. A mixed
training set is the way to get both: click-option or enter-text for headroom,
click-checkboxes for axis separation.

## Failure mode is the one the thesis targets

Across every in-band task the dominant failure is `no_tool_call` - the model stops
without acting - not truncation. enter-text records zero budget artifacts, and
click-option one out of twenty. That is premature stopping, a named behavior in the
expose's off-target panel, occurring spontaneously in the base model at a rate with
room to move in both directions. On FinQA the comparable number was 86.7%
non-termination dominated by truncation, with nothing left to shape.

## The 512-token result was an artifact, and it is worth recording

A first pass at a 512-token per-turn budget produced click-option 0.10 with 17 of
20 episodes at `hit_generation_cap`, and click-dialog 0.90. Both were budget
artifacts: at 2048 the same tasks give 0.50 and 1.00. Kept at
`/tmp/bg_probe_cap512.log` on the box.

The cause is Qwen3's thinking block. Measured on one click-option turn:

| Config | Generated | Think tokens | Tool call |
|---|---|---|---|
| thinking on (default) | 327 | 304 | `click(bid=24)` correct |
| thinking off | 24 | 0 | `click(bid=O2F2ioz)` wrong |

Thinking costs about 300 tokens per turn and does real work - without it the model
passes the visible label instead of the element id. Turn 1 fits in 512; later turns
carry every prior turn's think block plus injected tool results and do not. This is
the same mechanism behind FinQA's 77% clipped ratio, and it is the token-efficiency
tradeoff the thesis studies showing up inside the environment.

The stop-reason instrumentation is what made this visible: 9 behavioral stops and 1
truncation were separated instead of being pooled into "10 wrong answers".

## Integration notes

- `python -m browsergym_env.server.app` runs headless with no API key, our existing
  no-Docker pattern. Server up in about 4 seconds.
- The tool bridge is already upstream: `harness.py` defines five MCP `Tool` specs
  (`click(bid)`, `fill(bid,text)`, `send_keys(text)`, `scroll(direction)`, `noop`)
  plus `build_browsergym_action_str(name, args)`. The adapter wraps that rather
  than writing the translation.
- `reset(seed=N)` is deterministic, confirmed empirically: the same seed twice
  returns an identical goal and accessibility tree, a different seed returns a
  different page, and the first seed still reproduces after an intervening reset.
  GRPO group determinism comes free here, unlike FinQA which needed a patch.
- `reset(task_name=...)` switches task per reset, confirmed empirically including
  switching back and reproducing the original page. `BROWSERGYM_TASK_NAME` is only
  a default, so one server serves several task families and a shifted eval split
  is just a different `task_name`.
- Observations are small: the click-test accessibility tree is 63 characters
  against FinQA's roughly 1200-character tool results. Context pressure is not the
  binding constraint here.
- MiniWoB HTML is not shipped with `browsergym-miniwob`. Clone
  `Farama-Foundation/miniwob-plusplus`, serve `miniwob/html` over HTTP, and set
  `MINIWOB_URL`. That static server must be up before any browsergym server.
- `playwright install-deps` fails on Ubuntu 24.04 - playwright 1.44 asks for
  `libasound2`, renamed by the t64 transition. Install the libs by hand.

## Distribution shift

BrowserGym's stated pipeline is train on MiniWoB, evaluate on WebArena, which is
the held-out and shifted split structure the protocol requires, off the shelf
rather than invented. WebArena needs backend infrastructure that has not been
assessed. A cheaper shifted split available immediately is a held-out MiniWoB task
family, expressible as a different `task_name` per eval split.
