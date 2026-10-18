# Flight scrolling qualification and resource assessment

Checked: 2026-10-06. **Scrolling and the nine scripted controls pass. The
measured workflows exceed the approved agent budget. Policy difficulty and
flight-specific GPU capacity remain unmeasured.** No fallback, model rollout,
training run or experimental budget change was made.

## Interface and controls

`training.env_config.enable_scroll: true` adds `scroll(bid, direction)` in both
training and evaluation. The tool hovers over the supplied element and sends a
120-pixel wheel event up or down. It uses the installed BrowserGym primitives;
the pinned OpenEnv harness's directional-only scroll signature is incompatible.
The task, scorer, visible-HTML formatter, original tool schemas and defaults
remain unchanged. Existing configs do not enable scrolling.

The native audit used the visible `#area` container. The small step retains
overlap in the 156-pixel results viewport. It accumulated price and duration
text from visible panels, identified panels by observed bids, confirmed the
list bottom with an unchanged observation after another downward scroll, and
returned to a visible booking button. It never used hidden DOM IDs, attributes
or privileged task state to choose an action. This is one inspection strategy,
not a demonstrated optimum; the fixed wheel step is part of this interface.

Three development seeds, 4022700000-4022700002, cover November, October and
December departures; shortest and cheapest objectives; and three or four
alternatives. On each seed:

- Correct visible-state selection terminated with reward 1.
- Deliberately selecting a strictly worse flight terminated with reward 0.
- Invalid origin submission remained nonterminal with zero reward and a visible
  error marker; correcting it and booking the target terminated with reward 1.
- Reset after mutation reproduced the initial observation exactly. A call after
  termination did not dispatch or overwrite the terminal reward.

All nine controls passed. This establishes useful control cases and native
reward semantics on these seeds; it is not 100% model accuracy, a representative
task distribution or evidence of trainability.

Native edge checks also passed: malformed scroll arguments return recoverable
action-error feedback without dispatch; a bid containing both quote types and a
backslash reaches a clean native element-not-found error; the corrected adapter
reproduces an entire successful path byte for byte; invalid calls after done
preserve reward. Hovering a partially visible panel works, but may reposition
the page before the wheel event. The measurements always target the container.

## Measurements

Counts use the pinned Qwen3-1.7B tokenizer, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Initial prompts with the new tool
schema are 966-968 tokens and fit the 4,096-token prompt allowance.

The scripted audit is CPU-only and unrestricted by the agent's budgets. It
measures complete paths to assess those budgets; it does not run a policy with
an altered allowance. Pre-booking feedback excludes the terminal booking
response, which need not enter a later model context.

| Seed suffix | Objective / alternatives | Correct path: actions / scrolls | Correct pre-booking feedback | Recovery pre-booking feedback | Calendar feedback on either path |
| --- | --- | ---: | ---: | ---: | ---: |
| 000 | Shortest / 3 | 18 / 9 | 13,036 | 14,682 | 6,399 |
| 001 | Shortest / 4 | 21 / 11 | 17,743 | 19,241 | 9,815 |
| 002 | Cheapest / 3 | 14 / 6 | 8,501 | 10,002 | 3,188 |

Actions are not assistant turns. A separate replay combines origin selection
with destination filling, and destination selection with opening the calendar;
both later actions use already-observed IDs. Other actions remain separate.
These declared batches produce 12-19 turns for correct paths and 16-23 for
recovery. More batching is possible, including returning several intermediate
scroll observations in one turn; these counts are not lower bounds.

The replay uses the production `_run_multiturn_episodes`, the unchanged
4,096-token/8-turn limits, and empty reasoning with template-rendered action
text. Every trace stops with `hit_generation_cap` after action 5, opening the
calendar, in assistant turn 3. This is a replay of saved scripted observations,
not a model evaluation or an independently repeated batched browser rollout.
The script asserts its generated action text fits each remaining generation
allowance. It does not bypass the evaluator's budget checks.

For the complete traces, re-rendered context through the final booking call,
excluding its terminal feedback and including empty-reasoning action text, is
9,947-19,440 tokens for correct paths and 11,593-21,081 for recovery. The largest
value across all nine is 21,081. This is a tokenizer/context measurement, not an
ordinary-training tensor capture or GPU-memory measurement.

Calendar HTML is expensive, but even deleting its feedback entirely would leave
5,313-7,928 pre-booking feedback tokens on the correct paths. Calendar-only
compression therefore cannot make these same paths fit 4,096. A broader
observation redesign or different strategy might; neither has been validated.

## Assessment and proposed next diagnostic

The family offers new testable distinctions: complete versus incomplete result
coverage, selecting the wrong optimization criterion, invalid-input recovery,
returning to a previously inspected option, and terminal wrong bookings. These
are candidate measurements of observable behavior. Establish opportunities and
denominators before using them as off-target metrics. A repeated scroll can be
necessary exploration; a short trajectory can be a valid optimization. Neither
alone demonstrates reward-bias substitution or feedback neglect.

The current useful paths are dominated by an interface/protocol cost before a
policy can compare flights. **Keep `book-flight-nodelay`; do not start E0/E1/E2
under the assumption that the present budget has qualified.** A model could use
a different route, but no observed complete route currently witnesses budget
feasibility. A larger GPU alone cannot remove the configured limits.

Recommendation: first approve a separate, bounded flight diagnostic with the
current visible-HTML interface, then profile the actual workload on physical
GPU 1. A concrete starting proposal is:

| Setting | Approved campaigns | Proposed flight diagnostic only |
| --- | ---: | ---: |
| Prompt tokens | 4,096 | 4,096 |
| Trajectory tokens, including feedback | 4,096 | 24,576 |
| Context tokens | 8,192 | 28,672 |
| Assistant turns | 8 | 32 |
| Batch size / rollouts | 4 / 8 | 4 / 8 |

The pinned model config declares 40,960 positions, so the proposed context is
within its declared positional limit. This does not establish GPU-memory fit.

The largest rendered trace uses 20,115 tokens beyond its initial prompt.
The proposed trajectory allowance adds 4,461 tokens beyond that empty-reasoning
trace; 32 turns add nine beyond the declared longest recovery replay.
**ASSUMPTION:** those margins are a reasonable starting point for qualification,
not a proven sufficient allowance for Qwen3 reasoning or harder seeds. They are
not a minimum requirement. None of these proposed values has been activated.

Such a change would create a distinct protocol. Hold it identical across any
later flight E0/E1/E2 arms, start trained arms from the original base, and keep
the table/inbox results at their original budgets. Direct cross-family length
comparisons would mix task, tool and budget differences. Decision 0025 requires
explicit approval before running the enlarged diagnostic; this report requests
no implicit launch or campaign authorization.

Next qualification should first screen the base policy on fresh declared
development seeds, with success variation and error opportunities recorded.
Then measure ordinary training generation, loss and backward peaks using the
same precision, LoRA, micro-batch, 4-by-8 geometry and pinned stack. Record
allocated, reserved and total device memory, concurrent processes, realized
sequence lengths and throughput. A short or zero-LR path cannot establish
reliable capacity for long ordinary-training trajectories.

Proposed launch scope, subject to the same approval: at most 16 greedy base
episodes under a fresh diagnostic seed block, followed only after their review
by at most three ordinary training updates from the original base on a separate
fresh block. Use fresh run IDs and stop on an integrity error, OOM or two total
allocated GPU-hours. Freeze exact IDs, seeds and configs in the launch manifest
before execution. This is a feasibility screen, not the 200-episode E0 or a
trained comparison. If it does not produce representative long trajectories,
report capacity as unresolved rather than treating short-run survival as proof.

## Hardware conclusion

The box reports two NVIDIA L4 devices with 23,034 MiB each; only physical GPU 1
is in scope for GPU work. Both reported zero used device memory in the saved
final `resource-check.json`. This audit loaded no policy and used no GPU memory,
so it establishes no flight training peak or allocation-time estimate.

The retained inbox attempt at 9,216 context tokens failed in ordinary backward
on this L4 recipe. That is a concrete reason to doubt that the proposed larger
flight protocol will fit; it does not quantify its peak or establish that a
48-GB or 80-GB device is required or sufficient. See
[the original failure](inbox_s4021_e1_failure_findings.md).

**A specific GPU-capacity request is premature.** The scientific rationale and
token evidence are now concrete; the missing step is approval and capacity
profiling of the selected protocol. If GPU 1 then fails or has inadequate
headroom, prepare the supervisor request with the native failed phase, measured
peak/throughput, requested capacity and time, and the uncertainty. Do not infer
the requirement from an OOM allocation size alone. Meanwhile, metric design and
analysis of the retained campaigns can continue. No family fallback is approved.

## Verification and artifacts

New opt-in tests failed before implementation. Invalid-argument regression tests
also failed before the correction. The repository gate passes: 492 tests,
7 skips, formatting, lint, types and all 17 setup checks. Independent review
identified the invalid-argument exception issue; the fix and follow-up review
leave no reported correctness findings in the reviewed scope.
Independent numerical review also found no material assessment errors.

All 13 files in `flight-qualification-s4022-ops/scroll-remote-manifest.json`
match their harvested SHA-256 values. Evidence includes:

- `oracle-v3/`: declared seeds/config/source hashes, nine complete event traces
  and a passed exit receipt; the failed pre-scroll attempts remain untouched.
- `scroll-edges/`: final-source hashes, native correction checks and passed
  receipt. `v3-adapter.py.txt` preserves the earlier adapter used for the nine
  paths and matches its recorded hash. Only invalid-argument handling changed
  afterward; the final adapter replayed one full correct path identically.
- `path-measurement.json`, `measure_paths.py`: per-path counts, scripted budget
  replay and input/source hashes. No model accuracy is estimated.
- `resource-check.json`: separately harvested and hash-verified final GPU
  inventory, empty compute-process list and closed qualification-server port.

OpenEnv remains at `024eedc90305cc8bd7a5b44f44d1b987102e957b`, MiniWoB at
`eb59fed60fabe8951350275ba8650633b740013b`, and BrowserGym at 0.14.3. All recorded
runtime source hashes were checked against the local files, using the retained
v3 adapter for that phase. Workers and their environment servers exited; all
outputs are harvested. No result directory, checkpoint or campaign config was
overwritten, and nothing was committed or pushed.
