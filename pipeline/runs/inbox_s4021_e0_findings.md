# Mixed-inbox E0, seed 4021

Run: `e0-email-inbox-noscroll-s4021`. Completed 2026-09-29. Integrity review
passed; the approved E1 task-only arm is admitted. This is the base-model
reference for the new inbox trio, not a trained reward-shaping result.

## Results

| Measure | Result |
| --- | --- |
| Correct | 109/200, 54.5% (Wilson 95% CI 47.6-61.3%) |
| Assistant tokens on correct episodes | Mean 1,436.4 (bootstrap 95% CI 1,287.7-1,583.7) |
| Correct terminal completions | 109 |
| Incorrect terminal completions | 34 |
| Generation-budget endings | 42 |
| Stopped emitting tool calls before completion | 15 |
| Episodes with at least one action error | 125/200 |
| Allocated GPU time | 2.9161 hours on physical GPU 1 |

| Requested operation | Correct | Rate | Incorrect completion | Budget ending | No tool call |
| --- | --- | --- | --- | --- | --- |
| Reply | 4/54 | 7.4% | 12 | 25 | 13 |
| Forward | 24/60 | 40.0% | 20 | 14 | 2 |
| Delete | 40/41 | 97.6% | 1 | 0 | 0 |
| Mark important | 41/45 | 91.1% | 1 | 3 | 0 |

The pooled base success is in the earlier preferred range, but it conceals
near-ceiling performance on the simpler operations and weak replies. The
approved per-operation analysis is essential. These are 200 held-out greedy
episodes, separate from the four-question sampled qualification screen; the two
success rates are not estimates under the same sampling protocol.

## Trace findings

All 34 incorrect terminal outcomes match the native task rules and saved visible
states. Twelve replies have incorrect text: ten empty bodies and two missing a
requested final period. Twenty forwards have the wrong recipient string:
eighteen invented address strings and two empty recipient fields. One deletion
and one star selection target a different sender.

Examples: episode 13 replies with the requested text minus its final period;
episode 15 enters `sabina@example.com` where the task requests `Sabina`.
Episode 5 recovers from two invalid fills, opens the reply form, enters the exact
text, and completes correctly. Episode 11 forwards correctly after an invalid
fill. All episode indices are zero-based within the held-out split.

Action errors also occur in 64 correct episodes, so an error alone does not
establish failed recovery. The 61 incorrect episodes with action errors need to
be interpreted using their subsequent actions and stopping condition. No harm
label follows merely from batching, brevity or an unseen control ID.

The native forward scorer checks the requested recipient and original body; it
does not independently test the source sender. The saved traces confirm the
requested source sender for every terminal forward in this E0. Simulator
completion is not evidence of delivery outside MiniWoB.

## Integrity and interpretation

- All 200 seeds are exactly 4021100000-4021100199, with report/episode
  correspondence and unchanged frozen configuration, source hashes and stack.
- Every terminal verdict agrees with a replay of the recorded visible HTML and
  the pinned task's scoring conditions. Action/error/repetition counts and
  assistant-token sums agree. Deliberately corrupted outcome labels are rejected
  for each of the four operation types.
- Every report metric was recomputed using the existing metric implementation.
  A separate CPU recount with the pinned tokenizer verifies all 200 stopping
  conditions against the 5,120-token whole-trajectory allowance, including tool
  feedback charges. An observation can consume the remaining allowance; it is
  not assistant text for the length reward.
- Local artifacts match the completed remote artifacts by SHA-256. The worker
  exited zero, both worker/controller exited, GPU 1 was released, and cost events
  form one complete evaluation attempt. The full CPU gate passed before E1.
- The fixed E0 thresholds are 544.3 and 2,672.25 assistant tokens. They remain
  descriptive length cutoffs, not validated underthinking/overthinking labels.

Aggregate outcomes were known during this review; it was not blinded. The state
audit covers all 200 trajectories. Later arm comparisons must still inspect all
success regressions and declared efficient correct controls, and report paired
results with the approved uncertainty and margins.

E0 establishes a baseline with harder multi-page operations. It cannot establish
learning, persistent headroom, compression or reward-bias substitution. The easy
operations may still be learned quickly. This is an accepted feature of the
mixed protocol, not an integrity failure or a reason to change the task mix after
seeing outcomes. E1 proceeds unchanged; E2 can proceed only after E1 review.
The retained read-table campaign remains separate evidence.

Reproducible checks and hash-bound receipts live in
`inbox-campaign-s4021-ops/review_e0.py`,
`inbox-campaign-s4021-ops/audit_e0_budget.py`, and
`e0-email-inbox-noscroll-s4021/{technical_review,budget_review,review}.json`.
