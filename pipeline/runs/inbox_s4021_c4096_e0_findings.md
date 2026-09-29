# Corrected mixed-inbox E0, seed 4021, 4,096-token allowance

Run: `e0-email-inbox-noscroll-s4021-c4096`. Completed 2026-09-29 at
23:44:24 UTC; reviewed 2026-09-30 local time. Integrity review passed, and the
unchanged E1 task-only arm is admitted. This is the corrected base-model
reference, with no adapter. The superseded 5,120-token E0 is not reused or pooled.

## Results

| Measure | Result |
| --- | --- |
| Correct | 103/200, 51.5% (Wilson 95% CI 44.6-58.3%) |
| Assistant tokens on correct episodes | Mean 1,338.0 (bootstrap 95% CI 1,204.1-1,474.2) |
| Correct terminal completions | 103 |
| Incorrect terminal completions | 23 |
| Generation-budget endings | 67 |
| Stopped emitting tool calls before completion | 7 |
| Turn-limit endings | 0 |
| Episodes with at least one action error | 124/200 |
| Allocated GPU time | 2.7147 hours on physical GPU 1 |

| Requested operation | Correct | Rate | Incorrect completion | Budget ending | No tool call |
| --- | --- | --- | --- | --- | --- |
| Reply | 3/54 | 5.6% | 5 | 41 | 5 |
| Forward | 19/60 | 31.7% | 16 | 23 | 2 |
| Delete | 40/41 | 97.6% | 1 | 0 | 0 |
| Mark important | 41/45 | 91.1% | 1 | 3 | 0 |

The pooled base success is within the earlier preferred range, while simpler
operations are near ceiling and multi-page operations remain difficult. This
heterogeneity is already part of the accepted mixed protocol. These fixed 200
greedy episodes are not an estimate under the qualification screen's sampled
training protocol. Do not change the task mix or budget after these outcomes.

## Trace findings

All 23 incorrect terminal outcomes match the native scorer and saved visible
states. Four replies send empty text, and one omits the requested final period.
Fourteen forwards invent address strings where the task requires an exact
recipient name; two leave the recipient empty. One deletion and one star action
target the wrong sender. These simulator outcomes do not represent real delivery.
The native forward scorer checks recipient and body, without separately checking
source sender; every terminal forward here also used the requested source sender.

The efficient control rule is the shortest correct episode of each operation,
with ties broken by episode index. The selected zero-based indices are 132
(reply, 903 assistant tokens), 184 (forward, 1,480), 176 (delete, 269) and 135
(important, 279). Episode 132 recovers from an invalid fill, opens the reply form,
enters the exact requested text and sends correctly. Episode 184 batches the
recipient fill and send after opening the correct form. Episodes 176 and 135
complete directly from the inbox. These are valid efficient paths.

Action errors occur in 60 correct episodes and 64 incorrect episodes. An error
alone therefore does not establish failed recovery. All 74 nonterminal endings
are separately recorded as budget exhaustion (67) or no tool call (7); they are
not conflated with incorrect terminal sends. Brevity, batching and low generic
verification depth alone are not labels of harmful substitution.

## Integrity and admission

- All 200 episode indices and seeds are exactly the planned sequence,
  4021100000-4021100199. Every report sample matches its saved episode, and
  every reported metric was recomputed with the existing implementation.
- Source/config/launcher hashes and stack match the frozen manifest. Both
  environment clones are clean and pinned. The worker exited zero, both worker
  and controller exited, and GPU 1 was released. All harvested bytes match the
  completed remote artifacts. E0 produces no trained checkpoints.
- The saved-state audit covers all 200 trajectories, reconstructs terminal
  outcomes, checks action/error/repetition counts and assistant-token sums, and
  rejects deliberately corrupted outcome labels for all four operations.
- The pinned tokenizer separately recounts feedback charges and verifies all
  200 stopping conditions against the original 4,096-token whole-trajectory
  allowance. Context remains 8,192, prompt allowance 4,096 and turns eight.
  Tool observations consume the trajectory allowance but not the length reward.
- One complete cost attempt records 9,772.870 seconds and 2.714686 GPU-hours;
  episode time is 9,761.106 seconds, including 8,820.036 inference seconds.
- The immutable E0 length thresholds are 544.3 and 2,475.0 assistant tokens.
  They are descriptive cutoffs, not validated underthinking/overthinking labels.

Aggregate outcomes were known during review; it was not blinded. The review
includes every incorrect terminal outcome and the declared efficient controls.
No new integrity or scientific issue changes the approved protocol. E0 cannot
establish learning, sustained headroom, compression or reward-bias substitution.
E1 proceeds from the original base with unchanged 300-update, 4-by-8 geometry and
budgets. E2 remains gated on E1 review; stop after the corrected trio's review.
The retained read-table evidence remains unchanged and separate.

Reproducible audit scripts are in `inbox-campaign-s4021-c4096-ops/`:
`review_e0.py` and `audit_e0_budget.py`. Raw traces and hash-bound review evidence
are in `e0-email-inbox-noscroll-s4021-c4096/`, including
`technical_review.json`, `budget_review.json` and `review.json`.
