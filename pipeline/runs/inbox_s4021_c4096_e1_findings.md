# Corrected inbox E1: interim checkpoint-100 results

Reviewed on 2026-10-01. This is an interim checkpoint record, not the final E1
review or admission of E2. The primary endpoint remains update 300. The corrected
run completed all 300 training updates; its remaining evaluations continue.

## Held-out checkpoint 100

The 200 questions use seeds 4021100000-4021100199 and exactly the same initial
observations as corrected E0. All budgets remain 4,096 trajectory tokens, 4,096
prompt tokens, 8,192 context and 8 turns.

| Operation | E0 correct / questions | E1 update 100 correct / questions |
|---|---:|---:|
| Reply | 3/54 | 13/54 |
| Forward | 19/60 | 40/60 |
| Delete | 40/41 | 37/41 |
| Important | 41/45 | 38/45 |
| Total | 103/200 (51.5%) | 128/200 (64.0%) |

E1's Wilson 95% success interval is 57.14%-70.33%. The paired comparison has
43 gains and 18 regressions, a net gain of 12.5 percentage points. This is a
checkpoint observation from one training seed, not evidence of stable final
performance or an E2 shaping effect.

For the requested comparison at equal update count, retained read-table-2 E1
scored 183/200 (91.5%) at update 100, from E0's 43/200 (21.5%). Inbox starts
higher and reaches a lower success rate at this checkpoint. Different families,
seeds, prompts and observation formats prevent attributing that difference
solely to family complexity. These are held-out scores, unlike the sampled
training success percentages previously reported during the run.

## Interim integrity checks

All five harvested checkpoint-100 files match their remote hashes. The report
matches all 200 saved trajectories and recomputes exactly with corrected E0's
fixed thresholds. The 41 pinned source hashes and frozen training/checkpoint
budget settings match the campaign manifest and protocol.

The existing E0 state auditor passed 197 trajectories. Three cases were checked
manually against the saved states and pinned task source, with outcomes known:

- Episode 1 correctly starred Tiff. Visible-HTML rendering omitted the third
  email-thread wrapper while retaining its sender and action block. The saved
  post-action state marks bid 47 as clicked. E0's identical initial observation
  independently identifies Tiff by opening bid 40. The auditor's ancestry
  assumption did not cover this flattened representation.
- Episode 53 forwarded Karin instead of Kirstyn, with an empty recipient and
  the wrong body. Its failure agrees with the native scoring conditions.
- Episode 166 forwarded Jess instead of Wandis. The recipient Emeline was
  correct, but the body differed from the requested email; the failure agrees
  with the native body check.

No production code, observation renderer, scorer or frozen E0 auditor changed.
The input hashes and manual cases are recorded in
`inbox-campaign-s4021-c4096-ops/e1-checkpoint100-interim-review.json`.
Whole-arm review, including tokenizer budget recounts, training-group audits,
checkpoint provenance, costs and the remaining evaluations, is still pending.
No E1 `review.json` or E2 admission has been issued.
