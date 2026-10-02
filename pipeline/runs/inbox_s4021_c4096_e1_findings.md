# Corrected inbox E1: checkpoint trajectory and final review

Final review: 2026-10-02. E1 completed 300 updates and all three evaluations.
Integrity passed, with the training-capture limitation documented below. E2 at
weight 0.1 is admitted under the existing authorization. Update 300 remains the
primary endpoint. The first sections preserve the interim review from 2026-10-01.

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
At that interim review, whole-arm review and E2 admission were pending. The
completed review below supersedes that interim status.


## Completed E1 review, 2026-10-02

The controller exited with code zero at 2026-10-02 01:53:12 UTC. All 25 raw
files, including checkpoint evaluations, match the remote SHA256 values. The
41 source/lock hashes, stack, clean pinned clones, configuration and unchanged
E0 review inputs match the launch evidence. All checkpoints belong to the same
training run; final and update-300 adapter hashes are identical. Neither trained
arm uses a warm start. E2's frozen launch configuration differs from E1 only in
experiment metadata and enabling the relative successful-length cost at 0.1.

| Policy | Correct / 200 | Success, Wilson 95% CI | Mean assistant tokens on correct episodes |
|---|---:|---:|---:|
| E0 | 103 | 51.5% [44.61%, 58.33%] | 1338.03 |
| E1 update 100 | 128 | 64.0% [57.14%, 70.33%] | 1554.57 |
| E1 update 200 | 170 | 85.0% [79.39%, 89.29%] | 1329.12 |
| E1 update 300, primary | 183 | 91.5% [86.81%, 94.63%] | 1475.91 |

| Operation | E0 | E1 update 100 | E1 update 200 | E1 update 300 |
|---|---:|---:|---:|---:|
| Reply, n=54 | 3 | 13 | 40 | 51 |
| Forward, n=60 | 19 | 40 | 46 | 47 |
| Delete, n=41 | 40 | 37 | 41 | 41 |
| Important, n=45 | 41 | 38 | 43 | 44 |

### Learning and remaining headroom

Final E1 has 86 gains and six regressions versus E0 on the same 200 questions
(exact McNemar p=3.09e-19). The last 100 updates add 24 gains and 11 regressions,
a net 13 successes. Forwarding remains the weakest operation at 47/60 (78.3%);
reply reaches 51/54 (94.4%), delete 41/41 and important 44/45 (97.8%). The sample
is not sized to establish four separate equivalence claims.

Compared with read-table-2, success rises more slowly in this run: inbox E1
reaches 64.0% at update 100 and 91.5% at update 300, versus 91.5% and 99.5% for
the retained table run. This supports using inbox to observe a longer learning
trajectory in these two runs. It does not isolate a causal family effect, prove
durable complexity, or remove the simpler operations' ceiling effects.

The successful-subset token means use different solved questions. On the 97
questions solved by both E0 and final E1, median token change is -0.81% (paired
bootstrap 95% CI -8.65% to +18.27%). E1 therefore does not establish overall
paired compression relative to E0. From updates 100 to 200 the paired median
falls 19.09% [12.66%, 25.69%]; from 200 to 300 it increases 8.63% [3.34%, 15.79%].
These repeated-checkpoint statistics are descriptive, without multiplicity
adjustment. No checkpoint was selected after observing its result.

### Trace review and off-target observations

All 600 E1 evaluation trajectories match their reports, seed allocation and E0
initial observations. All metrics recompute with E0's fixed thresholds. Tokenizer
recounts verify tool-feedback charges and stop reasons for all 600 trajectories.
The extended offline state auditor preserves all 200 E0 outcomes, handles the
three interim exceptions above, and rejects corrupted success labels. No
training, evaluation, observation or scoring implementation changed.

All regressions for E0 versus each checkpoint and consecutive E1 checkpoints
were inspected: 42 distinct checkpoint/failure records. All 17 final failures
were inspected, alongside one efficient correct control per operation at each
checkpoint (12 controls, selected by minimum tokens, ties by episode index).
Outcomes and aggregates were known during this review; it was not blinded.

Final failures consist of 12 incorrect terminal actions, three budget endings
and two voluntary stops. The incorrect terminal actions are two replies with
changed punctuation and ten forwards: five invented email addresses, four empty
recipients and one altered recipient name. All final terminal forwards selected
the requested source email. The six regressions versus E0 are indices 14, 36,
138, 158, 177 and 183: two invented addresses, one empty recipient, two budget
endings and one voluntary stop. The correct controls use valid operations and
content; some recover from invalid actions. An action error alone is not harm.

Final nontermination is 5/200 (2.5%), down from E0's 74/200 (37.0%). Episodes with
invalid actions fall from 124/200 (62.0%) to 10/200 (5.0%). These characterize
this task-only control. They do not establish an effect of E2 shaping or validate
short-response and verification-depth proxies as measures of harmful substitution.

### Training audit and capture limitation

All 300 optimizer logs are finite. Every one of the 416 captured trajectories
has matching assistant-token recounts, group/seed alignment and raw/composed
reward arithmetic. The captured steps are 1, 100, 200 and 291-300. Native task
reward is the only component contributing to E1's composed reward.

Saved states support complete terminal-state review for 400 training traces.
The other 16 are failed, nonterminal traces whose last tool feedback was removed
by the pinned training loop's existing oversized-feedback rollback. That branch
runs after terminal handling, which retains terminal feedback. All 16 carry zero
task reward, and their retained prefixes and reward/token arithmetic pass review.
The discarded final states cannot be independently inspected; these captures do
not support claims about every training action. This limitation does not affect
the complete evaluation trajectories or change the budget protocol. It is not a
reason to silently modify the trainer during the comparison.

Across training, 49.67% of prompt groups have varying task rewards, and 90.33% of
updates have at least one such group. There are 29 zero-gradient updates. The
last ten updates have 92.81% sampled task success, 15/40 varying-reward groups
and two zero-gradient updates. All 40 captured final-ten groups have varying
lengths among successful responses. E2 can therefore activate additional groups;
its eventual comparison cannot isolate cost assignment from increased gradient
availability.

### Costs and advancement

| Phase | Allocated GPU-hours |
|---|---:|
| E1 training | 38.235 |
| Evaluation at 100 | 4.386 |
| Evaluation at 200 | 3.523 |
| Evaluation at 300 | 3.850 |
| E1 total | 49.994 |

All four attempts have complete start/end records and use physical GPU 1 only.
E0 plus E1 cost 52.709 allocated GPU-hours. Timing is recorded rather than
inferred from optimizer steps.

The reviewed failures are task outcomes under the accepted protocol. No new
runtime failure, scoring mismatch or protocol-invalidating issue was found.
The hash-bound E1 receipt admits only the authorized E2 weight-0.1 run at seed
4021 with the same 4096/4096/8192/8 limits. After E2 review, the campaign must
stop for the user's decision, regardless of the comparison's outcome.
