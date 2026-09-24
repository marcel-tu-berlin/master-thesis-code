# Gate 4 E1 - completion and feasibility review

Checked and harvested: 2026-09-16. Run: `gate4-e1-calibration-s4002`.
Status: execution and trajectory review complete; Gate 4 fails. The confirmed
training/evaluation termination discrepancy is corrected and requalified under
decision 0014; see [termination readiness review](termination_readiness_findings.md).
This pilot retains its original episode protocol and failed family verdict.

## Result

The 300-update pilot and all three scheduled evaluations completed. The final
menu policy solved 100/100 calibration instances and exhausted its budget on
0/100. The Wilson 95% intervals are 96.30%-100% success and 0%-3.70% budget
exhaustion. These decisively fail the declared below-90% success criterion and
at-least-10% E3 opportunity criterion. No calibration extension is indicated.
An earlier checkpoint cannot replace the predeclared final endpoint.

This establishes that click-menu-2 does not qualify for the planned combined
campaign under this recipe. Saturation alone is not a setup defect; the separate
termination discrepancy below affects the length protocol. An explicitly
redesigned E2-only study remains possible. The accepted next option is the
bounded harder-family search in `docs/plans/environment-options.md`. The pilot
review and launch-safety fixes are complete; model-free family checks may proceed.

## Checkpoint observations

All checkpoints use the same calibration instances and E0 reference thresholds.
These are observations of one training seed, not independent replications.

| Policy | Menu success | Menu budget endings | Mean model tokens, correct menu episodes |
|---|---:|---:|---:|
| E0 | 67/100 | 4/100 | 1,840.9 |
| E1 step 100 | 85/100 | 15/100 | 1,688.3 |
| E1 step 200 | 100/100 | 0/100 | 985.7 |
| E1 step 300 | 100/100 | 0/100 | 784.1 |

The conditional token means above describe each policy's solved episodes;
they are not a paired effect estimate. On the 67 menu instances both E0 and
final E1 solved, the median paired change is -1,198 model tokens, with a
paired bootstrap 95% interval of [-1,198, -1,160]. Final E1 gains 33 successes
and loses none on the 100 paired instances. This is task-only training, so
its shortening must not be attributed to a length reward.

| Split | E0 | Step 100 | Step 200 | Step 300 |
|---|---:|---:|---:|---:|
| Dialog | 18/20 | 18/20 | 14/20 | 12/20 |
| Tree | 12/20 | 13/20 | 15/20 | 15/20 |
| Transfer | 18/20 | 19/20 | 20/20 | 20/20 |

Dialog loses six previously solved instances and gains none by the final
checkpoint. This is an observed transfer regression on a small calibration
sample, not a replicated effect or evidence about shaping. Tree has two final
budget endings; dialog and transfer have none. Lower success on another
family alone is insufficient to qualify it for E3.

## Late sampled training evidence

All 13 declared batches are present: updates 1, 100, 200 and 291-300, with
32 records each. The late ten batches contain 40 seed-aligned groups and 320
trajectories. Of these, 317 succeeded and none received a budget penalty.
Only 3/40 groups retain task-reward variation; all 40 have cosine-reward
variation among correct rollouts and at least a 10% spread between their
shortest and longest correct lengths. Seven of the last ten optimizer updates
have zero recorded gradient norm.

Thus E3's target disappeared on both the sampled late-training evidence and
the greedy menu calibration. E2 retains candidate length opportunity, including
before task completion, but this is not a causal estimate of what shaping would
achieve. Its training/evaluation trajectory definition needs resolution below.

## Completion and provenance checks

- Training records all 300 updates with finite losses, gradient norms and
  learning rates; recorded training runtime is 33.17 hours.
- Training finished around September 15 23:09 UTC. Final evaluation report
  publication was September 16 05:37 UTC, or 07:37 Berlin time.
- All 480 scheduled evaluation records are present. Report counts match the
  episode files; every seed matches the declared range and E0 pair. Reserved
  final-test ranges were not used.
- Frozen launch and evaluation configs agree. The scheduled reference is
  identical to `reference_at_launch.json` and the accepted E0 report.
- The approved deployed source manifest still matches all 39 entries.
  Training and evaluation record the same pinned OpenEnv and package stack.
- Reports, logs, observations and episode records were harvested without
  adapters. Separate checksum dry runs for the main run and checkpoint-eval
  subtree reported no remote/local differences.

Machine-readable counts, intervals and paired statistics are retained in
the run's `completion_review.json`; paired calculations use `eval.paired`.

## Blinded trajectory review

The seed-17 sample followed the declared disjoint-stratum priority across the
640 E0 and E1 checkpoint calibration trajectories. Available counts were 24
budget endings, 38 voluntary stops, zero remaining invalid/repeated-action cases,
and 141 efficient correct cases. The independent invalid/repeated-action stratum
was empty after the earlier strata took priority, so its six-case shortage was
reported without replacement. Eighteen cases were reviewed, six per remaining
stratum. Policy, checkpoint and instance identifiers were hidden until labels
were written. The sampler metadata, source hashes, blinded cases, labels and
unblinding key are in `gate4-e1-calibration-s4002/trajectory-audit/`.

- Six budget cases were actual exhaustion: five reached the generation cap
  before a required next action; one spent all eight turns on tree noops.
- Six voluntary stops were premature abandonment, with unexplored menu/tree
  choices or repeated noops. They were not budget endings.
- All six short correct controls followed valid goal-directed paths. Four
  selected a menu target; one clicked the requested OK button; one submitted
  without selecting anything, exactly as its goal requested.

The last two controls expose false positives in the `unsupported_claim_rate`
proxy: a terminal first action can be the complete legitimate solution. These
two of six stratified controls are examples, not an estimate of population
prevalence. Similarly, action count minus one is interaction depth, not validated
verification. Keep the recorded metrics, but do not interpret either as harmful
behavior without task-grounded labels. No independent invalid-action stratum was
available, so the audit does not validate that detector. This was a single
reviewer audit, not an inter-rater reliability study.

## Confirmed termination discrepancy

The installed TRL 1.6.0 tool loop continues generation after an environment
reports `done`; it only stops on no further tool call, its iteration limit or
the generation cap. The adapter preserves the terminal reward and rejects
subsequent actions, but the model still generates their arguments, reasoning
and final replies. Evaluation instead stops immediately on environment `done`.
The installed source and captured completions both confirm this difference.

In the late sample, all 319 completed trajectories contain later assistant
tokens; 158 include rejected post-completion tool calls. The production tokenizer
and `model_token_count` count 144,057 post-completion tokens out of 384,315
assistant tokens in completed episodes, or 37.48%. The median per-episode
fraction is 29.99%. The boundary is the last real tool result before the
done-guard responses; tokens within an assistant message that contains both a
terminal action and later calls cannot be separated by this analysis.

All 40 groups still have at least 10% correct-length spread before that boundary.
Two inspected short/long pairs solve the same menu goals through the same valid
actions with different reasoning lengths. Candidate E2 opportunity therefore
survives removing the tails, but the existing full-trajectory length signal
includes behavior that the evaluation loop never measures.

This does not erase the observed calibration successes or change the failed
Gate 4 decision. It does block treating earlier runtime/replay passes as proof
of a common episode boundary. Any termination correction changes the training
trajectory distribution and reward inputs: previous captures and this pilot
remain evidence for the old protocol, not validation of the corrected one.
Preserve them and repeat affected readiness checks before model sampling or
training. No historical tokens, reports, rewards or checkpoints were rewritten.
The user accepted alignment at environment completion in decision 0014.
Requalification is pending; model-free oracle checks are independent.

## Monitoring

The two scheduled local checks did not run: their wrappers stopped with
`MODEL_SUBAGENT: parameter not set`. This affected monitoring only; the GPU
worker and its scheduled evaluation continued independently. This manual
check recovered and verified the results. Future check wrappers need their
model configuration validated before scheduling.
