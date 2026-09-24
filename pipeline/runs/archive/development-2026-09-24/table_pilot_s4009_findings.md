# Seed-4009 table pilot findings

## E0 base-policy reference

`table-pilot-e0-s4009` completed its declared 100 greedy `development_table`
episodes at seeds 4009300000 through 4009300099. Both phase exits returned 0.
The frozen base evaluation had no checkpoint, checkpoint step 0, and was not a
smoke run. All six canonical files match their remote SHA256 values. No
`checkpoint_origin.json` was present, and no adapter checkpoint was harvested.

The visible-table audit passed for all 100 episodes. It checked observations,
submitted values, terminal rewards, action accounting, native action errors,
and stops. All 41 environment-terminal trajectories had values and rewards
consistent with the visible table; no transport or post-terminal action defect
appeared. The metrics recomputed from the stored samples, the reference
thresholds were finite, and the evaluation log had no traceback, CUDA OOM, or
connection-reset signature.

E0 recorded 24/100 correct episodes (Wilson 95%: 16.7%-33.2%), with 41
`env_done`, 51 `hit_generation_cap`, and 8 `no_tool_call` stops. This is the
paired base reference only. The 30% sampled-screen floor does not apply to this
greedy E0 evaluation. The existing qualified source digest
`89c6c77e1e7d643fd44495dda4c3395342785afb8eaaf61977175ae134dc14ec` and
stack digest
`74570e8f708dece02bd348b714ea0db0bb1b063ab84b5ed44169da3a74697245` remain
bound. The source-bound E0 gate passed, admitting the declared 300-update E1
control pilot.

## Final pilot review, 2026-09-20

**Verdict: technically valid, but the family fails the declared final-E1
success-headroom gate.** Compression opportunity remains; E3 is only a warning.
No extension or shaped run is admitted under the current protocol. Step 300
remains the endpoint; earlier checkpoints are descriptive observations.

Training completed all 300 updates in 28.46 hours. The three scheduled
evaluations completed at 2026-09-20 01:51:31 UTC. Training, evaluation and
controller exit codes were zero. Review evidence and its runnable offline check
are in `table-pilot-s4009-ops/final-review/`.

| Policy | Correct / 100 | Wilson 95% success interval | Mean model tokens, correct episodes | Budget endings / 100 |
|---|---:|---:|---:|---:|
| E0 | 24 | 16.69-33.23% | 1596.58 | 51 |
| E1, step 100 | 93 | 86.25-96.57% | 1320.19 | 5 |
| E1, step 200 | 100 | 96.30-100% | 664.92 | 0 |
| E1, step 300 | 100 | 96.30-100% | 580.87 | 0 |

All policies answered the same 100 development questions at seeds
4009300000-4009300099, disjoint from the declared 500-question training pool.
Their initial observations match exactly. The paired E0/final-E1 success table
contains 24 both-correct and 76 wrong-to-correct cases, with no regressions.
On the 24 both-correct questions, mean model tokens fell from 1596.58 to 568.75;
the median paired E1-minus-E0 difference is -905 tokens. The table's changing
correct subsets must not be treated as a paired token comparison. These are
task-only learning results, not an E2 treatment effect or multi-seed evidence.

### Technical and trajectory evidence

All 31 harvested run files match remote SHA256 hashes. The qualified source
digest and stack match admission, including training and every evaluation.
The frozen E0 reference and evaluation protocol match their admitted inputs.
The saved checkpoints identify steps 100/200/300, and checkpoint-final has the
same adapter hash as checkpoint-300. All 300 logged updates are finite; no
traceback, CUDA OOM or connection-reset signature appears in either phase log.
The full local project check also passed after review.

The audit recomputed report metrics with E0's frozen thresholds and checked all
400 evaluation trajectories against the visible table, submitted fields,
terminal rewards, action counts and stop reasons. It also checked all 416
captured training trajectories across the thirteen declared batches. No missing
observation, reward mismatch or post-terminal generation/action was found.
Every final greedy episode used two correct fills followed by submit.

The seed-17 stratified review selected six cases each for budget endings,
voluntary stops, invalid/repeated actions and efficient correct controls from
E0 and all three checkpoints, assigning each to its first eligible stratum.
There were no shortages. Policy/checkpoint and seed/index labels were masked
for case inspection, then revealed to verify the judgments. The reviewer
already knew the aggregate outcomes; this is not an independent blinded study.
All 24 cases support their assigned behavior. Short controls solve the task;
they must not be labeled harmful merely because their reasoning is short.

### Core criteria and secondary warning

Success headroom fails unambiguously: the final Wilson interval is wholly
above 90%. The single paired extension is reserved for an interval spanning
90%, so this result does not admit more evaluation. Late sampled training
also saturates: 314/320 correct trajectories, five mixed-success groups out
of forty, 87.5% zero-task-reward-variance groups, and six of the last ten
updates with zero logged gradient norm.

E2 opportunity survives. All forty late groups contain at least two correct
solutions whose length range exceeds 10% of the within-group correct median.
Each group's shortest correct solution uses at least 29.9% fewer model tokens
than its longest. For example, the same question at update 291 has valid
276-token and 549-token solutions. The median within-group correct cosine
reward range is 0.0201. These are sampled alternatives, not an estimate of
achievable E2 improvement; no shaped policy has been trained. Added length
reward would also activate groups where E1's task signal is constant, so any
later contrast must acknowledge the difference in active learning signal.

Budget exhaustion is 0/100 at the endpoint (Wilson 95%: 0-3.70%), with no late
within-group budget-cost variation. This is the secondary E3 warning under
decision 0017, not the reason for the failed core verdict.

The frozen-threshold underthinking proxy flags 99% of final correct episodes.
Their grounded, valid three-action solutions show why this threshold flag is
not itself evidence of harmful underthinking. Preserve the metric and clarify
its interpretation when freezing the campaign; do not silently redefine it.

### Decision now required

The setup demonstrably learns read-table-2, but the 300-update control exhausts
its measured success headroom. Keep this as a valid development finding and
potential compression/control task. The current joint research protocol needs
a harder qualified family. Alternatively, a narrower read-table-2 E2 study
could test further compression while preserving success, but that requires an
explicit amendment to the success-headroom requirement and bounded claims.
Neither path is launched before the user's choice. Final-test data remain unused.

The automatic watcher did not start this review: its SSH subprocess repeatedly
exited 255. Harvest and review were completed in the active session after SSH
became reachable. The GPU controller had independently finished normally.
