# Five matched 0.1 placebos: combined findings and next steps

The five authorized placebo runs completed: table, inbox and tabs on L4 GPU 1;
checkboxes and tic-tac-toe on the auxiliary home RTX 4090. Each has 300 training
updates and three 200-question evaluations. The last cell, tabs, finished on
2026-10-24 at 01:40 UTC. Its artifacts are now harvested and hash-verified;
L4 GPU 1 and the original tabs processes were verified idle/exited. Home was
already retired and its dedicated HTTP service stopped at the user's request.

Outcomes differ across families, each with one seed and some with host
differences. In these runs, E2 is much shorter than its matched placebo on
table and inbox; inbox also shows adverse tool behavior. On checkboxes,
large compression accompanies lower task quality.
Tic-tac-toe does not establish the joint success-preservation/compression claim.
Tabs retains a severe E2 collapse relative to both placebo and task-only E1.
The placebo controls do not justify a general claim that length shaping yields
safe efficiency gains, or that every extra variable reward produces the same
effect.

## Primary final comparisons

All cells use one training seed. Counts are native successes out of 200.
Token changes compare E2 with placebo only on their joint successes; negative
means shorter E2 responses. The full reports carry intervals and coverage.

| Family | E1 | Placebo | E2 | Joint successes | Median E2 token change | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Table | 199 | 189 | 198 | 188 | -79.64% | Preservation, compression and all four generic behavior bounds pass |
| Inbox | 183 | 196 | 199 | 195 | -83.93% | Preservation and compression pass; invalid/repeated-action bounds fail |
| Checkboxes | 191 | 183 | 162 | 152 | -81.01% | Compression passes; success preservation and wrong-submission/no-tool bounds fail; invalid/repeat bounds pass |
| Tic-tac-toe | 85 | 58 | 66 | 32 | -6.28% | Preservation and 10% compression not established; only voluntary-stop behavior bound passes |
| Tabs | 189 | 124 | 5 | 5 | +85.09% | Severe success regression; compression and all four behavior bounds not established |

Checkbox native success is a permissive threshold: exact completions are
139/156/62 for E1/placebo/E2. E2 therefore loses considerably more exact task
quality than native success alone shows. Tic-tac-toe's accepted batch boundary
also matters: 21 of placebo's 58 native wins occur only after its continuation
allowance is exhausted in the saved-prefix sensitivity. Native outcomes remain
primary; neither distinction is silently substituted for the declared endpoint.

The placebo is not inert. Against E1 it is less successful and longer on table,
more successful and substantially shorter on inbox, and less successful on
tic-tac-toe and tabs. Checkbox native success is not distinguishable from E1
(-4.0 points [-8.5, +0.5], McNemar p=0.134), though its preservation bound
fails. Exact completions are borderline higher (+8.5 points [0.5, 16.5],
exact McNemar p=0.053), while joint native-success responses are much longer
(median +171.25%).
On tabs, placebo loses 72 E1 successes and gains seven; it is much stronger than
E2 but still falls well below task-only specialization. Those secondary results
matter when interpreting the mechanism control.

The matching component multiset removes the response-specific assignment
within each sampled group. It does not fix total reward variance, gradient
noise, future trajectories or optimization history. Question-paired intervals
quantify these fitted policies; they do not supply training-seed replication.
The campaign was authorized after the original outcomes were known, so the
new mechanism comparisons remain exploratory and their nominal intervals are
not simultaneous confirmation across five families.

## Provenance, coverage and cost

Original experiments, frozen runtime sources and raw results are preserved.
Inbox required approved audit-only recovery of reference-body coverage; all
600 outcomes and the original failed audit remain unchanged. Tabs passed its
normal technical review. Independent counterpart authority is retained beside
each report; the final tabs/combined review is recorded under
`placebo-l010-20261018-ops/tabs-analysis-20261024/review-status.json`.

Table and inbox match L4 hardware but span kernel/run-date differences. Tabs
matches the original L4 platform stamps. Checkbox E2/placebo share the home
4090, while E1 is an L4 reference. Tic-tac-toe placebo ran on home while E1/E2
ran on L4; its layout/hit-testing equivalence was not established. Matching
initial observations and package pins do not remove these accepted host limits.

Task-grounded action labels distinguish rejected calls, actual dispatch,
visible targets, native outcomes and observations available at decision time.
Unknown terminal attribution and precommitted batch actions remain explicit.
The reports' consistency samples are not independent blinded validation of
semantic harm, intent or reward-bias substitution. That is still thesis task 8.

The new placebos used approximately 135.48 L4 allocation-hours (table 40.632,
inbox 42.201, tabs 52.645) and 37.90 home allocation-hours (checkboxes 17.818,
tic-tac-toe 20.078), including all checkpoint evaluations. These are separate
hardware accounting totals, not equivalent compute, active utilization, money
or energy. There is no claim that speed compensates for lost success.

## Recommended next steps

The current `project:thesis` list makes writing and measurement validation the
best next work. Recommendations here do not authorize another experiment.

1. Use task 3 to integrate the five-family result table, paired intervals,
   negative findings, costs and mechanism limitations into methods, results and
   discussion. Keep table's strongest result alongside inbox's behavior shift,
   checkbox's exact-goal loss and tabs' collapse. Avoid one pooled claim across
   incompatible tasks and hosts.
2. Resolve task 8 with a bounded independent label-validation exercise on saved
   traces if the thesis needs claims beyond exact observable events. Otherwise
   narrow the claim to those events and explicitly retain unknown coverage.
   The existing offline module and review plan already define the starting point;
   no GPU run is needed for this decision.
3. Bring task 2 back for an explicit scope decision. If robustness evidence is
   essential and time permits, prioritize a small, declared matched-seed
   replication of the key contrasts over a broad dose/model/E3 expansion. Fix
   the comparison and host plan before launch, and retain every declared outcome.
   If time is constrained, finish a clearly labelled single-seed exploratory
   thesis rather than expanding automatically.

Task 4, submission requirements and front matter, can proceed independently.
Optional larger/alternate models, archived development results and task 7's E3
remain deferred. No new dose, seed, model, family, E3 or restart is authorized.

## Per-family evidence

- [Table](placebo_read_table_2_s4016_findings.md)
- [Inbox](placebo_email_inbox_noscroll_s4021_findings.md)
- [Checkboxes](placebo_click_checkboxes_soft_s4232_findings.md)
- [Tic-tac-toe](placebo_tic_tac_toe_s4230_findings.md)
- [Tabs](placebo_click_tab_2_s4231_findings.md)
