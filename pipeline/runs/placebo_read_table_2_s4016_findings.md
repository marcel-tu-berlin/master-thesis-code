# Read-table-2 seed 4016: matched 0.1 placebo

Reviewed 2026-10-20 (Europe/Berlin). Placebo completed at 2026-10-20
00:45 UTC. Its 29 non-weight files, including checkpoint evaluations, were
harvested and hash-verified after L4 access returned. This report adds the
authorized placebo comparisons to the completed original findings. Update
300 remains primary.

## Result

E2 outperforms the shuffled-cost placebo on the declared endpoints in this
seed. E2 solves 198/200 questions versus placebo's 189/200, with ten gains and
one regression. The paired success difference is +4.5 percentage points
(95% bootstrap interval +1.5 to +8.0; exact McNemar p = 0.01172).
On 188 jointly correct questions, E2 uses a median 79.64% fewer assistant
tokens (95% interval 77.51% to 81.13% fewer). Every jointly correct E2 response
is shorter.

The one-sided 95% upper bound on gross regressions is 2.35%, below the
declared 5% margin. The compression interval clears the 10% target. All four
declared off-target bounds also pass. These are the specified endpoint
criteria, not proof of complete behavioral equivalence: E2 introduces one
wrong submission and changes the failure distribution.

Placebo does worse than E1 at the final checkpoint: ten fewer successes,
longer jointly correct responses, and more invalid actions. Both E2 and
placebo keep almost every training group active, while only E2 produces
sustained compression. This supports the relevance of assigning the cost
according to length, beyond merely adding a variable reward component, in
this run. It does not isolate total-reward variance or gradient noise, nor
establish the mechanism across training seeds or task families.

This is a subsequent exploratory comparison after the original results were
known. Intervals describe held-out-question uncertainty within one training
seed. The checkpoint observations are not replications, and the nominal
intervals are not simultaneous campaign-wide confirmation.

## Design and verification

The protocol is `docs/plans/placebo-l010-campaign.md`, decision 0028, and the
original `docs/plans/e0-e2-campaign.md`. Runs:

- Placebo: `e2placebo-l010-read-table-2-s4016`.
- Primary comparator: `e2-read-table-2-s4016`.
- Secondary comparator: `e1-read-table-2-s4016`.

The frozen placebo/E2 configs differ only in experiment ID, description and
`successful_length.placebo: true`. Weight remains 0.1. Both start from the
pinned Qwen3-1.7B with fresh LoRA and retain 300 updates, batch four, eight
rollouts, `naive_sum`, `scale_rewards: none`, the optimizer settings and
4096 trajectory / 4096 prompt / 8192 context / eight-turn limits. Retain the
implemented batch/continuation semantics; these limits are not a new strict
per-action transcript cutoff.

All historical runtime source hashes match, with no GPU routing patch for
this L4 cell. Python 3.12.3, package pins, OpenEnv and episode-boundary stamps
match. All three runs used L4; original E1/E2 record Linux 6.8.0-139, while
placebo records 6.8.0-146, with glibc 2.39. This is the same host/GPU class,
but not a simultaneous controlled timing experiment. Kernel and run-date
differences remain timing and reproducibility caveats. The home-host
differences in the other families do not apply to this table contrast.

At every checkpoint, all arms have the same 200 seeds, 4016100000 through
4016100199, and identical initial observations. The 416 captured training
seed/prompt/group/slot identities also match. The original E1/E2 artifacts
match their saved review hashes; their completed comparison is not repeated.

The passing technical receipt covers 300 optimizer records, 600 placebo
episodes, native outcomes, metric recomputation, tokenizer recounts,
checkpoint/final adapter identity, frozen configuration/source/stack and four
complete cost attempts. Local verification rechecked the 29 harvested files,
reran the table auditor for all 600 new episodes and replayed the seeded
shuffle capture check. Of 416 captured rollouts, 360 had changed assignments.
The observer's raw component values remain unshuffled; its composed reward
contains the applied shuffled assignment.

## Checkpoint results

| Update | E1 correct / 200 | Placebo correct / 200 | E2 correct / 200 | E1 / placebo / E2 mean tokens on successes |
|---|---:|---:|---:|---:|
| 100 | 183 | 193 | 184 | 1506.58 / 974.13 / 1094.62 |
| 200 | 197 | 185 | 199 | 868.89 / 1399.79 / 372.11 |
| 300 | 199 | 189 | 198 | 463.95 / 1258.26 / 228.68 |

Final success rates and Wilson 95% intervals are E1 99.5% [97.22%, 99.91%],
placebo 94.5% [90.42%, 96.90%], and E2 99.0% [96.43%, 99.73%].
The success-conditioned means above describe different subsets; paired
efficiency uses the intersection below.

| Contrast | Update | Gains / regressions | Joint successes | Median paired token change, 95% CI |
|---|---:|---:|---:|---:|
| E2 minus placebo | 100 | 7 / 16 | 177 | +23.41% [+14.14%, +36.67%] |
| E2 minus placebo | 200 | 15 / 1 | 184 | -71.10% [-73.58%, -68.20%] |
| E2 minus placebo | 300 | 10 / 1 | 188 | -79.64% [-81.13%, -77.51%] |
| Placebo minus E1 | 100 | 16 / 6 | 177 | -42.71% [-47.11%, -37.88%] |
| Placebo minus E1 | 200 | 3 / 15 | 182 | +61.11% [+39.30%, +78.15%] |
| Placebo minus E1 | 300 | 1 / 11 | 188 | +156.76% [+130.06%, +185.32%] |

E2 is initially longer and has fewer observed successes than placebo; the
update-100 success contrast is not significant (McNemar p=0.093).
The later reversal
is retained; no favorable checkpoint is selected. At update 300 the absolute
paired median is -852.5 tokens [-952.5, -776.0], with paired means
1260.69 for placebo and 228.97 for E2. All 188 pairs shorten, with none tied.

The secondary final placebo-minus-E1 success difference is -5.0 percentage
points [-8.5, -2.0], exact McNemar p = 0.00635. Its eleven regressions have a
one-sided upper bound of 8.94%, failing preservation; its token increase
also fails compression. Of 188 joint successes, five shorten and 183 lengthen.
The one gain is index 123. All indices below are zero-based.

## Task-grounded behavior

All 600 new trajectories were checked against the two requested labels,
visible table values, executed actions, feedback and submission state.
All new paired regressions at updates 100/200/300 were inspected. The
retained evidence contains each case's expected and final fields, executed
calls, errors, stops and paired successful reference.

At update 300, placebo has ten budget stops and one wrong submission:

- Nine episodes reach 4096 model tokens with no executed action: indices
  14, 66, 71, 80, 120, 126, 147, 185 and 186.
- Index 168 fills Male and aqua correctly, then tries to fill a table cell
  with Allyn. It consumes its remaining continuation allowance without
  submitting. Its 3237 model tokens exclude charged tool feedback.
- Index 151 submits Arllyn instead of the displayed Arlyn. E2 also fails
  this question, using Arlin. This is a shared failure, not an E2 gain.

E2 solves all ten placebo budget-stop cases. Its sole regression relative
to placebo is index 182: it copies Punjabi from Language into the Religion
field instead of Judaism. Placebo submits the correct Florina and Judaism.
This remains a real error despite the passing gross-regression margin.

Placebo has 26 invalid calls in 14 final episodes. All attempt to fill
non-input elements; 13 of these episodes still submit correctly. The one
unsuccessful invalid-action case is index 168 above. There are no repeated
actions or voluntary no-tool stops in either final E2 or placebo.
Final stop transitions are 190 env-done to env-done and ten
placebo budget stops to E2 env-done.

Intermediate placebo errors also have concrete causes. At update 100,
there are six wrong submissions and one no-action budget stop. Wrong cases
copy labels or bid strings instead of values, overwrite correct fields with
unrequested values, or confuse Language with Religion. At update 200,
there are four wrong submissions and eleven budget stops, nine before any
action and two after filling the correct values. Index 30 twice omits the
click's required bid; index 123 attempts extra table-cell fills and never
submits. The four wrong submissions use label names, an empty last name,
Arlin for Arlyn, or bid strings. None is a runtime failure or an audit exception.

The primary E2 regressions at update 100 comprise nine budget stops, six
label-for-value substitutions and one swapped-field submission. The sole
regression at both updates 200 and 300 is index 182. These cases and all
secondary placebo regressions are retained in `behavior-review.json`.

### Declared off-target bounds

These are final episode counts. "New" means present in the named arm but
absent in its paired reference, not the net count difference. Bounds are
one-sided exact 98.75% bounds, alpha 0.05/4, with a 5% margin.

| Contrast / event | Reference -> arm | New / resolved | Upper bound |
|---|---:|---:|---:|
| E2 minus placebo: wrong submission | 1 -> 2 | 1 / 0 | 3.15% |
| E2 minus placebo: voluntary no-tool stop | 0 -> 0 | 0 / 0 | 2.17% |
| E2 minus placebo: invalid action | 14 -> 0 | 0 / 14 | 2.17% |
| E2 minus placebo: repeated action | 0 -> 0 | 0 / 0 | 2.17% |
| Placebo minus E1: wrong submission | 1 -> 1 | 1 / 1 | 3.15% |
| Placebo minus E1: voluntary no-tool stop | 0 -> 0 | 0 / 0 | 2.17% |
| Placebo minus E1: invalid action | 4 -> 14 | 14 / 4 | 12.15% |
| Placebo minus E1: repeated action | 0 -> 0 | 0 / 0 | 2.17% |

The primary contrast passes all four margins. The secondary contrast fails
the invalid-action margin. Non-termination/budget exhaustion is separately
reported: placebo 10/200, E1 and E2 0/200 at the final checkpoint.
Neither nonsignificance nor passing selected margins establishes full
behavioral equivalence.

### Masked controls

The retained seed-17 procedure was applied to the nine evaluations in the new
E1/E2/placebo comparison, preserving stratum priority and selecting up to six
cases per stratum. This is a new sample, not a claim to reproduce the original
seven-evaluation sample. The pooled correct-response first quartile is
387 tokens. Available strata are 37 budget, one voluntary, 53 invalid/repeated
and 433 efficient-correct cases; 19 were selected.

Arm/checkpoint/seed/index were hidden until judgments were saved, but recorded
outcome, reward, termination, stop reason, invalid/repeated counts, token
counts and wall times remained visible. Agreement of all 19 judgments with
those visible labels is consistency-only, not independent label validation.
Length and timing could also reveal the arm or checkpoint. The six efficient
controls, revealed as E2 cases, all fill the two
correct values and submit in three actions. The sample also contains a
voluntary false completion claim without submission, rejected extra fills,
missing click arguments, label/ID confusion and budget exhaustion.
The sample is not balanced by arm and is not an independent blinded rating:
aggregate results and original findings were already known, and some traces
could be recognizable. Label validation rests on the deterministic audit of
all 600 placebo episodes against the table and executed actions, rather than
on agreement with the labels shown in this small manual sample.

The frozen E0 short-response threshold remains 1213.7 tokens (upper threshold
4096). It flags 100% of E2's correct final responses, 98.49% of E1's and
57.67% of placebo's. Valid short controls show why this legacy flag cannot
be interpreted as harmful underthinking.

## Training activity

| Measure | E1 | Placebo | E2 |
|---|---:|---:|---:|
| Mean active prompt-group fraction | 34.17% | 98.58% | 99.08% |
| Updates with any live group | 64.00% | 100% | 100% |
| Zero-gradient updates / 300 | 108 | 0 | 0 |
| Correct rollouts in final ten captured updates / 320 | 316 | 318 | 318 |
| Task-variable groups in that tail / 40 | 3 | 2 | 2 |
| Composed-reward-variable groups in that tail / 40 | 3 | 40 | 40 |

The placebo preserves the within-group cost multiset, including failure
zeros, but changes its assignment. It does not guarantee identical total
reward variance, gradient magnitudes, state visitation or optimization noise.
The nearly equal liveness of E2 and placebo makes a simple "any extra live
reward is sufficient" explanation inadequate for this observed difference.
It does not supply training-seed replication or causal decomposition.

## Measured costs

Four complete attempts per arm cover training and evaluations at
100/200/300. No unmatched cost starts or failed attempts occur in these
run ledgers. Sum each end record once; checkpoints' nested temporary timers
are not counted again.

| Arm | Training GPU-hours | Evaluation GPU-hours | Total GPU-hours |
|---|---:|---:|---:|
| E1 | 28.134 | 7.471 | 35.605 |
| Placebo | 30.781 | 9.851 | 40.632 |
| E2 | 22.582 | 5.007 | 27.588 |

The table placebo therefore took about 40 hours 38 minutes for training and
all three evaluations, including about 30 hours 47 minutes of training.
These are wall-time allocations, not energy, kernel time or monetary cost.
Deployment/preflight, review work and the unchanged original E0 evaluation
are outside this per-arm table.

At the final checkpoint, E2 saves a mean 48.38 inference seconds per question
against placebo [43.62, 53.39], and 42.63 seconds on their joint successes
[39.15, 46.25]. The separate joint-success figure prevents failed episodes
from supplying the successful-compression claim. Placebo instead costs
38.54 additional seconds per question versus E1 [33.56, 43.68], or 32.80
additional seconds on joint successes [28.97, 36.73].

As a descriptive same-L4 cost calculation, charging E2's full training cost
against its all-question inference savings relative to placebo gives about
1680 deployment episodes to amortize training. E2 training itself used 8.199
fewer GPU-hours than placebo, so there is no positive incremental training
cost to recover in that comparison. Placebo costs 2.646 more training
GPU-hours than E1 and saves no inference time; it has no finite break-even
against E1. Research evaluation expense remains separate. These timing
calculations are conditional on the measured runs, including the kernel and
run-date differences, and are not a deployment or causal speed guarantee.

## Evidence and remaining review

Reproducible analysis and task-grounded evidence are in
`pipeline/runs/placebo-l010-20261018-ops/table-analysis-20261020/`:
`analyze.py`, `analysis.json`, `review_cases.py`, `behavior-review.json`,
the seed-17 selection and saved judgments, and the original `validation.json`.
The analysis reuses the retained paired-statistic implementation, 20000
bootstrap resamples and seed 0. Analysis and behavior outputs reproduced
byte-for-byte. All 20 operations tests and 30 paired-statistic tests passed.
The original validation file stated those external check results as fixed
text; it did not itself execute them. The correction validation now retains
the actual test and regeneration logs, exit codes and output hashes.
No training/evaluation implementation was changed.

Authentication is restored. Independent Opus 5.5 at xhigh found two reporting
issues: the visible labels limit the masked sample to a consistency check,
and test/regeneration claims needed retained execution evidence. Both are
addressed above. Corrective Opus 5.5/xhigh review passed on 2026-10-20;
its verdict, input hashes and execution receipt are retained as
`counterpart-correction-{output,receipt}-20261020`.
Task 9 stays open for
the remaining inbox/tab findings and pending counterpart reviews. The L4
controller owns further authorized advancement; this result authorizes no
additional experiment.
