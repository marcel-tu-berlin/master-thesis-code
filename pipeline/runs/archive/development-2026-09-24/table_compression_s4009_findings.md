# Successful-response compression development study

Decision 0019 and docs/plans/read-table-compression.md declare the protocol.
This study asks whether the competent E1 update-300 adapter can be compressed.
It preserves all earlier results and adds no model-gradient history logging.

## Offline calibration, 2026-09-23

Both proposed costs were tested on 416 saved E1 training trajectories in 13
batches. No held-out trajectory entered selection. The final ten batches contain
35 all-correct groups with varying lengths and five mixed-correctness groups.
The declared signal statistic is the median, across those all-correct groups,
of mean absolute centered length-reward contribution. It is not a measured
parameter gradient or a prediction of achieved token savings.

| Cost | Weight | Median signal | 90th percentile | Meets declared calibration |
|---|---:|---:|---:|---|
| Linear | 0.1 | 0.001950 | 0.003031 | No |
| Linear | 0.2 | 0.003900 | 0.006062 | No |
| Linear | 0.4 | 0.007800 | 0.012124 | No |
| Relative | 0.1 | 0.017582 | 0.020507 | Yes, smallest admitted weight |
| Relative | 0.2 | 0.035163 | 0.041013 | Yes |
| Relative | 0.4 | 0.070327 | 0.082026 | No |
| Previous cosine, descriptive only | 0.4 | 0.002110 | 0.003499 | Reference |

The declared rule selects the relative cost at weight 0.1. The rule requires a
median at least 0.01 and a 90th percentile at most 0.05. These are engineering
calibration limits, not established scientific thresholds. Linear could still
work at another normalization or coefficient; this comparison does not establish
that relative rewards learn better.

The selected reward is task success minus 0.1 times a bounded sigmoid of length
relative to other correct responses to the same training question. Wrong
responses receive zero task reward and zero length cost. Shorter successful
responses receive higher total rewards. Every successful total exceeds every
failed total for the binary BrowserGym task.

The pinned tokenizer independently reproduced all 416 token counts. CPU replay
through the installed reward registry, composer and TRL advantage/loss paths
passed all 78 batch/design/weight combinations. The installed-stack tests passed
14 checks, including exact adapter tensor/prediction preservation and continued
trainability. Local full validation passed 463 tests with seven stack-dependent
skips, plus formatting, lint, types and 17 setup checks. This establishes wiring
and arithmetic; it does not establish learned compression.

Evidence: runs/table-compression-s4009-ops/offline_comparison_verified.json,
native_replay.json and installed_tests.log. The original offline output and its
source snapshot are retained; the verified rerun produced identical candidates.

## Admitted model stages

First qualify the new reward and adapter initialization with three full-geometry
updates and ten evaluation episodes. After technical review, evaluate the
starting E1 adapter on 100 fresh development instances. Then run matched
task-only and relative-cost continuations, each initialized from the exact same
E1 adapter with a fresh optimizer and 90 additional updates. Evaluate updates
30, 60 and 90; 90 is primary. Compare treatment against both the continued
control and the starting adapter, including paired success preservation and
jointly-correct token use. E3 remains secondary.

## Starting competent model, reviewed 2026-09-23

The original E1 update-300 adapter solved all 100 fresh development questions
(Wilson 95% interval 96.30-100%). Mean assistant-token use was 567.95 (bootstrap
95% interval 542.07-594.11). Every episode ended at environment completion after
three valid actions; none had an invalid or repeated action or exhausted its
budget. All visible-table answers, action traces and aggregate metrics were
independently checked. The source adapter hashes match the admitted E1 checkpoint.

This establishes the starting comparison for the two matched continuations.
It does not establish a compression effect. Evidence is in
runs/table-compression-start-s4009/{eval_report,technical_review,review}.json.
No continuation or compression verdict is available yet.

## Matched continuation result, reviewed 2026-09-24

Both arms started from byte-identical E1 update-300 adapter tensors with fresh,
matched optimizer and scheduler state. The relative arm adds only the selected
successful-response relative cost at weight 0.1. All 90 updates were finite;
all 416 observed rollouts terminated, and each of 13 groups had nonconstant
relative-cost signal. Source, stack, initializer and scheduled-evaluation
provenance passed review.

Across all 90 treatment updates, four had zero recorded gradient norm (steps
40, 68, 76 and 77); the other 86 had positive gradient norm. Each zero-gradient
step had environment reward 1 for every rollout, constant raw length component
-0.5, and `frac_reward_zero_std: 1`. Mean centering therefore removed all
reward-driven signal for those batches. This does not imply unchanged parameters:
optimizer momentum and weight decay can still act. Evidence: the complete
train_log.json, not only the 13 sampled observation batches.

The primary update-90 treatment/control comparison was 99/100 versus 92/100
correct: eight gains and one regression. The one-sided 95% regression upper
bound was 4.66 percentage points. Among 91 jointly-correct episodes, median
assistant-token change was -90.90% (paired bootstrap 95% interval -91.25% to
-90.34%), meeting the declared 10% compression target. Treatment/start was
99/100 versus 100/100, with one regression and -77.30% median tokens (95%
interval -78.75% to -75.44%). Control/start was 92/100 versus 100/100 and
+158.55% median tokens (95% interval 140.30% to 172.42%).

| Update | Control correct / mean correct tokens | Relative correct / mean correct tokens |
|---:|---:|---:|
| 30 | 98 / 937.80 | 99 / 223.95 |
| 60 | 98 / 948.81 | 97 / 155.80 |
| 90 | 92 / 1463.10 | 99 / 127.76 |

The relative trajectory shortened at every observation; control lengthened.
At step 90, treatment had one wrong submission and no invalid or repeated
action; control had eight wrong submissions and nine invalid actions. Neither
arm had non-termination, voluntary stops or repeated actions. Training sampled
53,080 model tokens in treatment versus 457,483 in control (416 rollouts each);
train wall time was 4.02 versus 6.24 hours. These are development findings,
not confirmatory evidence. The admitted horizon is complete; no further model
run is admitted without a new user decision.
