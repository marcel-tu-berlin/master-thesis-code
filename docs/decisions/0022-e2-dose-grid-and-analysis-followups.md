# Restore the E2 dose grid and defer interpretation checks

Decided: 2026-09-24, before any final-campaign run, at the user's request.

Keep decision 0021's independent from-base E0/E1/E2 comparison, relative
successful-response cost, fixed recipe and deferred E3. Supersede its restriction
to a single E2 weight and no placebo: the thesis calls for three nonzero doses,
a cost-assignment placebo and at least three training seeds per condition.

- E0: original Qwen3-1.7B evaluation, no training.
- E1: task-only reward, the zero-dose control.
- E2: task reward minus lambda times relative successful-response cost, at
  lambda = 0.05, 0.1 and 0.2, each trained independently from the original base.
- E2 placebo: lambda = 0.2, with the same cost uniformly shuffled within each
  prompt group. Task reward is never shuffled. This tests cost assignment; it
  does not promise matched total-reward variance or gradient noise.

The central 0.1 dose has development evidence; 0.2 also passed the saved-batch
signal calibration. The 0.05 dose deliberately probes weaker pressure and is
not required to clear the earlier candidate-selection signal floor. No new
training pilot is required. Complete seed 4016 first, then the same cells at
4017 and 4018; retain null or adverse results and do not select replication by
significance. This is 15 trained runs plus three matching E0 evaluations.

Keep 200 held-out questions per seed and planned updates 100/200/300, with 300
primary. Statistical precision and off-target construct validation are deferred
to interpretation, not launch gates. Do not enlarge samples or change margins
after seeing these results. Report imprecision as such; keep unvalidated action
proxies descriptive until task-grounded trajectory review supports a stronger
label. A later precision calculation is not a prospective power justification.

Measure costs from the first final run. Additive timing fields do not alter
reward, token counting, seed mapping, termination or optimization. Earlier runs
without these fields have unknown measured costs; never backfill them as zero.

Full allocation, execution order and cost definitions:
[E0-E2 campaign](../plans/e0-e2-campaign.md).
