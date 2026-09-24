# Successful-response reward comparison and E1 continuation

Declared: 2026-09-23, before computing candidate results. Authority: decision
0019 and the user's explicit approval. This is development work; no old held-out
result is reused to select the new formula or coefficient.

## Offline comparison

Use the 13 saved training batches from table-pilot-e1-s4009 (416 trajectories).
The final ten batches are the primary calibration sample because continuation
starts from that competent policy. Batches 1/100/200 are sensitivity checks.
Keep the recorded model_token_count ruler and independently verify it with the
pinned tokenizer before admission. Never score the old held-out episodes for
candidate selection. Hash the inputs and this declaration in the output.

For binary environment success y, total reward is y - lambda * C:

- Linear: C = y * min(L / 4096, 1).
- Relative: C = y * sigmoid((L - mean_correct) / max(std_correct, 1)), using
  population standard deviation of correct lengths inside each eight-rollout
  prompt group. No correct responses means all costs zero; one correct response
  or equal correct lengths means cost 0.5 for each correct response. The
  one-token denominator floor prevents division by zero and limits amplification
  of sub-token numerical noise. Failures never enter these length statistics.

Compare lambda 0.1, 0.2 and 0.4 for each candidate. Retain naive_sum and
scale_rewards: none. Include the old weight-0.4 cosine as a descriptive ruler.

Hard checks: finite rewards; successful totals in [1-lambda, 1]; failures zero;
shorter correct responses preferred below the linear cap; group isolation;
permutation equivariance; no mutation of inputs or RNG; zero-weight identity;
doubling lambda doubles its centered contribution. Check no/all/one successes,
equal lengths, large outliers and the cap with labelled fixtures.

Selection rule: among late all-correct groups with differing lengths, seek a
median mean-absolute centered length advantage of at least 0.01 and a 90th
percentile at most 0.05. These are engineering calibration limits, not validated
predictors of compression or accuracy. Take the smallest tested coefficient
that meets them. Prefer linear if both designs qualify because its fixed
per-token interpretation needs fewer assumptions. Otherwise choose the
qualifying relative candidate. If neither qualifies, stop before model runs and
report the discrepancy; do not expand the grid after seeing results. Report
coverage, signal distributions and mixed-group behavior, not only the winner.

Native CPU replay must verify the selected implementation's actual reward
dispatch and mean centering against the independent formulas. Existing loss
replay can be reused; no model-parameter gradient attribution is added.

## Controlled continuation

After the offline selection, initialize both arms from the byte-identical final
adapter of table-pilot-e1-s4009 (original update 300). Use a new optimizer and
scheduler in both arms, explicitly recorded as a new training stage, not an
exact interrupted-run resume. Preserve the original adapter and optimizer files.

Use seed 4009, the same 500 training instances, 4 prompts x 8 rollouts,
microbatch 1, one update per rollout batch, learning rate 5e-5, 10% warmup,
90 additional updates, and the qualified remaining recipe. Save all thirds
(30/60/90). The control has task reward only; treatment adds only the selected
cost and coefficient. Primary endpoint is 90; no selection of an earlier result.

Use 100 fresh development instances at offset 400000 (4009400000-4009400099),
disjoint from training, earlier development offset 300000, and old test offset
100000. Evaluate the starting adapter and both arms' scheduled observations
greedily on these same instances. Pin the existing development E0 metric ruler.
These are development findings, not a fresh confirmatory thesis claim.

Reuse the first contrast's paired success-regression bound (one-sided 95%,
at most 5 percentage points) and jointly-correct median token reduction target
(10%, with paired 95% bootstrap interval). At n=100 the success gate permits
at most one regression. Compare treatment against the matched continued control
and separately against the starting adapter. Report gains, all stop/action
outcomes, active reward groups, and training/inference cost. E3 remains a warning.

Before this stage, qualify adapter initialization and the new reward with CPU
checks and a bounded three-update full-geometry execution check, including
sampler synchronization and checkpoint reload. A smoke is insufficient.
Use the existing capture mechanism only for readiness; do not expand gradient
preservation. Model runs require source/config/checkpoint-bound admission,
physical GPU 1, no overwrite, current RUNNING, and review before advancement.
Routine passing gates advance under the user's authorization. Stop on failed
technical invariants or a new scientific decision. No extra models, seeds,
families, coefficient sweeps or longer horizons are admitted by this plan.
