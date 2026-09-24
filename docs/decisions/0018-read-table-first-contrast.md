# Continue read-table-2 through the first efficiency contrast

Decided: 2026-09-21. Accepted by the user after reviewing the seed-4009 pilot
and the literature reassessment. Supersedes decision 0017's final-E1 success
ceiling and decision 0004's training-family choice for this first contrast.

Use read-table-2 for the first informative E1/E2 result. A successful task-only
policy may saturate success and still provide an informative compression
experiment. Require verified learning, valid shorter correct trajectories,
within-question length-reward variation, and an evaluation that can detect
success loss. Remove the final-E1 below-90% admission requirement. Keep the
sampled development screen and technical requirements unchanged. E3 remains
an optional warning; it does not gate E1/E2.

The seed-4009 pilot remains a failure under its original success-ceiling rule.
Its 100/100 development result, all measurements and review receipts remain
unchanged. Its technical review and remaining E2 opportunity admit the first
contrast under this new decision. No old result is invalidated. This amendment
uses development evidence and precedes the new evaluation and shaped training.

Reuse the completed seed-4009 E1 checkpoints. Start E2 from the identical base
model with matched training geometry, instances, recipe and 300-update horizon.
Keep the existing cosine endpoints, token counting, environment interface,
termination, stack, and mean-only advantages. Freeze the first weight, split,
success-preservation margin and analysis before any new model evaluation.
The concrete declaration is in [the first-contrast plan](../plans/read-table-first-contrast.md).

The user's instruction to try more families or environments after the first
result defers the proposed cross-family evaluation. This result concerns new
instances of read-table-2, not structural transfer or general agent safety.
The behavioral panel remains descriptive unless grounded trajectory review
supports its interpretation. Preserve the cost-assignment placebo and multiple
training seeds as follow-up work; one E1/E2 pair cannot establish a mechanism
or training-seed robustness.

Automatically harvest, review and advance the declared passing stages until
the first contrast has a technical and scientific verdict. A null, negative or
statistically inconclusive contrast is reportable evidence, not permission to
search weights or endpoints until a benefit appears. Stop for a verified
technical failure or a genuinely new protocol decision, and after delivering
the first result for review. Do not start other families, E3 or a broad sweep.
