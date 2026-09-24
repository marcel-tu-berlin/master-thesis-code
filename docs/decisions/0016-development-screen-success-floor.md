# Lower the development-screen success floor to 30%

Decided: 2026-09-17. Status: accepted by the user after the fresh read-table-2
screen, before the seed-4008 diagnostic. Supersedes the 40% lower bound in
decision 0002 and the active development-screen admission rule.

Use a 30-80% sampled base-success band for development family screening, together
with the existing within-question and cost-opportunity requirements. For the
fixed sixteen-question, eight-rollout screen, retain at least four mixed-success
groups, at least four groups with two correct solutions whose token range is
at least 10% of their median, at least 10% budget exhaustion, and at least two
groups with varying budget cost. All technical checks must pass.

The lower bound is an operational heuristic, not a demonstrated learnability
threshold. It excludes near-zero-success tasks while allowing harder tasks
that provide actual within-group reward variation. The observed read-table-2
screen has mixed success in fifteen of sixteen groups and both targeted costs;
its evidence is in the [requalification findings](../../pipeline/runs/feedback_requalification_findings.md).
Thirty percent is a round, more permissive development threshold, not a claim
that it is optimal. Pooled success alone never admits a task.

This decision uses already observed development data. Preserve the original
40% failure and its captured summary unchanged. Record a separate admission
under this amendment; do not relabel the screen as a prospective pass or claim
independent validation from it. No earlier measurements are invalidated, no
final-test data are used, and no screen extension is authorized.

The user selected read-table-2 and authorized continuation. Admit the existing
conditional sequence: paired E0 reference, then a fresh seed-4008 30-update E1
diagnostic. Only a diagnostic pass admits the separate seed-4009 300-update
pilot. Keep the model, stack, rewards, trajectory/turn budgets, geometry,
development allocations, and post-training feasibility rules unchanged.

The 30% screening floor is for sampled base-policy groups. It is not a new
threshold on greedy E0 calibration and does not relax the post-training
success ceiling, E2 opportunity, E3 budget-cost or bounded-uncertainty rules.
The final pilot is still required before claiming family suitability. Changing
this decision again requires another explicit recorded protocol decision.
