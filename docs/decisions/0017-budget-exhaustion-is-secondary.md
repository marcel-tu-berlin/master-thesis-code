# Treat budget exhaustion as a secondary diagnostic

Decided: 2026-09-18. Status: accepted by the user after the seed-4008 paired
diagnostic and its single extension. Supersedes the E3 admission requirements
in decision 0016 and the active family-suitability and Gate 4 plans.

The core research requires learning headroom and an informative efficiency
signal. E2's correctness-conditioned length signal is the current candidate.
E3 was added to investigate budget problems; its opportunity must not determine
whether an otherwise suitable family can advance.

Keep the 10% budget-exhaustion reference and its existing Wilson 95% interval.
If the interval's lower bound is below 10%, record an E3 opportunity warning.
Keep reporting within-group budget-cost variation, including the existing
two-varying-groups screen diagnostic. Neither budget criterion blocks family
admission, requires another sample extension, or forces a family change.
Budget exhaustion still means `max_turns` or `hit_generation_cap`; voluntary
`no_tool_call` remains separate. Measurement definitions and budgets do not change.

The following remain hard requirements:

- Technical, reward, trajectory, provenance and evaluation checks pass.
- The sampled development screen meets the accepted 30-80% success band and
  has at least four mixed-success groups among sixteen questions.
- The screen has at least four groups with two correct solutions whose token
  range is at least 10% of their median. At trained endpoints, inspect valid
  shorter correct controls and within-question length-reward variation. The
  E2 planning margin remains a 10% reduction; this is an opportunity criterion,
  not a claim that shaping has achieved it.
- Final-E1 success retains headroom: its Wilson upper bound is below 90%.
  Only ambiguity in this success criterion can trigger the single paired
  100-question extension. An unresolved hard criterion then requires review.

This amendment follows observed development results. Preserve the original
seed-4008 failure, review receipt and all measurements. Record a separate
admission under this decision; do not relabel it as a prospective pass.
The 200-question diagnostic passes the core criteria and carries an E3 warning,
so the existing fresh seed-4009 pilot is admitted. No further seed-4008 samples
are needed. No final-test data have been used and no old measurements are voided.

Continue with the paired seed-4009 E0 reference, its technical review, and the
300-update task-reward-only E1 pilot. Keep 500 training questions, 100 same-family
development questions at offset 300000, checkpoints 100/200/300, and the qualified
model, runtime, recipe and budgets. Step 300 remains the endpoint. E0-to-E1
compression does not establish an E2 treatment effect: E2 must later be compared
with the trained E1 control. Review the final core verdict before freezing and
launching the shaped campaign. E3 is an optional, separately interpreted extra;
no E3 sweep or additional budget-focused evaluations are admitted here.
