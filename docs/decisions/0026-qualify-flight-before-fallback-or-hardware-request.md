# Qualify flight booking before fallback or a hardware request

Decided: 2026-10-06, following the user's acceptance of the family recommendation
and explicit restriction on fallback.

Qualify `book-flight-nodelay` as the next MiniWoB++ family. Its search, result
comparison, correction and terminal booking decisions could expose failure modes
beyond the completed table and inbox studies. This is a candidate rationale,
not evidence of model difficulty or reward-bias substitution.

Switching to `login-user-popup` or any other family requires the user's explicit
approval. If the current limits prevent an informative flight study, investigate
the cause and prepare an evidence-backed resource request for the supervisor
before proposing a fallback. The user decides whether to send that request and
whether to pursue another family while waiting.

The user subsequently authorized opt-in scrolling, complete-path measurement
and assessment. This extends the candidate's tool interface, not the retained
campaigns' interfaces or budgets. Results and the next diagnostic proposal are
in `pipeline/runs/flight_s4022_scroll_findings.md`.

## Qualification and decision order

1. Replay scripted correct, deliberately incorrect and recoverable paths through
   the actual pinned wrapper. Verify visible action IDs, hidden-state exclusion,
   deterministic resets, native scoring and termination. Use observation-grounded
   actions; privileged task state may audit outcomes but must not supply choices.
2. Measure prompt and feedback tokens and the sequential decisions needed for
   complete workflows. Check autocomplete, calendar navigation, result coverage
   and backtracking. Separate missing tools or observations from model errors.
3. If the interface is valid, use a declared bounded base-policy screen on fresh
   development seeds, recording success variation, successful-response length
   variation, failure types and costs. A scripted solution is not evidence that
   the model is trainable; a small model screen is not evidence of durable
   post-training complexity.
4. Diagnose any limiting resource before changing the protocol. A turn cap is an
   experimental setting; reaching it does not establish a VRAM shortage. A token
   cap can reflect feedback overhead, inefficient behavior or a task requirement.
   A missing scroll action or incorrect scorer is an interface issue. Larger
   hardware is justified only by a measured capacity or runtime requirement for
   a scientifically useful protocol.

Decision [0025](0025-preserve-agent-budgets-across-families.md) still governs:
4,096 trajectory tokens including feedback, 4,096 prompt tokens, 8,192 context
tokens and eight turns. A larger GPU does not itself authorize larger budgets.
Any proposed change must state the new values, scientific reason and effect on
comparability, and receive explicit approval before execution. Preserve existing
results and use fresh run IDs. Keep physical GPU 1, the pinned model/stack and
4-by-8 geometry. This qualification does not authorize a full E0/E1/E2 campaign.

## Evidence required for a supervisor request

Prepare a concrete draft only if qualification establishes a need. Include:

- The additional scientific question and the observed limitation preventing it.
- Representative trace IDs and token/turn distributions, separating valid
  solutions, model failures and interface failures.
- The exact workload: model, precision, context, rollout geometry, software
  versions, GPU model and usable memory, with concurrent processes accounted for.
- Measured allocated/reserved/device memory through generation, loss and backward,
  or measured throughput if runtime is the constraint. A zero-LR capture or
  cropped tensor replay alone does not establish ordinary training feasibility.
- The proposed capacity and allocation time, their derivation and uncertainty.
  An OOM allocation size is not the total memory requirement; extrapolations must
  be labelled and checked on the proposed hardware before a full campaign.
- Why the requested resource addresses the observed limit, which scientifically
  equivalent options were checked, and what useful work can continue meanwhile.

The [earlier inbox memory failure](../../pipeline/runs/inbox_s4021_e1_failure_findings.md)
is supporting history only. It does not prove that flight booking needs a larger
budget or establish a minimum GPU size for this family.

While access or hardware is pending, continue local qualification preparation,
analysis of retained results and thesis work. Any runnable flight diagnostics
remain within the approved protocol. Family changes remain a separate decision.
