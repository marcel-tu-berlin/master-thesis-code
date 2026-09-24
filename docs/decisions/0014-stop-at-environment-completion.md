# Stop training and evaluation at environment completion

Decided: 2026-09-16. Status: accepted; E1/E2/E3 runtime requalification passed
on 2026-09-17. Evidence: [termination readiness review](../../pipeline/runs/termination_readiness_findings.md).

New model runs use `env_done_or_budget_v1`: after a dispatched action reports
environment completion, no further action or assistant turn is generated for
that episode. Completion includes terminal failure as well as success. Other
episodes in the batch continue independently. The last permitted assistant
turn is also the last generation; there is no extra unacted turn after the
tool-iteration limit. Voluntary stopping and generation-cap endings remain
separate outcomes.

An assistant turn is generated before its tool calls are executed. If it contains
multiple calls, every token already generated remains counted and trained, but
calls after the first terminal action are not dispatched. The terminal tool
feedback is retained in audit messages. It is not added to model inputs, token
masks or sampling log probabilities because there is no subsequent generation.
The same applies to feedback after the last allowed turn. Earlier interleaved
feedback retains the native TRL tokenization, masking and budget accounting.

The pinned TRL 1.6.0 loop has no environment-done boundary and regenerates after
its final tool iteration. The pipeline overrides that loop with explicit stop
guards; the rest of its generation and token bookkeeping is retained. A hash
guard refuses a changed upstream method. This path is qualified for synchronous
text tools, the interface both current domains use. No package patch or upgrade
is required. Evaluation already stops at the accepted boundaries.

This changes training trajectories, efficiency-reward inputs and potentially
E3 classification at the last allowed turn. Old captures and the completed Gate 4
pilot remain evidence of the old protocol. They do not validate this correction
and cannot be mixed with corrected runs as matched reward-ablation arms. Their
artifacts and source snapshots remain unchanged. The Gate 4 saturation finding
still describes the policy that was actually trained and evaluated; it is not
relabelled a result from the corrected protocol.

Before new model screening, run the CPU regressions and installed-stack
tokenizer/tool-loop checks, including mixed terminal/live slots, terminal failure,
multiple calls, the last allowed turn, overlong feedback and unchanged
nonterminal behavior. Then use fresh `g3-envdone-e1/e2/e3.yaml` configurations
for sequential three-update diagnostics at the accepted 4 x 8 geometry. Review
the deployed controller after each arm, including reward/advantage/loss/gradient
replay, checkpoint reload and evaluation. Preserve the existing cumulative
two-hour diagnostic ceiling. Do not authorize a broad batch or another Gate 4
pilot from this decision; the next research task is the bounded harder-family
screen after readiness passes.

The observed mismatch is documented in
[`gate4_e1_findings.md`](../../pipeline/runs/gate4_e1_findings.md).
