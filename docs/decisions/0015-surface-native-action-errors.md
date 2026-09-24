# Surface native action errors through domain adapters

Decided: 2026-09-17. Status: accepted; fresh E1/E2/E3 readiness passed.
The completed family screen fails its base-success floor; evidence and the
pending family/protocol choice are in
[the requalification findings](../../pipeline/runs/feedback_requalification_findings.md).

The user chose to repair and requalify read-table-2 after the bounded screen
exposed missing BrowserGym action feedback. Environment-specific response
normalization belongs in each domain adapter. The shared training/evaluation
core continues to consume tool feedback, reward and done through the existing
domain contract; it must not branch on a MiniWoB family or prescribe its answer.

For BrowserGym, preserve a nonempty top-level error. Otherwise, when the native
failure flag is set, surface the message stored in BrowserGym observation
metadata. If details are absent, report the failure explicitly. Keep this in
the common action path used by click, fill and noop and by both training and
evaluation. Successful feedback, reward, termination, tool schemas and budgets
retain their existing definitions. No upstream package patch is required.

This changes the policy's observation context after a failed action. It can
change subsequent actions, generated length and the remaining trajectory
budget. The interrupted `family-screen-table-s4004` through `-s4006` runs remain
diagnostic evidence of the earlier interface, not matched controls for the
corrected one. Earlier readiness qualifies its recorded source only. Preserve
all captures and use new experiment IDs and source fingerprints.

The readiness checker must accept the explicitly configured MiniWoB family.
Its recipe, geometry, source/stack and cross-arm equality guards remain in
force. A missing task list or mismatched E1/E2/E3 families still blocks launch.
Shared config validation rejects malformed task lists in training and eval
split overrides; omitted/null lists keep their existing default semantics.
This removes a menu-only admission assumption without making arbitrary
environments scientifically qualified by configuration alone.

First reproduce and close the regression with CPU tests and real-server
valid/invalid actions, tool-schema parity and correct/wrong terminal controls.
Then run three full-geometry updates each for E1/E2/E3 on read-table-2 using
seed block 4010, with four development eval questions at offset 700000. Review
each completed run with the existing native arithmetic/loss/gradient controller.
Keep the cumulative two-hour readiness ceiling.

After readiness passes, repeat the bounded frozen-policy screen on sixteen
fresh questions in seed blocks 4011-4014, four questions and eight rollouts per
block. Keep the previously declared thresholds and 90-minute screen ceiling.
The short E1 diagnostic and final pilot remain conditional on that screen.
The user's instruction to continue through passing gates remains in force.

## Transport failure found during requalification

The first E1 attempt exposed keepalive disconnections and training's handling
of tool exceptions as policy feedback. Evaluation already distinguishes invalid
model arguments from exceptions inside a tool. Apply that same distinction to
training: argument-binding failures remain feedback; tool-body failures abort
before the affected batch can produce a learning update. Do not score a broken
environment as policy failure or reconnect silently within an episode.

Both OpenEnv clients and the shared server retain keepalive traffic but disable
the Pong deadline that a long native compute pause can exceed. OpenEnv's active
connection and message deadlines remain in effect. The pipeline serves each
pinned OpenEnv ASGI app through one shared entry point; environment-specific
observations and actions remain in the adapters. Forward server logs to the
runner instead of leaving an unread subprocess pipe. No dependency upgrade,
reward change, trajectory-budget change or family-specific recovery is needed.

Preserve the failed `feedback-v1` capture and restart the three diagnostics
with `g3-feedback-v2-e1/e2/e3.yaml`. Count the failed attempt against the same
two-hour readiness ceiling. Fresh screening blocks 4011-4014 remain unused and
the screen rules are unchanged. Captures containing transport failures cannot
establish model difficulty; the affected evidence is identified in the
requalification findings.
