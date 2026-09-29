# Preserve agent budgets across families

Decided: 2026-09-29, following the user's explicit correction and restart request.

All active experiments retain the completed seed-4016 E0/E1/E2 limits:

| Limit | Value |
| --- | --- |
| Model context | 8,192 tokens |
| Prompt allowance | 4,096 tokens |
| Training whole-trajectory allowance | 4,096 tokens |
| Evaluation allowance, including checkpoints | 4,096 tokens |
| Successful-length reference maximum | 4,096 tokens |
| Tool-calling turns | 8 |

Tool feedback remains charged to the trajectory allowance; assistant-token
measurement is unchanged. Write the keys explicitly in every config. The active
config test checks both the recorded values and the runtime budget resolvers.
Historical configs and artifacts retain their original values.

Approval to extend a family or remove an adapter observation cutoff is not
approval to change these limits. Any budget change requires explicit user
approval before execution. This supersedes the budget-flexibility interpretation
in decision 0024 and the earlier family qualification notes.

The inbox attempt increased the trajectory allowance to 5,120 tokens, context
to 9,216 and turns to 12 during qualification, then carried those settings into
the campaign. That was an authorization error. Its E0 and failed E1 remain
preserved with their costs and reviews, separately from the corrected comparison.
The original read-table results are unchanged.

The user authorizes a complete inbox restart, including E0, at the original
limits if feasible on GPU 1. Use fresh `-c4096` run IDs at seed 4021, unchanged
question allocation and new base initialization for each arm. Verify the ordinary
training path before relaunch. Automatically harvest, review and advance only
E0 -> E1 -> E2 weight 0.1, then stop for the user's decision. A runtime or
integrity failure stops advancement; do not silently alter the protocol again.
The exact protocol and remaining observation/prompt differences are in
`docs/plans/inbox-family-extension.md`.
