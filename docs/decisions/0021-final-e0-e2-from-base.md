# Separate from-base reward shaping from competent-policy continuation

Decided: 2026-09-24, explicitly selected by the user after decision 0020.

The first final campaign has exactly three conditions:

- E0: evaluate the original Qwen3-1.7B base policy, with no adapter or training.
- E1: train from that base using task success only.
- E2: train independently from the same base using task success plus the
  relative successful-response length cost, weight 0.1, from the first rollout.

E1 and E2 share the model revision, initialization seed, training questions,
optimizer, 300 updates, 4 x 8 geometry, budgets and evaluation questions. Their
only reward difference is `successful_length.enabled`. Neither loads an E1
adapter. Keep `compose_method: naive_sum` and `scale_rewards: none`.

The accepted reward, written in TUI-readable notation, is:

```text
S = successful rollouts for the same question
mu = mean assistant-token count within S
sd = population standard deviation of token counts within S
z_i = (L_i - mu) / max(sd, 1 token)
sigmoid(z) = 1 / (1 + exp(-z))

C_i = sigmoid(z_i)  if i is in S; otherwise 0
R_E1(i) = R_task(i)
R_E2(i) = R_task(i) - 0.1 * C_i
A_i = R(i) - mean reward of the full eight-rollout question group
```

Use the existing assistant-token measurement and correctness threshold; no
measurement changes. An empty S contributes no length cost. Equal successful
lengths, including a singleton S, give cost 0.5: correctness can still distinguish
success from failure, but successful responses have no relative length contrast.
Uniform total rewards give zero policy-gradient advantages. There is no added
absolute length penalty, fallback reward or gradient-history collection.

Decision 0019's competent-policy continuation remains a separate future design
option, recorded in LAB_NOTES.md. Its development result motivated selecting the
relative reward; it is not evidence that training with this reward from the base
will have the same outcome. Linear remains an optional future comparison.

Defer non-termination training (E3) until E0-E2 are complete and reviewed. No E3,
placebo, weight sweep, extra model or family is included in the active configs.
Keep all earlier run artifacts and configs in the development archive; do not
change their contents or scientific status. No more readiness experiments are
required by this decision, and preparation does not launch the final campaign.

Concrete allocation, commands and analysis: [E0-E2 campaign](../plans/e0-e2-campaign.md).
