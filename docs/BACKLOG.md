# Backlog

Deferred work, with enough context to pick up cold.

## Fork-masking entropy experiment (deferred 2026-06-10)

Dropped from the 5-config matrix during the slim pass. Not for any technical
defect — it was a matrix-scope call. Two ways to bring it back, both wanted:

### Option 1 — restore the reward-side fork-mask config

Recreate `pipeline/configs/e1-token-entropy-forkmask-qwen-7b.yaml`: identical to
`e1-token-entropy-qwen-7b.yaml` plus `rewards.token_entropy.fork_mask_top_frac: 0.2`.

- The reward code (`TokenEntropyReward.fork_mask_top_frac`) and the schema
  whitelist entry are still in the tree, so this is config-only — it works
  immediately, no code change.
- It is a genuine, distinct signal: `fork_mask_top_frac` averages the entropy
  bonus over only the top-X% highest-entropy tokens, a *shape* change (not a
  global scalar), so it survives the `advantage_weighted` per-group z-scoring
  that nullifies `reward_scale`.
- Question it answers: does concentrating the entropy bonus on high-uncertainty
  "forking" tokens beat rewarding mean entropy across all tokens?

### Option 2 — faithful Wang-2025 gradient masking

The current `fork_mask` masks the **reward** (which tokens' entropy feeds the
per-completion reward). Wang-2025 masks the **gradient** over forking tokens.
So Option 1 is *inspired by*, not equal to, the paper's mechanism.

A faithful version intervenes at the policy-gradient level (mask or down-weight
the gradient contribution of non-forking tokens during the GRPO update), not in
a reward function. That is a TRL-trainer-level change, not a config toggle —
needs its own design pass. Do Option 1 first; it is the cheap, comparable arm.
