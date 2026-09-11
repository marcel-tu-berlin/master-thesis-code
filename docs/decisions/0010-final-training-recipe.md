# Final training recipe for new experiments

Decided: 2026-09-11. Status: accepted.

All new training configs use this recipe and state every value explicitly:

| Setting | Value |
|---|---:|
| `learning_rate` | `5e-5` |
| `kl_beta` | `0.0` |
| `optim` | `adamw_torch_fused` |
| `lr_scheduler_type` | `constant_with_warmup` |
| `warmup_ratio` | `0.1` |
| `weight_decay` | `0.1` |
| `temperature` | `1.0` |
| `vllm_importance_sampling_mode` | `token_truncate` |
| `rewards.compose_method` | `naive_sum` |
| `training.scale_rewards` | `none` |

The learning rate is the top stable point of the same-seed probe ladder. The
optimizer, scheduler and zero KL coefficient remove costs that did not buy a
meaningful signal under this short-horizon LoRA setup; decision 0007 records the
original rationale. `token_truncate` is the verified correction for the
length-correlated sampler mismatch. The P2 baseline then completed 50
browsergym updates with that optimizer recipe and no runtime error or numerical
failure.

`naive_sum` plus `scale_rewards: none` is a measurement decision. It keeps a
shaped reward's configured weight as its actual dose. Group scaling can cancel
that weight when task reward is constant within a prompt-group, and
`advantage_weighted` can silence the binary non-termination penalty in uniform
groups. The P2 probes used `scale_rewards: group`, so they did not empirically
validate `none`; this part is selected from the reward algebra and guardrail,
not from the P2 outcome.

This decision supersedes decision 0007 for future runs. It does not invalidate
or relabel old experiments. Their frozen configs and archived launch configs
remain the source of truth for the recipe that produced them.
