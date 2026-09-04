# Standing rule: recipe defaults since 2026-08-24

Decided: 2026-08-24. Status: accepted. Moved verbatim from `LAB_NOTES.md` on 2026-09-04.


Three `grpo_runner` defaults changed on 2026-08-24, none of them a measurement,
all of them the optimizer recipe a new run trains under:

| Knob | Before | After | Why |
|---|---|---|---|
| `optim` | `paged_adamw_8bit` (hardcoded) | `adamw_torch_fused` | LoRA r=16 on Qwen3-1.7B is ~17M params, so fp32 Adam state is ~140 MB. 8-bit paging saved nothing and quantised the moments of exactly the parameters being trained. |
| `lr_scheduler_type` | `cosine` (hardcoded) | `constant_with_warmup` | Over 150 steps cosine spent the last ~40 steps near zero LR; the average LR was about half the stated one. The setup is step-starved, so every step should carry the full LR. |
| `kl_beta` | 0.001 | 0.0 (TRL's default) | At 0.001 the KL term was ~1e-4 of the loss (KL ends near 0.1) while `beta > 0` costs a full ref-model forward every step. |

`optim` and `lr_scheduler_type` are config keys now, so the frozen config records
them. **The seed-42 browsergym campaign (e30-e36) trained under the old recipe**,
and its `configs/` copies pin all three old values explicitly, so a re-run of
any of them reproduces the run on disk. The frozen `runs/<exp>/config.yaml`
copies written before this date do not carry the two new keys; a re-run from a
frozen copy would resolve to the new defaults. Use the `configs/` copy.

**The old-recipe campaign is closed (decision 2026-08-24).** The seed-43
replication of e0m/e30/e31/e32/e36/e33 is dropped, not deferred: everything
from here runs on the improved pipeline, and no seed replication of anything
is planned until the new setup produces a first result worth replicating.
Nothing already on disk is invalidated - no arm is compared across the recipe
boundary - and e30-e36 stay citable as the single-seed campaign they are.

Also closed the same day: `model.*` had no key whitelist (`training.*`,
`rewards.*`, `eval.*` all did), so `model.lora_rnk: 8` validated, trained at the
registry rank, and froze the 8. `_KNOWN_MODEL_KEYS` rejects it now.

**Phase-2 knobs (A/B, not defaults).** Three more passthroughs landed the same
day, all at TRL's defaults unless a config sets them, so no run changes without
a key in its frozen config: `training.num_iterations` (mu, optimizer steps per
rollout batch), `training.use_liger_kernel` (fused linear GRPO loss; BACKLOG 1)
and `model.vllm_enable_sleep_mode` (the colocated engine sleeps at level 2
during the policy update, so `gpu_memory_utilization` can exceed what the
backward leaves free; the runner drops `expandable_segments` for it, which
torch cannot combine with vLLM's memory pool). Each passed a browsergym
`--smoke` with train + eval on 2026-08-24; the 50-step A/B against
`probe-p2-base` (seed 42, new recipe, e30's task and geometry) is what decides
whether any of them becomes a default. Read with `python -m probes.p2_compare`.

Where the step goes, finally measured: the `num_iterations: 2` smoke logged a
generation step at 50.6 s and the reuse step that follows it (policy update
only, same batch) at 5.3 s. The update is about a tenth of a step; the rest is
rollout - vLLM sampling plus eight synchronous playwright turns. So a kernel
speedup (liger, unsloth) can move at most that tenth, `num_iterations: 2` buys
a second optimizer step for about +10% wall clock, and the rollout side
(sleep mode -> more concurrent sequences) is where step time lives.
