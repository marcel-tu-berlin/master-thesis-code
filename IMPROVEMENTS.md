# Deferred research options

These options are outside the active E0-E2 campaign. They are not prerequisites
or scheduled work. The previous menu-2/seed-42 sweep plan is superseded by the
current [campaign declaration](docs/plans/e0-e2-campaign.md).

- **Model scale:** an optional Qwen3-4B qualification and matched replication,
  with its own E0 and task suitability check. Existing 1.7B family conclusions
  need not transfer to another scale. Only measure memory/backend alternatives
  if a larger-model study is selected.
- **Quantization:** if needed for that study, hold precision fixed across its
  compared arms and verify trainer/sampler consistency. The existing
  token-truncate checks do not qualify a quantized trainer/vLLM pair.
- **Alternative lineage:** an optional Llama replication needs independent
  task/template qualification, its own E0 and a separately declared comparison.

Current recipe decisions already settle Liger, vLLM sleep, optimizer geometry,
budgets and reward composition. Their rejected alternatives are documented in
`docs/decisions/`; they are not open implementation tasks here.
