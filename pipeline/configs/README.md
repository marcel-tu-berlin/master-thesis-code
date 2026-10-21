# Active experiment configs

The current authorized batch is the five weight-0.1 `e2-placebo-*.yaml` configs
listed in [the placebo protocol](../../docs/plans/placebo-l010-campaign.md).
`e2-placebo-l020.yaml` remains a deferred historical preparation and is excluded.
Use the explicit host queues, never a filename glob. The E0/E1/E2 campaigns
described below have completed; retain their configs as the original recipes.

The initial campaign used `e0.yaml`, `e1.yaml` and `e2.yaml`. See the
[declared campaign](../../docs/plans/e0-e2-campaign.md) for commands, allocation
and analysis. E0 is evaluation-only; E1 and E2 independently start from the same
base model. E2 adds the relative successful-response length cost at weight 0.1.

E1 and E2 differ only in experiment metadata and `successful_length.enabled`.
No config loads an earlier adapter. Non-termination/E3 is deferred.

Earlier configs and the former template are in `archive/`; they are historical
records, not queued work. Do not launch a glob that includes E0 or the archive.
