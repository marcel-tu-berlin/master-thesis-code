# Active experiment configs

Only `e0.yaml`, `e1.yaml` and `e2.yaml` are active. See the
[declared campaign](../../docs/plans/e0-e2-campaign.md) for commands, allocation
and analysis. E0 is evaluation-only; E1 and E2 independently start from the same
base model. E2 adds the relative successful-response length cost at weight 0.1.

E1 and E2 differ only in experiment metadata and `successful_length.enabled`.
No config loads an earlier adapter. Non-termination/E3 is deferred.

Earlier configs and the former template are in `archive/`; they are historical
records, not queued work. Do not launch a glob that includes E0 or the archive.
