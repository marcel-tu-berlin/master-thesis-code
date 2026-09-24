# Development evidence through 2026-09-24

On 2026-09-24, all earlier top-level run artifacts were moved here on both the
Mac and GPU server. No file contents were changed. The local move preserved
1,440 entries across 134 top-level entries; the remote move preserved 1,760
entries across 65 top-level entries. Inodes, sizes, modification times and modes
were compared before and after the moves. The inventories differ because
checkpoints stay on the server and older harvested evidence stays on the Mac.

Historical references resolve as follows:

```text
pipeline/runs/<old-path>
  -> pipeline/runs/archive/development-2026-09-24/<old-path>
```

Frozen configs, reports, receipts and historical plans retain their original
paths and hashes. They are evidence, not launch instructions for the current
checkout. Restore a historical layout in an isolated evidence copy if an old
analysis script requires it; never overwrite new run outputs to do so.

Findings and batch summaries are versioned. Raw logs, trajectories, review
artifacts and adapters remain ignored by Git and are preserved where they were
stored. This archive is not a new backup of server-only checkpoints.

Key development records:

- [Relative continuation](table_compression_s4009_findings.md)
- [Original cosine comparison](table_first_s4009_findings.md)
- [Reward audit](table_first_s4009_reward_audit_findings.md)

These are development results. The relative continuation does not establish
the outcome of relative shaping from the original base model.
