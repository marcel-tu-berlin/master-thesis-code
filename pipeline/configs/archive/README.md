# Archived configs

Configs of finished or superseded experiments. Kept for the record; nothing here
backs a planned run. The live campaign configs stay one level up.

Two warnings before re-running anything from this directory:

- Configs predating commit `a19b1ff` state no `batch_size` and trained at 1;
  today they would resolve to 4, a different experiment under the same
  `experiment_id` (see "Void results" in the root CLAUDE.md).
- Every browsergym config before e30 trained under the `sequence_mask` ISR
  filter and none of their numbers carry over to the e30-era campaign.
