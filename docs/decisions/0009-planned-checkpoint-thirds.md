# 0009: Planned checkpoint observations at training thirds

Date: 2026-09-10

Future campaigns may set `eval.checkpoint_schedule: thirds`. The schedule derives
steps from `training.max_steps`, which must be a positive integer divisible by 3.
The save interval is `max_steps // 3`; a conflicting explicit `training.save_steps`
is rejected. The frozen training config records the resolved interval.

Report all three observations: one third, two thirds, and the final step. No best
checkpoint is selected by success, efficiency, or any other outcome. Retain all
periodic checkpoints; the final evaluation still loads `checkpoint-final`.

Every observation uses the same held-out and shifted instances, seed mapping,
generation budget, decoding settings, and reference thresholds. Scheduled eval
requires the existing `eval.reference_report` with samples for every split, so
thresholds cannot drift with the policy. The saved evaluation protocol checks
the config and reference data on resume.

Checkpoints within one run are repeated measurements of one training seed. The
training seed remains the experimental unit in multi-seed analysis. Compare
conditions across seeds separately at each planned step; never pool checkpoints
as seed replicates. The dose analysis rejects mixed steps or repeated run roots.

E0 is a fixed untrained reference, evaluated once. `--base-model` rejects the
checkpoint schedule. Configs without the field retain final-only evaluation and
their existing save interval and output locations.

Historical e30-e36 configs and results are unaffected. Their intermediate
checkpoints were not harvested and cannot be reconstructed locally. This decision
does not authorize reinterpreting those results as checkpoint observations.
