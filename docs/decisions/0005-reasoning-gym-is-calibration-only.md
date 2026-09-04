# Standing decision: no poly re-runs; reasoning_gym is calibration only (2026-08-17)

Decided: 2026-08-17. Status: accepted. Moved verbatim from `LAB_NOTES.md` on 2026-09-04.


The thesis (section 6.4) fixes reasoning_gym's role: method development and
difficulty calibration, explicitly not a source of headline results. The
headline efficiency claims live on browsergym, where e31 delivered the clean
compression result the poly campaign never could. Consequences: the
e24bs4/e25bs4 seed-43/44 replication is dropped (its single-seed CIs all
crossed zero anyway, and the pair trained under the `sequence_mask` ISR filter,
so a straight replication would replicate a filtered experiment), and no poly
compression claim goes in the write-up - e24/e25 numbers are citable only as
method-development history with their caveats attached
(`pipeline/runs/e24_e25_4k_pair_findings.md`). Do not re-add a poly arm without
a thesis-level reason.
