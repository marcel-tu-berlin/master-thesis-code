# Standing decision: the campaign trains click-menu-2 only (C4, 2026-08-13)

Decided: 2026-08-13. Status: accepted. Moved verbatim from `LAB_NOTES.md` on 2026-09-04.


click-dialog-2 leaves the training mix and no family replaces it. The check
behind the decision, run against `browsergym_difficulty_correction.md`'s
fifteen families: the only mid-band candidate, click-checkboxes-large (0.60
greedy at a 4096 budget), has a 6066-token median trajectory and would train
truncated in 16 of 20 episodes - the e9-e21 artifact moved to the training
side - and every other family is saturated (>= 0.85) or dead-by-looping
(<= 0.10). Nothing both sits in the 40-80 band and fits the box. dialog-2
itself runs ~0.875 per sampled rollout, above the band, contributing
near-dead prompt groups; the B3 probe showed menu-2 owning the batch produces
the audit's first positive within-run slope. Consequences: the campaign
configs are menu-only (e30 onward, paired against `e0m-browsergym-base-menu`),
dialog-2 moves to the shifted eval split - as an untrained 0.80-base family it
now reads substitution in both directions instead of degradation only - and
single-family training narrows external validity to one family, which the
write-up must say. Selecting any new family later requires a sampled probe at
the training temperature first (greedy difficulty is not sampled difficulty).
