# Use one optimizer update per rollout batch

Decided: 2026-09-11. Status: accepted.

New experiments set `training.num_iterations: 1`.

The `num_iterations: 2` probe reduced wall time per optimizer update by 35.2%,
but 50 updates then used only 25 fresh rollout batches instead of 50. At equal
fresh-question coverage it took more time, and the matched training-reward
contrast did not show a benefit. No final-checkpoint evaluation established
that reusing each rollout batch preserved or improved policy quality.

Fresh experience is the safer campaign baseline. Reuse may be reconsidered
only in a study that holds fresh rollout coverage constant and evaluates final
checkpoints with a predeclared non-inferiority margin.
