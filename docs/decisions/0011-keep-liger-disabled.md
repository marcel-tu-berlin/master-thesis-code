# Keep Liger disabled

Decided: 2026-09-11. Status: accepted.

New experiments set `training.use_liger_kernel: false`.

The 50-step Liger probe was 4.8% faster per update and did not show a clear
reward loss. It nevertheless failed the equivalence gate: the installed TRL
1.6.0 integration does not pass the accumulated-batch token denominator into
the fused DAPO loss. With micro-batch size one, the baseline and Liger paths
therefore weight trajectories differently and produce different gradients.
That is a training-objective change, not a transparent speed optimization.

Liger may be reconsidered only after the loss and gradient match the baseline
on identical inputs. A longer-context memory check comes after that parity
test, because the completed probe did not measure peak memory or establish that
a 16k completion budget fits.
