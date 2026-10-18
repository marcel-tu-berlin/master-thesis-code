# Tic-tac-toe seed 4230: E1 complete, E2 blocked

E1 completed 300 updates and all three held-out evaluations on 2026-10-10 at
04:09:20 UTC (06:09 Berlin), with exit code zero and a passing technical review.
The run retained the declared recipe and budgets.

| Policy/checkpoint | Wins / 200 | Success |
|---|---:|---:|
| E0 base | 8 | 4.0% |
| E1 update 100 | 61 | 30.5% |
| E1 update 200 | 76 | 38.0% |
| E1 update 300 | 85 | 42.5% |

Final E1 outcomes were 85 wins, 106 losses, four draws and five unfinished
episodes. Successful episodes averaged 1,677.35 assistant tokens. This is
task-only specialization evidence; no shaping effect can be assessed before E2.
The technical review records 136 zero-gradient updates and 416 observed rollouts.
Training took approximately 37 hours 58 minutes; the three evaluations added
approximately 12 hours. These are completed-run observations, not an E2 forecast.

## Blocker and evidence

E2 preflight failed at 04:09:26 UTC before creating its run directory.
`nvidia-smi` returned exit code 18 and reported a driver/library mismatch.
Read-only checks on 2026-10-11 reproduce it: NVML reports 580.178 while
`/proc/driver/nvidia/version` reports loaded kernel driver 580.173.02.
No host repair, reboot, run retry or configuration change was attempted.
The installation cause was subsequently established from APT history: Ubuntu's
`/usr/bin/unattended-upgrade` upgraded the NVIDIA packages from 580.173.02 to
580.178.04 on 2026-10-09, 06:32-06:34 UTC, during E1 training. The host has not
rebooted since 2026-09-10. The updated module is already installed for both the
running 6.8.0-139 kernel and the newer 6.8.0-146 kernel.

Fresh-process checks on 2026-10-11 distinguish CUDA computation from training
startup: CUDA initialization and a tiny PyTorch matrix multiplication pass on
GPU 1, but NVML raises `NVMLError_LibRmVersionMismatch`. The pinned vLLM selects
`UnspecifiedPlatform`, and its `DeviceConfig()` fails with `Failed to infer
device type`. Skipping the launcher guard would therefore not start the
unchanged E2 training stack. E1 had initialized vLLM before the driver update;
its subsequent evaluations use the Transformers/PyTorch path, which does not
require successful NVML initialization. A coordinated host reboot is the
standard recovery; no host changes were made during these checks.

The exact APT entry and short probes are saved as
`gpu-update-timeline-20261011.json`, `cuda-fresh-process-20261011.json`,
`vllm-fresh-process-20261011.json`, and `vllm-device-config-20261011.json`
in the campaign operations directory.

All 73 files bound by E1's remote technical review were rehashed successfully.
All 30 non-weight artifacts, including checkpoint evaluations at 100 and 200,
are now harvested and hash-verified. Checkpoint weights remain on the box.
The mirror's original unanchored checkpoint exclusion also skipped intermediate
evaluation children; the retained partial receipt has been superseded by a full
receipt. Correct that helper before restarting it.

The watcher last recorded successful status on 2026-10-08 and subsequently
logged SSH exit-255 errors; its delivery receipt still referred to E1 training.
The completion/blocker notification had not been delivered when the user asked
for status. The mirror's SSH connection recovered during this check.

The watcher and mirror are retired while the host is blocked. `RUNNING.md`
records E2 as blocked before launch, and task 9 remains open and blocked.
Continuation requires healthy GPU access and a controller restricted to the
unstarted E2 arm. The complete scientific comparison remains pending.

Detailed evidence and harvest receipts are retained under
`tic-tac-toe-campaign-s4230-ops/`, including `gpu-blocker-20261011.json` and
`e1-full-harvest-source-20261011.json`.
