# Mixed-inbox E1 memory failure, seed 4021

Run: `e1-email-inbox-noscroll-s4021`. Failed 2026-09-29 at 19:55 UTC.
Automatic advancement is paused. E2 has not launched. No retry or protocol
change was made.

## Confirmed failure

The native training loop raised `torch.OutOfMemoryError` in
`accelerator.backward(loss)` / `loss.backward()` before completing optimizer
update 1. The progress bar remained at 0/300. The worker exited 1 and the
controller recorded the failed exit. Both processes and the environment server
have exited, and the GPU is released.

The allocator reported a 2.90 GiB request with 2.19 GiB free on a 22.03 GiB
device. The process held 19.84 GiB, including 17.82 GiB allocated by PyTorch and
1.69 GiB reserved but unallocated. The error's `GPU 0` means logical CUDA device
0 after masking: launch evidence, the cost ledger and live process observations
confirm physical GPU 1. Physical GPU 0 was not used.

One pre-update batch was preserved: four questions with eight rollouts each.
The recorder's `optimizer_step: 1` labels the attempted update, not a completed
update; `global_step_before_update` is 0. There is no checkpoint, completed
training log, or evaluation report. These records are failed-attempt diagnostics,
not results from a trained policy.

The cost ledger contains one complete start/end pair marked failed:
430.36 seconds and 0.11954 allocated GPU-hours. All eight original run files were
harvested and verified against remote SHA-256 hashes. Source, stack, frozen
settings and pinned environment clones match the approved launch. No drift or
concurrent GPU process was found.

## What the qualification established, and what it missed

The earlier inbox qualification completed 32 native loss/backward shards and a
zero-LR optimizer step. Its model precision, 4x8 geometry, micro-batch 1,
5,120-token trajectory allowance, 9,216-token context, vLLM reservation 0.24,
gradient checkpointing, and disabled Liger/sleep settings match this attempt.
The qualification recorder captures extra tensors and uses different questions;
it is not identical execution to the ordinary training path.

Both batches reached 5,120 trajectory tokens. The failed batch had three capped
trajectories and mean length 2,889.56; qualification had four and mean length
3,769.69. Thus this evidence does not support explaining the failure simply as
longer completions. The immediate cause is exhausted peak backward memory; the
specific allocation/lifetime difference from the passing screen is unresolved.
The failed run did not capture per-shard tensors or an allocator snapshot, so an
exact causal attribution would require a separate diagnostic.

This failure refutes reliable full-campaign memory feasibility for the present
recipe. It does not invalidate the completed qualification observations or E0
evaluation. The retained read-table results are unaffected. No trained E1 result
exists to compare against E0 or E2.

## Pause boundary

The existing controller must reject E2 because E1 has no passing review. A
hash-bound failed review preserves the evidence and explicitly denies
advancement. The watcher is unloaded while a corrective decision is pending;
its installation remains available for an approved restart.

The recommended next unit of work is memory diagnosis and qualification on
physical GPU 1, preserving the approved scientific settings if possible. It must
verify the ordinary training path and adequate memory headroom before a fresh
E1 attempt. Micro-batch 1, gradient checkpointing and expandable segments are
already active. Changing precision, budgets, Liger or sleep behavior would need
an explicit protocol decision and appropriate equivalence/requalification work.
No such change or diagnostic GPU run is authorized by this failure review.

Keep this run immutable. Any approved retry needs a fresh run ID and recorded
authorization; do not overwrite this directory or amend its launch manifest.
Evidence: `e1-email-inbox-noscroll-s4021/` and
`inbox-campaign-s4021-ops/e1-failure-remote-evidence.json`.
