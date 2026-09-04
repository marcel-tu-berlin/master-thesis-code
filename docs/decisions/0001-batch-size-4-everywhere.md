# Standing rule: batch_size 4 everywhere

Decided: 2026-08-06. Status: accepted. Moved verbatim from `LAB_NOTES.md` on 2026-09-04.


Every arm of every comparison runs `batch_size: 4`, `n_rollouts: 8`, and the same
`max_steps`. Fixing only one of the three rebuilds the confound: a treatment arm
that sees more prompts, or more updates, than its baseline is not a reward
ablation. `grpo_runner.py` now defaults `batch_size` to 4 (a19b1ff) so a config
that omits it can no longer silently train at 1, and `train.py` prints the
geometry at startup so the value is in every log.

**`batch_size` is not the memory knob - `micro_batch_size` is.** Worth stating
plainly because the two get conflated. At `batch_size: 4`, `n_rollouts: 8`,
`micro_batch_size: 1`, a step is 32 completions pushed through the model **one at
a time** with `gradient_accumulation_steps=32`. Training-side memory therefore
never constrains `batch_size`; only vLLM's generation pool grows with it, and
that is capped by `gpu_memory_utilization`, so it queues rather than OOMs. 4 is
not a memory-forced half of 8.

**Why 4 and not 8.** Coverage gains almost nothing above 4 (0.46^4 = 4.5% dead
steps, 0.46^8 = 0.2%) and the step costs exactly double: the live poly run
measures 210 s/it at 32 completions, so 8 would be ~420. At a fixed prompt budget
a larger batch also buys gradient quality by spending optimizer steps - 600
prompts is 150 LoRA updates at bs=4 and 75 at bs=8, and this setup is
step-starved, not noise-starved. bs=1 was pathological; 4 is past the knee.

The evidence is asymmetric, which settles it. `bs=8` is measured only on
browsergym (`e27bs8probe`: 40/40 steps with gradient, 674 s/it, no OOM). `bs=4`
is measured only on poly (live, 210 s/it, 20.3 of 23 GB). `bs=4` on browsergym is
*dominated* - strictly fewer concurrent sequences than the bs=8 that already
passed - so it needs no probe. `bs=8` on poly is the one cell nothing backs: 64
sequences at 5120 tokens into 2.7 GB of vLLM headroom. Choosing 4 is the only
option that is measured or dominated everywhere it runs.

Remaining campaign at bs=4, from the 674 s/it measured at bs=8 on browsergym
(~340 s/it at bs=4): e27/e28/e29 are about 14h/arm at 150 steps, 43h for the
three, against 84h had they run at bs=8.
