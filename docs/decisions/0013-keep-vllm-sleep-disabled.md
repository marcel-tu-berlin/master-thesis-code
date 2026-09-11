# Keep vLLM sleep mode disabled

Decided: 2026-09-11. Status: accepted.

New experiments set `model.vllm_enable_sleep_mode: false` and keep
`model.gpu_memory_utilization: 0.3` unless a separate memory probe justifies a
change.

The tested sleep configuration paired sleep mode with memory utilization 0.6.
It was 2.8% slower and lost 12.4 percentage points of paired training reward.
The installed TRL and vLLM call path reloads the original model after policy
weight synchronization, which is consistent with the growing sampler/trainer
probability gap. The probe therefore does not represent the trained policy
reliably.

Sleep mode may be reconsidered only after verifying that synchronized weights
survive every optimizer update and every tool turn. Any retest must hold memory
utilization fixed before testing a larger cache separately.
