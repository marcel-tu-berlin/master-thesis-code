# OpenEnv patches

Local changes to the `meta-pytorch/OpenEnv` clone at
`training.env_server.repo_path` (default `/workspace/OpenEnv/envs`). The clone is
not a submodule, so without these files the patches exist only on the GPU box and
die with it.

Reapply after updating the clone:

```bash
cd /workspace/OpenEnv && git apply /path/to/openenv-patches/<name>.patch
```

## finqa-concurrency-and-seeding.patch

Base: `d372fab`. Two fixes to `finqa_env`, both needed for GRPO.

1. `create_app` gets `max_concurrent_envs` from `MAX_CONCURRENT_ENVS`. Shipped
   finqa omits it, so it defaults to 1 and the server closes every WS session
   after the first. The pipeline runs one server for many concurrent
   rollout-slot clients. reasoning_gym and textarena already wire this through.
2. `reset(seed=N)` selects the question by seed. Shipped finqa ignores `seed` and
   pulls from an unseeded shuffle, so rollout slots sharing a server get
   different questions and the GRPO group is not a group.

finqa was disqualified as a study environment by the e26 run
(`pipeline/runs/e26_finqa_qualification_findings.md`), so this patch is kept for
the record rather than for active use. Point 1 is the one worth remembering: any
new env must accept `MAX_CONCURRENT_ENVS`, and point 2 is the one worth checking:
any new env's `reset(seed=N)` must be a deterministic function of the seed.
