# Gate 4 observer verification

Verified on 2026-09-14 before E1 training. The observer saves 13 fixed batches
for the 300-step pilot, including updates 291-300. It computes the existing
diagnostic rewards and returns the original composed reward object unchanged.
It does not generate extra trajectories or update any model parameter.

## Evidence

- The CPU regression exercises all 300 update positions: exactly 13 complete
  batches are saved, with seed-aligned groups and variation retained among
  correct cosine rewards and budget penalties. Duplicate, incomplete,
  misaligned and pre-existing evidence is rejected.
- Installed TRL 1.6.0 dispatch and advantage/loss paths produce bitwise-identical
  prepared tensors, loss and selected-token log-probability gradients with
  observation off and on. The nonzero-gradient fixture includes a tool mask and
  active E3 cost. Python, NumPy and Torch CPU random states stay unchanged.
- A second check uses all 32 real saved E2 trajectories and the pinned Qwen
  tokenizer. All three diagnostic reward vectors exactly match the saved
  capture; native E1 rewards are identical before and after wrapping. Inputs
  and random states are unchanged. Recording that batch took 0.233 seconds.
  The injected step number 300 tests late capture; it is not an E1 training
  result.
- The full local lint, type and test gate passed. The native observer tests
  also passed in the deployed runtime. No extra GPU diagnostic run was needed.

The native gradient check holds generation and model forward fixed and compares
gradients with respect to selected log probabilities. Gate 3 already supplies
the live model-update evidence. This observation check does not establish
post-training cost variation; the E1 observations must still be reviewed.

## Source qualification

Gate 3 admitted source
`9cb8ac80183cb2329b2ec789d61d3349110b8e1af823ed4f2eb6439436a062cd`.
The runtime source with the observer is
`a7844327d2bde87ec9fa6e182455e943874b990b8796e9bb9a77a7c8a45ee34f`.

Every previously hashed file is unchanged except `training/train.py`; the only
new hashed file is `training/group_observation.py`. Removing the exact new CLI
argument, observer-wrapping branch and completion-check branch yields the
original training AST. The observer itself is covered by the checks above.
The old readiness report and source guard remain untouched; they should reject
the new manifest as a new source, rather than relabel old captures as new runs.
The qualification has a separate manifest and digest that include the verifier
and its regression test source as well as every runtime source. The pilot launch
checks that complete manifest against the deployed files and records both
runtime digests and the qualification digest. Editing the verifier or test makes
a previously generated qualification stale.

The locally checked qualification is
`2e7590c0ad5906d311e3d61c50a7c318ced2076ff6d1403ae58215b9c78d86e0`.
All deployed entries matched the locally reviewed files. The launch preflight
passed with this pinned digest and rejected a deliberately wrong digest before
creating a run or starting any GPU work. The approved manifest and launch
procedure are recorded in `docs/plans/gate4-pilot.md`.

The verifier lives in `probes/verify_group_observation.py`; it pins the original
training AST digest, so no ignored source fixture or temporary file is required.
Its JSON result is retained under `runs/gate4_observer_verification/`. From
`pipeline/` in the pinned GPU-box venv, with the saved Gate 3 artifacts present:

```sh
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m probes.verify_group_observation
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python -m pytest \
  tests/test_group_observation.py -q
```

These checks use CPU tensors and cached assets. Review the first live E1 batch
at startup, then require every planned observation and checkpoint evaluation
before the final Gate 4 verdict.

E1 startup on 2026-09-14 produced its first live observation: 32 records in
four seed-aligned groups, all diagnostic rewards finite, all composed rewards
equal to task reward, and all seeds inside the declared 500-question pool.
The process was training on physical GPU 1, with no failure marker in its log.
This closes the observer startup check, not the pilot's final feasibility gate.
