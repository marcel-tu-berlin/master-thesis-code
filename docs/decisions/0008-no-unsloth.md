# No unsloth; training stays on plain TRL

Decided: 2026-09-04. Status: accepted. Verified against source on 2026-08-24.

Unsloth is not used. The combination of unsloth with TRL's `environment_factory`
(the pipeline's only training path) is unsupported by both libraries, untested
anywhere, and would change the training semantics, not only the speed.

## Reasons

1. **No compatible version triple exists.** Unsloth declares
   `trl>=0.18.2,<=0.24.0` and `transformers<=5.5.0`; `environment_factory`
   exists only from TRL 0.29.0 (PR #5093, 2026-02-25) and requires
   `transformers>=5.2.0`. The pipeline runs TRL 1.6.0 / transformers 5.12.0.
   Installing unsloth next to the lock means `--no-deps`, outside the caps
   unsloth itself sets.

2. **The path is untested.** The strings `environment_factory`, `rollout_func`
   and `max_tool_calling_iterations` do not appear in unslothai/unsloth or
   unsloth-zoo (0 hits: GitHub code search plus downloaded `rl.py` /
   `rl_replacements.py`). Unsloth's only OpenEnv example (Wordle GRPO) uses the
   older `rollout_func` hook on trl 0.29.1; issue #3573 on catching up with
   current TRL is open.

3. **Unsloth changes the training semantics, not only the speed.** Its regex
   patches, checked against the installed TRL 1.6 (targets present): the TIS
   importance-sampling block is disabled (`vllm_importance_sampling_correction`
   patched to `if False`), `compute_loss` and the logprob computation are
   replaced by unsloth's own kernels, the PEFT ref-adapter block is removed,
   and `old_per_token_logps` is forced on under vLLM. Same objective, different
   implementation, different bugs: an unsloth run would be incomparable with
   every TRL-path number on disk. That is exactly the regression class this
   project exists to prevent (CLAUDE.md, "a regression is the worst outcome").

4. **vLLM 0.19.1 is unexercised by unsloth** (unsloth-zoo's engine glue
   branches up to 0.15).

## Consequence

Plain TRL stays the only training path. If the speedup ever becomes necessary,
it is a separate validation campaign (equivalence on the probe ladder), never a
drop-in swap.

## References

- Unsloth pyproject (version caps): https://github.com/unslothai/unsloth/blob/main/pyproject.toml
- PyPI metadata: https://pypi.org/pypi/unsloth/json
- Patch mechanics: https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl.py and https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl_replacements.py
- Issue #3573 (rollout_func / TRL support, open): https://github.com/unslothai/unsloth/issues/3573
- Issue #5673 (reward collapse under custom rollouts, fixed 2026-06): https://github.com/unslothai/unsloth/issues/5673
- TRL releases (`environment_factory` from v0.29.0): https://github.com/huggingface/trl/releases
- TRL GRPO / OpenEnv docs: https://huggingface.co/docs/trl/grpo_trainer and https://huggingface.co/docs/trl/openenv
- Unsloth OpenEnv Wordle example (rollout_func path): https://github.com/unslothai/notebooks/blob/main/python_scripts/Openenv_wordle_grpo.py
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
