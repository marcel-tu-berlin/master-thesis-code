# No unsloth; training stays on plain TRL

Decided: 2026-09-04. Status: accepted.
Initial source review: 2026-08-24. Follow-up source review: 2026-09-10.

Unsloth is not used in this pipeline. Its declared dependency bounds exclude our
pinned stack, and we have not validated its patches with our `environment_factory`
training path. This is a decision about this integration, not a claim that
Unsloth cannot train agents or work with OpenEnv.

## Reasons

1. **The declared dependencies conflict with our lockfile.** Unsloth 2026.9.4
   still declares `trl>=0.18.2,!=0.19.0,<=0.24.0` and
   `transformers<=5.5.0`. Our pipeline pins TRL 1.6.0 / transformers 5.12.0;
   `environment_factory` first appeared in TRL 0.29.0. Bypassing dependency
   resolution does not establish compatibility.

2. **The official example uses a different integration.** Unsloth's OpenEnv
   Wordle notebook uses TRL's `GRPOTrainer` with a custom `rollout_func`, not
   `environment_factory`. It installs TRL 0.29.1 with `--no-deps`, transformers
   4.56.2 and vLLM 0.15.1. This demonstrates an intended Unsloth/OpenEnv path,
   but does not validate ours. The absence of interface names in patch files
   is not proof of incompatibility; patches can operate on inherited TRL code.

3. **Training equivalence needs validation.** The original review found patches
   replacing loss/logprob computation, bypassing TRL's importance-sampling block,
   removing the PEFT ref-adapter block and forcing old-policy logprobs under
   vLLM. The follow-up confirmed that the importance-sampling block is still
   bypassed and Unsloth handles sampling logprobs separately. Different kernels
   alone do not prove a different objective or make comparisons impossible, but
   we cannot assume that our loss settings, masks and corrections retain their
   meaning. A switch would not itself invalidate historical runs; mixing training
   implementations in a reward ablation introduces another variable.

4. **Our generation stack is not covered by that example.** We pin vLLM 0.19.1;
   the Wordle notebook pins 0.15.1. Neither that difference nor the version
   branches inspected in engine glue prove that newer vLLM cannot work. They
   leave our combination unvalidated.

## Other environments and direct integration

OpenEnv does not inherently prevent Unsloth: the official Wordle example combines
both, and Unsloth itself uses TRL for GRPO. The relevant distinction is who owns
the episode loop. With `environment_factory`, TRL handles generation and tool
interaction; with `rollout_func`, our code handles them and returns trajectories.
TRL's factory can also wrap ordinary Python environments, so removing OpenEnv
while keeping the factory would not resolve the trainer compatibility question.

All three alternatives raised in the follow-up have direct Python interfaces:

| Environment | Direct interface | Implication |
| --- | --- | --- |
| ALFWorld | The text environment has batched `reset()` / `step(actions)`, returning observations, scores and done flags. | A custom rollout can drive the text version without the visual THOR environment. |
| WebShop | `WebAgentTextEnv` exposes text observations and search/click actions through Gym, using a simulated browser. | A custom rollout can drive it; product data and search setup are still required. |
| TextWorld | Its Python/Gym-like interface accepts commands and returns observations, scores and done flags. | A custom rollout can drive selected games, with explicit success criteria and episode limits. |

Feasibility is an inference from these interfaces and the Wordle example, not a
validated training result. The search found no official Unsloth online-GRPO
recipe for these three environments. Inference or supervised training on saved
trajectories would not establish that compatibility. Switching environments alone
does not remove the need for a different, validated training integration.

## Work introduced by switching to rollout_func

This inventory concerns the rollout switch, not installing Unsloth or implementing
a new environment. Much of the interaction logic already exists in our evaluation
loop and adapters. The main new piece is exact trajectory assembly for training.

| Responsibility | Code or integration we would own |
| --- | --- |
| Task identity | Carry the correct seed/task into the callback. It receives prompts and the trainer, while our seeds live in separate dataset columns; identical prompt text cannot identify a task. |
| Environment lifecycle | Allocate or reuse isolated rollout environments, reset with the right seed, and clean up on success or failure. Reuse the adapters and server manager. |
| Initial prompts | Append the reset observation, supply allowed tool schemas, and apply the native chat template without changing what the policy sees. |
| Multi-turn generation | Drive generation, actions and feedback using the configured sampling settings and remaining budget. Reuse generation helpers where compatible. |
| Parsing and dispatch | Parse reasoning/text/tool calls, validate names and arguments, execute multiple calls in order, and provide the intended error feedback. Reuse evaluation logic. |
| Conversation history | Preserve assistant reasoning and tool responses, and keep generation contexts consistent with the sequences used to compute training logprobs. |
| Budgets and stopping | Enforce trajectory/context/turn limits, including partial calls and oversized observations. Keep environment completion, voluntary stopping and budget exhaustion distinct. |
| Trajectory tensors | Return aligned `prompt_ids`, `completion_ids` and sampled-token `logprobs` across all turns, without dropping or duplicating tokens. |
| Loss masks | Build `env_mask` for model versus observation/inserted tokens, and verify that the selected Unsloth loss applies it correctly. |
| Reward bridge | Supply episode reward, termination state and structured messages to existing reward components; the factory's live-environment/message contract is no longer supplied automatically. |
| Group ordering | Return one result per requested rollout in its original group position, restoring order after any concurrent execution. |
| Failures | Keep invalid model actions separate from infrastructure failures; do not turn a broken server into reward zero or silently bias sampling through retries. |
| Diagnostics and checks | Preserve relevant rollout diagnostics and verify seeds, dispatch, contexts, token counts, masks, stopping reasons, rewards and evaluation parity on known trajectories. |

Two existing measurements must survive the switch unchanged. The cosine reward's
`model_token_count` counts selected assistant fields, including reasoning and
serialized tool arguments; replacing it with `sum(env_mask)` would change its
ruler. The current non-termination component penalizes budget exhaustion while
unfinished, not every unfinished episode: voluntary stopping remains separate.

GRPO updates, backpropagation, optimization, LoRA updates and checkpoint saving
remain trainer responsibilities. The inspected TRL 0.29.1 custom path also
retains dataset repetition into prompt groups and vLLM weight synchronization.
These need checks under Unsloth, not fresh implementations. Batching active
episodes for throughput can follow a correct implementation; it is not necessary
for the first compatibility probe.

The 2026-09-10 follow-up inspected local code, documentation, package metadata and
upstream source. It did not install the alternative stack or run GPU validation.

## Consequence

Plain TRL stays the only training path. If the speedup ever becomes necessary,
it is a separate validation campaign (equivalence on the probe ladder), never a
drop-in swap. Do not change environments solely to obtain Unsloth. First validate
the alternative integration on a controlled task, including our rewards and token
accounting, then measure its performance. This follow-up changes no experiments
or measurements and invalidates no existing results.

## References

- Unsloth pyproject (version caps): https://github.com/unslothai/unsloth/blob/main/pyproject.toml
- PyPI metadata: https://pypi.org/pypi/unsloth/json
- Patch mechanics: https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl.py and https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl_replacements.py
- Issue #3573 (historical custom-rollout support report, still open at follow-up; not proof of current universal failure): https://github.com/unslothai/unsloth/issues/3573
- Issue #5673 (custom-rollout masking and reward-collapse report, closed at follow-up): https://github.com/unslothai/unsloth/issues/5673
- TRL releases (`environment_factory` from v0.29.0): https://github.com/huggingface/trl/releases
- TRL GRPO / OpenEnv docs: https://huggingface.co/docs/trl/grpo_trainer and https://huggingface.co/docs/trl/openenv
- Unsloth OpenEnv Wordle example (rollout_func path): https://github.com/unslothai/notebooks/blob/main/python_scripts/Openenv_wordle_grpo.py
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- TRL 0.29.1 trainer (callback contract, masking, grouping and weight sync): https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py
- TRL 0.29.1 generation helper: https://github.com/huggingface/trl/blob/v0.29.1/trl/experimental/openenv/utils.py
- ALFWorld interface: https://github.com/alfworld/alfworld
- WebShop interface: https://github.com/princeton-nlp/WebShop
- TextWorld interface: https://github.com/microsoft/TextWorld
