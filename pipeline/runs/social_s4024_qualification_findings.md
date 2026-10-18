# Social-media family qualification, seeds 4024/4025

Date: 2026-10-07. User-authorized qualification only; no E0/E1/E2 campaign is authorized.

## Protocol

Qwen3-1.7B at the existing pinned revision; visible HTML with click, noop and opt-in scroll. No fill tool is needed. Prompt allowance 4,096 tokens, whole-trajectory allowance 4,096 including tool feedback, model context 8,192, and eight tool-calling turns. No pipeline semantics or existing campaign configs changed.

Two separate native base-policy captures use four predeclared questions and eight samples per question. The existing one-step warmup method exercises loss/backward at zero actual update LR. Unchanged trainable-parameter hashes must be verified after each capture. This is a preliminary screen, not a learning curve.

## Native controls

All 32 native adapter controls passed: correct selection, one omitted target, an extra wrong-action selection, and correction of that wrong action before submission, on each of eight fixed questions. Correct and recovered submissions received binary reward 1; deliberate errors received 0. Selection-state visibility, deterministic reset after mutation, and the terminal guard passed. Twelve additional source-level controls checked the exact pinned scorers, including wrong-user and excess-selection cases; these supplement the native tests.

| Variant | Question seed | Posts | Matching posts | Required selections | Actions | Scrolls | Initial prompt tokens | Preterminal feedback tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 4024000000 | 10 | 6 | 6 | 14 | 7 | 1017 | 6544 |
| all | 4024000001 | 11 | 5 | 5 | 13 | 7 | 1029 | 6140 |
| all | 4024000002 | 10 | 8 | 8 | 15 | 6 | 1039 | 7245 |
| all | 4024000003 | 10 | 2 | 2 | 10 | 7 | 1036 | 4502 |
| some | 4025000000 | 9 | 5 | 1 | 8 | 6 | 1027 | 3503 |
| some | 4025000001 | 6 | 6 | 3 | 7 | 3 | 1000 | 2881 |
| some | 4025000002 | 10 | 2 | 2 | 10 | 7 | 1035 | 4553 |
| some | 4025000003 | 11 | 1 | 1 | 9 | 7 | 1038 | 4177 |

These feedback counts describe the recorded scripted paths. They omit assistant output and chat wrappers, so they are not complete trajectory costs or proven minimal solution costs. Scripted actions are not minimum model-turn counts: independent calls may be batched. Six of eight correct paths exceed the trajectory allowance on feedback alone. The two cheaper paths are social-media-some questions.

## Model screens

The social-media-all capture completed in 672.75 seconds (0.18688 allocated GPU-hours). It solved 0/32 rollouts, with 0/4 task-active groups and 0/4 successful-length-active groups. No episode terminated; 14/32 recorded completion sequences reached exactly 4,096 tokens. Recorded assistant messages per episode ranged from one to three, so none approached the eight-turn limit. The capture does not persist an explicit stop-reason field; shorter recorded sequences are not automatically classified as voluntary stops.

The policy made 85 click calls and six noop calls, with zero scroll calls. The exact rendered prompt was decoded from the saved trainer prompt IDs and contains the scroll tool. One recorded example (reward-batch index 0, seed 4024000001) clicks the correct Like control, receives a visible active selection, then clicks it again and receives an inactive selection. This is a state-grounded deselection event, not evidence of reward-bias substitution in an untrained base policy.

All 32 loss shards and the accumulated gradients were finite. Trainable-parameter hashes were identical before and after the warmup step. The task gradient norm was zero, consistent with constant zero reward in every group. The original-budget native backward path completed without an out-of-memory error for these captured inputs.

The social-media-some capture also completed with unchanged weights, 32 finite loss shards, finite gradients and no out-of-memory error. Its gradient norm was zero. The policy made 92 clicks and 11 noops, with no scrolling calls. Its one to five recorded assistant messages per episode stayed below the eight-turn ceiling.

| Variant | Success | Environment termination | Task-active groups | Length-active groups | Recorded sequences at cap | Sampled peak GPU memory | Allocated GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| social-media-all | 0/32 | 0/32 | 0/4 | 0/4 | 14/32 | 20,692 MiB | 0.18688 |
| social-media-some | 0/32 | 0/32 | 0/4 | 0/4 | 14/32 | 20,738 MiB | 0.18406 |

GPU memory was sampled once per second, beginning during the first capture's generation; these are observed peaks, not exact maxima. Total allocated GPU cost was 0.37093 device-hours. Both captures used the same runtime source hash, frozen recipe and pinned model/tokenizer revision. The first and second native captures took 672.75 and 662.60 seconds respectively, excluding the CPU-only scripted controls.

## Assessment

Neither variant is admitted for an E0/E1/E2 campaign under this interface and budget. All sampled task rewards were zero, so E1 has no within-question learning signal. There are no successful peers from which E2's relative successful-length component could produce a signal. A longer run with this observed distribution is unsupported by the qualification evidence.

There are two distinct obstacles:

1. Policy behavior: the base model never used the available scroll tool and never submitted, even on the two cheaper scripted some paths. It spent its budget on the initial viewport, repeated actions and generated text. More available memory alone does not demonstrate that this behavior will improve.
2. Trajectory cost: six of eight correct scripted paths already exceed 4,096 tokens on preterminal feedback alone. The other two leave only limited room for assistant output and chat wrappers. These are measured paths rather than proven optimal costs, but they expose substantial budget pressure.

This is a small, fixed development sample of four questions per variant. It does not prove that the families are generally unlearnable. The native scorer and correction controls demonstrate that the tasks offer meaningful observable error categories, but the current base policy does not reach enough successful trajectories to study reward-bias substitution through this recipe. The untrained base-policy errors themselves are not evidence of substitution.

Both original-budget native backward checks fit the present L4. These results therefore do not establish a need for a larger GPU as the next intervention. Longer-context training remains untested here, and the earlier flight/inbox memory evidence remains separate.

Recommended next decision: a separately declared diagnostic with a more concise visible-state observation, preserving visible content, action IDs and selection state, then repeat the fixed screen. If exploration remains absent, consider a stronger base model. Neither change, a larger token budget, nor a fallback family has been executed. Full campaign training remains deferred.

## Evidence

- `social-qualification-s4024-ops/plan.json`: declared scope and geometry.
- `social-qualification-s4024-ops/manifest.json`: exact config and helper hashes.
- `social-qualification-s4024-ops/native-controls/`: native event traces, environment stamp and runtime source hashes.
- `social-qualification-s4024-ops/check_source_scoring.js` and `source-scoring-check.json`: runnable supplemental scorer check.
- `social-qualification-s4024-ops/*-screen.yaml`: sampled-screen configurations.
- `social-qualification-s4024-ops/qualification-summary.json`: verified aggregate results, integrity checks and costs.
- `family-screen-social-media-{all,some}-s{4024,4025}-c4096/`: frozen configs, native rollout/loss/gradient captures, parameter hashes, screen reviews and cost records. Adapters remain on the GPU box.

## Verification

The project gate passed: formatting, lint, mypy, 492 tests passed and seven skipped, plus all 17 setup-harness checks. The temporary oracle passed its explicit format/lint checks and all 32 native controls. No application code was edited for this qualification.
