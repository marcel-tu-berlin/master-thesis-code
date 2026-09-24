# Environment options after the E1 saturation warning

Research date: 2026-09-15. The user accepted the conditional sequence below:
finish and review the current feasibility study first; use this approach if
the current environment or family cannot support meaningful results. Candidate
families and migrations remain unqualified. No new GPU experiment or dependency
upgrade was performed for this analysis; decisions 0004/0010-0013 still apply.

2026-09-16 execution update: the completed Gate 4 review triggered this fallback.
The model-free screen is recorded in
[`family_oracle_findings.md`](../../pipeline/runs/family_oracle_findings.md).
Only read-table-2 survives under the checked interface and stack; its opt-in fill
tool is implemented. The affected readiness checks for decision 0014 passed
on 2026-09-17; the next model phase is the bounded read-table-2 screen. Evidence
is in [termination readiness review](../../pipeline/runs/termination_readiness_findings.md).
That screen subsequently stopped on a native action-error feedback defect;
the tested candidate verdicts and pending family choice are in
[the screen findings](../../pipeline/runs/family_screen_findings.md).
The candidate discussion below records the pre-screen rationale, not current
qualification status.

## Recommendation

Finish and review the declared E1 pilot. If its feasibility criteria fail,
qualify a small set of harder, text-observable MiniWoB families, including
tasks that need `fill`. This has the best prospect of preserving the verified
pipeline while adding meaningful decisions. Do not run another broad training
sweep or qualify alternatives merely because training reward is high.

If that bounded search fails, audit a small Agent World Model subset as the
first alternative already available in OpenEnv. A task-driven REPL is the
more controllable fallback, but requires us to own a task distribution and
weakens the variety of action-level measures. WebShop is a credible larger
change if a new OpenEnv wrapper is acceptable.

The unmeasured quantity for every new candidate is Qwen3-1.7B's sampled
performance and cost variation under our actual interface and budget. The
ranking below is based on source inspection and existing evidence; none of
these candidates has passed experimental qualification.

## What the warning establishes

The status check at 164/300 updates found 99.38% mean training reward over
the last ten updates, 95% constant-reward prompt groups, and eight updates
with zero gradient. These are training observations, not held-out accuracy.
The configured 500-instance pool is reused: 300 updates x four prompts is
1,200 prompt presentations, or about 2.4 passes. This makes memorization or
training-distribution saturation plausible, without establishing either.

Three kinds of headroom must be separated:

1. Task-reward variation lets E1 learn. A pooled success rate near 50% can
   still hide all-correct groups on easy instances and all-wrong groups on
   hard instances. Measure disagreement among eight rollouts of each task.
2. E2 needs correct solutions with different lengths and credible shorter
   paths. High task accuracy does not itself eliminate this opportunity.
3. E3 needs budget exhaustion to persist after E1, with within-group cost
   variation during sampled training. Voluntary stops are a separate outcome.

High baseline accuracy does not prevent detecting a success *drop* under
shaping. It limits improvement and often reduces task-reward variation.
Our below-90% rule is a declared suitability criterion for this campaign,
not a mathematical prerequisite for all efficiency research. Keep the
current Gate 4 rules unchanged; any narrower E2 study needs an explicit
protocol decision before testing shaped arms.

The accepted E0 reference gives menu success 67/100 and budget exhaustion
4/100. Tree gives 12/20 success and no budget exhaustion; dialog and transfer
give 18/20 each. Thus tree is worth retaining as an evaluation reference,
but its lower success alone does not establish E3 opportunity. Full counts
and intervals belong in [the E0 findings](../../pipeline/runs/gate4_e0_findings.md).

This interpretation follows thesis sections 4.3-4.5, 5.4 and 6.2: grounded
task reward, informative groups, and a cost that survives task-only training.

## What an OpenEnv upgrade would change

The latest published release checked was
[v0.4.2, released September 9](https://github.com/huggingface/OpenEnv/releases/tag/v0.4.2).
Our pin `024eedc90305cc8bd7a5b44f44d1b987102e957b` identifies itself as
`0.4.2.dev0`; inspected upstream main
`a3ee76444a8d1ec93a255db079bc848fc0540ad9` identifies itself as `0.4.3.dev0`.
These version strings do not replace commit-level provenance.

I compared GitHub tree blob IDs at those two commits. The Python files under
BrowserGym, Calendar, OpenApp, Agent World Model, Coding Tools and REPL are
unchanged. They are already present at our pin. Main adds ThinkingBox,
whose adapter explicitly describes itself as evaluation-only. The
[current catalog](https://github.com/huggingface/OpenEnv/blob/a3ee76444a8d1ec93a255db079bc848fc0540ad9/docs/source/environments.md)
does not contain a first-party ALFWorld or WebShop adapter.

An upgrade is reasonable if it fixes a demonstrated blocker. It is not
needed just to inspect these candidates, and upgrading OpenEnv alone does
not change BrowserGym's separately pinned reward conversion. After E1 is
harvested, use a separate pinned environment installation for any upgrade,
keep the completed run reproducible, and qualify the affected reset,
concurrency, reward, termination, token, and trainer contracts before another
training campaign. Do not combine an environment change with a TRL upgrade,
different loss, quantization, or model scale without a separate reason.

## BrowserGym: the best first search

Our adapter currently exposes `click` and `noop`; only `click` changes task
state. It cuts accessibility observations at 2,000 characters. This makes
typing tasks unreachable and can hide relevant content on longer pages.
The earlier search covered fifteen families under a much narrower interface
than the full MiniWoB catalog.

The [difficulty correction](../../pipeline/runs/browsergym_difficulty_correction.md)
also records obsolete per-turn evaluation budgets and greedy decoding.
Its long-trajectory and low-accuracy observations remain useful warnings,
but are not current qualification measurements. Do not reuse its apparent
winning family or its memory estimates as proof under today's stack.

### A scoring trap to exclude before spending GPU time

The installed BrowserGym 0.14.3 MiniWoB wrapper calculates
`float(RAW_REWARD_GLOBAL > 0)`. I compared the installed file byte-for-byte
with the [tagged upstream source](https://github.com/ServiceNow/BrowserGym/blob/v0.14.3/browsergym/miniwob/src/browsergym/miniwob/base.py)
and evaluated that exact expression: raw rewards -1, 0, 0.1, 0.5 and 1 map
to 0, 0, 1, 1 and 1 respectively.

In the pinned MiniWoB source, a wrong selection in
[`find-greatest`](https://github.com/Farama-Foundation/miniwob-plusplus/blob/eb59fed60fabe8951350275ba8650633b740013b/miniwob/html/miniwob/find-greatest.html)
can receive 0.1. The wrapper therefore reports success. Likewise,
[`click-checkboxes-soft`](https://github.com/Farama-Foundation/miniwob-plusplus/blob/eb59fed60fabe8951350275ba8650633b740013b/miniwob/html/miniwob/click-checkboxes-soft.html)
can give positive reward to a partially correct selection. Do not interpret
those reported successes as exact goal completion. This finding does not
establish an error in the current menu pilot; it blocks unqualified reuse
of those families. A scoring change would define a new measurement protocol.

### Proposed shortlist

All four initial families are registered in the installed BrowserGym 0.14.3
package and have HTML at the pinned MiniWoB commit. The ordering favors a
small interface change over a larger infrastructure change.

| Candidate | Why inspect it | Required work or main risk |
|---|---|---|
| `click-tab-2-medium` | Two tabs with textual links; a smaller search space than the failed hard variant. | Existing tools suffice in principle. Verify link identities appear in the actual accessibility observation and the complete task can be solved. |
| `click-collapsible-2-nodelay` | Search three collapsible text sections; removes the animation present in a previously poor family. | Existing tools suffice in principle. Removing animation may only remove timing noise; it does not prove learnability. |
| `read-table-2` | Read two table values, fill two fields, then submit. Exact all-fields correctness; a short sequence of distinct useful actions. | Add `fill` consistently to training and evaluation. Verify table/field relationships survive the observation limit. |
| `search-engine` | Enter a query, inspect results, select the requested result. Combines text entry, reading and navigation. | Add `fill`; possibly scrolling depending on actual observations. Pagination and long result lists can consume the budget. |

The task mechanics and reward branches were inspected in the
[pinned MiniWoB sources](https://github.com/Farama-Foundation/miniwob-plusplus/tree/eb59fed60fabe8951350275ba8650633b740013b/miniwob/html/miniwob).
These are candidates for a solvability audit, not claims of mid-band accuracy.
`read-table` is a useful easier control for `read-table-2`, not an additional
training candidate in the initial screen.

`email-inbox-delete` and `email-inbox-important` are reserves. They have
clear correct/wrong actions, but their source represents trash/star controls
as icon spans without explicit text labels. Inspect the actual tree before
using them with a text-only model. `email-inbox-noscroll` still samples reply
and forward tasks requiring typing; its name does not make it click-only.

Avoid visual counting/drawing tasks as an easy source of low accuracy.
For example, `number-checkboxes` asks the agent to draw a digit from a spatial
example. Also avoid `guess-number` as a headline distribution: its hidden
number is only 0-9, so new seeds do not create hundreds of distinct problems.

A fixed two-family mix is available through the current seed-to-family
mapping. First qualify its members individually, then inspect per-family
outcomes under the mix. A saturated family mixed with an impossible family
can produce a plausible pooled success rate and zero useful groups. Keep
family proportions fixed across E1 and shaped conditions. A previously
shifted family moved into training needs a new untouched shifted split.

## Other environments

| Option | Fit for the thesis | Assessment |
|---|---|---|
| Agent World Model | Database-backed tasks, multiple meaningful tools, inspectable state changes and redundant actions. | Best alternative already packaged in OpenEnv, but only after auditing a small subset. Difficulty at 1.7B is unknown. |
| Task-driven REPL | Controlled tasks, executable solutions and an exact final-answer verifier. | Most controllable fallback. Tasks and difficulty must be supplied; one execute tool makes the off-target panel narrower. |
| WebShop, text mode | Search, inspect, choose options and buy; partial matches and wrong purchases can support useful failure distinctions. | Credible external fallback; requires an OpenEnv adapter and isolated legacy dependencies. |
| ALFWorld, text mode | Navigation, object inspection and state-changing actions; explicit task goals. | Credible external alternative, but long action chains risk reproducing our budget problem. Requires a wrapper. |
| WorkArena / WebArena | More realistic browser workflows with task validators. | Broader validity, substantially more integration and operating cost. Poor first choice for fast iteration on 1.7B. |
| Coding Tools / Terminal-Bench 2 | Tests can ground success; reading, editing, testing and submitting expose useful off-target behavior. | Substantial task/runtime qualification; Coding Tools needs E2B and caller-supplied setup/verifiers. Terminal tasks risk a near-zero learning signal. |
| Calendar / OpenApp | Attractive interfaces and application state. | Current wrappers fail our task-reward requirement; an upgrade alone does not fix them. |
| Reasoning Gym / REPL-free math / games | Cheap or configurable task difficulty. | Useful controls, but often lack the action distinctions needed by the thesis. Reasoning Gym remains calibration-only under decision 0005. |

### Agent World Model: promising, but not a ready replacement

The [dataset](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K)
contains 1,000 scenarios with ten tasks each, SQL-backed state and verifier
code. A small set of short state-editing tasks could support both efficiency
and task-grounded behavioral review. This is an inference about suitability,
not a measured result for our model.

The paper reports experiments with 4B/8B/14B policies, a different history
handling scheme, and up to twenty interaction turns. Its quality study also
reports blocked tasks and implementation bugs. Its results cannot establish
that 1.7B will learn within our eight-turn, 4096-token trajectory budget.
See [the paper's setup and quality analysis](https://arxiv.org/html/2602.10090v3).

The [OpenEnv wrapper](https://github.com/huggingface/OpenEnv/blob/a3ee76444a8d1ec93a255db079bc848fc0540ad9/envs/agent_world_model_env/server/awm_environment.py)
requires several explicit choices:

- `seed` is accepted but does not generate a fresh task. Select and record
  actual scenario/task IDs; split by those IDs and initial state. Reusing
  ten tasks with different seed labels is not held-out evaluation.
- `verify` scores without ending; `done` ends without verifying and returns
  zero. The BrowserGym last-reward-wins bridge would lose a preceding score.
  Define scoring at termination and budget/voluntary stops explicitly, before
  cleanup; do not equate a successful verification call with task termination.
- Defaults include 0.1 for incomplete work and -1 for format errors. Those
  already shape behavior. E1 needs an explicit task-only reward policy.
- Prefer tasks whose code verifier can be audited against correct and wrong
  final states. Code-only scoring is not automatically a valid oracle, and
  using it differs from the paper's code-assisted LLM judging.
- Tool schemas vary by scenario. Keep a bounded, stable tool surface for the
  first subset, pin dataset/code assets, and prove all 32 rollout sessions
  have isolated state. The loader downloads data without a revision pin by
  default, and each session starts a subprocess.

For a source audit, two scenarios are enough. They are not a 500-instance
campaign distribution. Proceed only if an adequately sized, disjoint task
set can be qualified without inventing hundreds of synthetic variants.

### Why the superficially easy alternatives are lower priority

Calendar's [reward function](https://github.com/huggingface/OpenEnv/blob/a3ee76444a8d1ec93a255db079bc848fc0540ad9/envs/calendar_env/server/openenv_wrapper/mcp_env_environment.py)
still scores tool-execution status, not achievement of a specified goal.
OpenApp's [implementation](https://github.com/huggingface/OpenEnv/blob/a3ee76444a8d1ec93a255db079bc848fc0540ad9/envs/openapp_env/server/openapp_environment.py)
constructs a generic task whose validator always returns zero/unfinished;
`task_name` is reported as metadata, and the wrapper ends at its step cap.
The README's task-based positioning is insufficient evidence for our use.

[Coding Tools](https://github.com/huggingface/OpenEnv/blob/main/envs/coding_tools_env/README.md)
provides a useful tool interface but requires E2B and supplied setup/verify
commands. [Terminal-Bench 2](https://github.com/huggingface/OpenEnv/blob/main/envs/tbench2_env/README.md)
adds realistic tasks and runtime complexity; verifier isolation and exact
task images are part of validity, not incidental setup. Neither is an
obvious cheap way to get a learnable 1.7B baseline.

[WebShop](https://github.com/princeton-nlp/WebShop) has a text interface and
separate search/click actions, but requires a product index and Java in its
documented stack. Its graded score also needs an explicit full-success
definition; our generic 0.5 correctness threshold cannot be imported without
justification. [ALFWorld](https://github.com/alfworld/alfworld) can run text-only
without the embodied simulator, but still needs its assets and a verified
task-ID mapping. Neither is currently a first-party OpenEnv directory.

[WorkArena](https://github.com/ServiceNow/WorkArena) needs ServiceNow instance
access. [WebArena through BrowserGym](https://github.com/huggingface/OpenEnv/blob/main/docs/source/environments/browsergym.md)
requires website backends. ThinkingBox's new
[OpenEnv adapter](https://github.com/huggingface/OpenEnv/blob/a3ee76444a8d1ec93a255db079bc848fc0540ad9/envs/thinkingbox_env/README.md)
is evaluation-only. These are larger projects than qualifying a MiniWoB family.
FinQA retains its measured near-zero-learning rejection; broader difficulty
claims would require new evidence. Web Search explicitly supplies no task
reward and cannot serve as E1 without a task layer.

## A bounded next experiment, if this recommendation is accepted

1. Finish and review current E1 against its declared Gate 4 endpoint. Keep
   checkpoints 100/200/300; do not select an earlier checkpoint because it
   makes a family look more suitable. Determine E2 and E3 suitability separately.
2. Audit the four MiniWoB candidates without model training. On a few fixed
   instances, run a known correct path, a wrong terminal action, repeated/no-op
   actions and a reset after mutation. Verify complete observations and task
   identity, reward truth, budget handling and independent sessions. Reject
   unreachable or mis-scored tasks before sampling. Add only the required
   actions, then repeat the relevant readiness checks.
3. Screen at most four surviving families using four instances x eight
   sampled rollouts per family: 128 trajectories total, at the accepted
   training temperature and through the actual training generation path.
   Expand only the best two to sixteen independent instances each, for at
   most 320 screening trajectories overall. Measure per-group task/cost
   variation, both-success length variation, stop reasons, invalid actions,
   observation loss and runtime. These small samples are triage, not a pass
   certificate or a replacement for held-out evaluation.
4. Qualify only the leading family or fixed mix with a predeclared, short
   E1 diagnostic, for example thirty updates plus development evaluation.
   Check for early saturation, a learning floor and persistent target costs.
   Then run one full Gate 4 pilot if warranted. The diagnostic does not
   replace the final pilot endpoint. Use fresh development instances after
   selection; reserve final-test data. No lambda sweep at this stage.
5. If the bounded MiniWoB search fails, move to the AWM source/verifier audit
   before any GPU run. If its task pool or oracle cannot be qualified
   economically, choose between a task-driven REPL and a WebShop integration
   as an explicit thesis-scope decision.

Keep 4 x 8 geometry and the accepted reward/training recipe during selection.
A bigger training pool can test memorization in a separately declared pilot;
it does not guarantee harder tasks or persistent E3 cost. A larger model
could make presently impossible families learnable, but may saturate easy
ones faster. A smaller model could introduce a tool-use floor. Neither is
the first lever here.

Do not shorten the budget to manufacture difficulty. Increase a budget only
when valid solutions require it, with a measured memory/runtime check and
the same budget in all compared arms. IMPROVEMENTS.md's Liger option remains
blocked by decision 0011's failed loss/gradient parity; an OpenEnv upgrade
does not resolve that.

## Evidence and limits

Reviewed thesis.pdf, LAB_NOTES.md, BACKLOG.md, IMPROVEMENTS.md, decisions
0002/0004/0005/0010-0013, the readiness and Gate 4 protocols, historical
difficulty findings, current E0 findings, local adapter/reward code, installed
BrowserGym sources, pinned MiniWoB HTML, current OpenEnv source and primary
benchmark documentation. Source inspection used a fixed upstream commit;
no alternative environment was executed or benchmarked.

Verified the BrowserGym reward expression against the exact installed file
(SHA-256 `4d7b82b7b63403a9774969f4166ab858243bef381335ae1af3ced345e66552ab`)
and checked candidate registration against the installed package. Runtime
solvability, observation adequacy, model learning and new-environment
throughput remain unmeasured. All proposed new thresholds, counts and short
diagnostic budgets need freezing before their runs, not after their results.
