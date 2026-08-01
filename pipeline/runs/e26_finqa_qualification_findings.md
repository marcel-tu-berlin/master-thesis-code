# e26: FinQA qualification run (E1 task-success baseline)

Run: `e26-finqa-e1-baseline-qwen3-1_7b`, seed 42, Qwen3-1.7B + LoRA r16, naive_sum,
`env_reward` only. Train 300 steps in 8h08m (2026-07-31 21:23 -> 2026-08-01 05:34
UTC), eval 60 episodes, done 08:38 UTC.

## Verdict

FinQA does not qualify as the environment for the reward-bias-substitution study.
The instrumentation built for it does qualify.

## What the run measured

**Task success: 0/60 held-out episodes.** Accuracy 0.000, Wilson 95% [0.000, 0.060].
Training-side per-rollout accuracy over all 2400 rollouts was 1.0%, flat across the
run (first third 1.13%, middle 0.25%, last 1.63%). No learning trend.

**GRPO had a gradient on 10 of 300 steps.** `frac_reward_zero_std` averaged 0.967:
in 290 steps every rollout in the group scored 0, so the advantage was identically
zero and no weights moved. That is not a hyperparameter setting, it is what a 1%
success rate does to a group of 8 rollouts (probability all 8 miss is ~0.92).

**Off-target panel (60 episodes):**

| Metric | Value |
|---|---|
| non-termination rate | 0.867 [0.758, 0.931] |
| unsupported-claim rate | 0.000 |
| mean verification depth | 4.375 |
| mean steps | 3.45 |
| stop reasons | `hit_generation_cap` 34, `no_tool_call` 17, `env_done` 8, `max_turns` 1 |

All four tools were exercised: `get_table_info` 69 calls, `sql_query` 66,
`get_descriptions` 64, `submit_answer` 8. Trajectories are coherent and varied:
`get_descriptions -> get_table_info -> sql_query` appears 12 times,
`... -> sql_query -> sql_query` 10 times, and 8 episodes ran the full chain through
`submit_answer`. All 8 of those scored 0.0.

## Two blockers, both independent of the reward design

**1. The model cannot do the task.** FinQA is multi-hop numeric reasoning over
financial tables. The env scorer is generous (1% relative tolerance AND absolute
difference <= 1.0, with a percentage-point fallback and multi-value splitting), so
the eight submitted answers were wrong on the merits, not lost to formatting. At 1%
accuracy there is no specialized agent for an efficiency reward to shape, and no
task-performance dimension against which to detect a substitution trade.

**2. The per-turn budget binds.** 77% of training completions hit the 2048-token
turn cap (`clipped_ratio` mean 0.770). In eval, 34 of 60 episodes ended at
`hit_generation_cap`. This is the same truncation confound that invalidated the
e9-e21 sweep, reappearing on a new env: a completion truncated mid-thought never
emits a tool call, so it can never be scored correct. Blocker 2 partly causes
blocker 1, and raising the cap runs into the L4's 4096-token completion ceiling.

## What worked

The evaluation instrumentation from the protocol rebuild did exactly its job. Of 60
failures, the panel separates 34 budget artifacts from 17 behavioral non-responses,
1 turn-cap exhaustion, and 8 genuine wrong answers. Under the old code all 60 would
have been written as "wrong answer, accuracy 0.000" and the truncation would have
been invisible. The panel is env-agnostic and carries over unchanged to whatever
environment replaces FinQA.

## Environment criterion this establishes

An environment qualifies for this study only if it clears both bars at once:

- base-model accuracy in the 40-80% band, so GRPO gets within-group variance and
  there is headroom for the E1 baseline to specialize into
- at least two tools, so premature stopping, verification depth and unsupported
  claims are separable

FinQA clears the second and fails the first. reasoning_gym clears the first (the
poly campaign ran at 84-93%) and fails the second: one tool collapses the panel to
a single axis. The `repl` env is multi-turn with a task source we control
(`domains/repl/tasks.py`), so its difficulty is tunable, but its current task family
(sum/max/min of a list) is deliberately trivial and its tool surface is a single
`execute`.
