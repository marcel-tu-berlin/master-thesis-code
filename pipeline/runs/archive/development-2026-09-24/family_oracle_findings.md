# Harder-family oracle screen - 2026-09-16

Run artifacts: `family-oracle-s4003/`. Model-free checks only; no policy weights
were loaded and no training or sampled model evaluation was performed.

## Verdict

`read-table-2` is the only shortlisted family ready for a model screen.
Episode-boundary qualification passed on 2026-09-17; see
[termination readiness review](termination_readiness_findings.md).
It needs the new explicit `enable_fill: true`
tool interface. This is evidence that the task is observable, solvable and
correctly scored on the checked instances, not that Qwen3-1.7B has useful
learning headroom or that E3 cost will survive training.

| Family | Observed oracle result | Decision under the current interface and stack |
|---|---|---|
| click-tab-2-medium | Two instances award success for switching to an empty second tab without clicking the requested link. The third needs a target span whose ID is absent from the accessibility text. | Reject: stated goal and reward diverge, and the remaining path is not grounded in the model's action identifiers. |
| click-collapsible-2-nodelay | Correct/wrong terminal rewards behave as expected, but the clickable target spans have no IDs in the accessibility observation, even after expansion. | Reject under the current observation contract. The oracle used privileged HTML IDs, unavailable to the policy. |
| read-table-2 | All three correct paths succeed; all three paths with one incorrect field terminate with zero reward. Field labels, values and input IDs are visible. | Survives this oracle screen; needs fill and then sampled model qualification. |
| search-engine | All three correct paths earn raw MiniWoB reward 1 but are returned as terminal reward 0. | Reject on the pinned stack until the wrapper's URL validation is corrected and requalified. |

No MiniWoB/OpenEnv/package source was modified and no dependency was upgraded.
The first two families should not consume GPU time under the unchanged
accessibility-only interface. A deliberate observation change could reopen
collapsible tasks. Search Engine needs a wrapper correction, not a harder model.
Prefer screening the surviving table task before taking on either change.

## Design and checks

The plan was written before reset: four candidate families, each with development
seeds 4003700000-4003700002 (seed block 4003, offset 700000). These are not reserved
final-test instances. The same three seeds were used across families, without
claiming cross-family pairing of task content. Evidence includes complete initial
and resulting accessibility text, privileged HTML for oracle diagnosis, action
arguments, rewards, termination and error fields.

All twelve instances reproduce their goal and accessibility tree after an
intervening different reset. A second simultaneous client does not change the
first client's state; noops leave reward zero and the episode open. Across the
24 correct/wrong paths, the largest accessibility text is 678 characters, below
the production 2000-character limit. Missing link IDs in tab/collapsible tasks
are therefore not a truncation problem.

Table paths need three useful actions: fill two fields, then submit. Their wrong
controls fill one field correctly and the other incorrectly. Native adapter
replays additionally verified two consecutive noops, correct/wrong outcomes,
the post-terminal reward guard, and identical reset after the mutated/finished
episode. All six replays passed. They fit the existing eight-turn cap without
shortening the budget to create difficulty.

The typing adapter reuses the OpenEnv action builder and the existing `_act`
path. Native Transformers schema extraction found identical ordered schemas for
training's exposed methods and evaluation's tools: click, fill, noop. Omitted or
false `enable_fill` preserves the old click/noop tool set; non-boolean settings
are rejected. The tool addition changes context only when explicitly enabled.
It does not fix the separate training/evaluation termination discrepancy.

## Search Engine diagnosis

The correct target links use `href='#'`. Clicking one changes the page URL from
`search-engine.html` to `search-engine.html#`. BrowserGym 0.14.3's MiniWoB
validator rejects any URL different from its initial task URL before reading
the reward globals. Direct native inspection on all three correct paths found:

```text
WOB_RAW_REWARD_GLOBAL = 1
WOB_DONE_GLOBAL = true
reported reward = 0
task_info.error = "invalid url, terminating task"
```

This is a confirmed wrapper/task mismatch. The OpenEnv observation's ordinary
action-error field is empty, so aggregate model failure rates would hide it.
The direct inspection followed the same action arguments recorded through the
server. Wrong controls also report zero, but cannot validate reward fidelity
when correct actions are rejected by the wrapper.

## Evidence and limits

`plan.json`, `snapshots.jsonl`, `actions.jsonl`, `search_diagnosis.jsonl`,
`adapter_check.jsonl`, `fill_tools.json`, scripts and logs are retained together.
The native stack remains the pinned OpenEnv/BrowserGym/MiniWoB installation;
`provenance.json` records exact versions, source hashes and artifact hashes.
The local gate passed formatting, lint, types, 417 tests with five existing
optional-stack skips, and the setup harness. Native adapter/schema checks cover
the new tool bridge but do not replace a real trainer readiness capture.
A focused native review found no concrete defects in the fill opt-in; that
review inspected the diff and call paths, while the checks above supplied
execution evidence.

Three instances cannot establish population reliability, task diversity or
model difficulty. No sampled success band, per-group reward variation, E2
opportunity or persistent E3 opportunity has been measured for this family.
The affected readiness checks for the accepted episode-boundary correction
(decision 0014) now pass. Use the accepted four-instance by eight-rollout
screen for the survivor, inspect it, and expand only if warranted. Do not launch
another 300-update pilot from oracle success alone.
