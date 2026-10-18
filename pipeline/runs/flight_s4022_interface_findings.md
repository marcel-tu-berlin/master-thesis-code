# Flight qualification: initial interface check

Follow-up: opt-in scrolling and complete-path controls are now verified in
[flight_s4022_scroll_findings.md](flight_s4022_scroll_findings.md). The original
pre-scroll measurements below remain unchanged.

Checked: 2026-10-06. Status: **qualification incomplete; interface constraint
identified before policy sampling**. Decision 0026 governs the selected family,
hardware evidence and prohibition on autonomous fallback.

SSH access was restored. Both L4 GPUs were idle at the initial and final checks,
with 23,034 MiB reported per device. This phase ran on CPU with CUDA devices
masked out; no policy weights, training or model evaluation were launched.

## Completed checks

The deployed pipeline matched the local source by checksum before whole-tree
sync. OpenEnv and MiniWoB were clean at their declared pins, and the MiniWoB
static server returned the selected task. The production visible-HTML adapter
was used with click, fill and noop. No task, scorer, observation formatter or
agent budget was changed.

Three fixed development seeds, 4022700000-4022700002, produced byte-identical
initial observations after an intervening reset. Noop preserved each visible
page. The initial prompts, including the native tool template, contained 843,
841 and 843 tokens, respectively. All fit the 4,096-token prompt allowance.
This verifies these initial states, not complete task feasibility.

## Scripted-path stop

The declared next check comprised correct, wrong-booking and invalid-origin
recovery paths for those three seeds. It stopped during the first correct path;
none of the nine complete-path controls passed or supplied a policy-success
estimate.

Attempt 1 failed in the diagnostic's city matching: a rendered list marker was
included in the city label. Attempt 2 selected the visible autocomplete label
element directly and passed the previously failing step. Both attempts and their
source hashes are retained. This was a diagnostic error, not an environment or
model failure.

Attempt 2 entered the requested route, selected 11/07/2016 through the calendar
and opened the results for seed 4022700000. The observation showed only part of
the first flight, ending at the Duration label. Its duration value, price and
booking control, as well as the later flights, were outside the visible region.
The saved BrowserGym visibility metadata agrees: the first booking control and
later flight elements have visibility zero. The agent-facing adapter exposes
no scroll action, so the script could not inspect the alternatives through that
tool set. It stopped rather than selecting a hidden element ID.

The path also exposes a feedback-cost issue:

| Action | Feedback tokens | Cumulative feedback tokens |
| --- | ---: | ---: |
| Fill origin query | 594 | 594 |
| Select origin suggestion | 326 | 920 |
| Fill destination query | 465 | 1,385 |
| Select destination suggestion | 335 | 1,720 |
| Open calendar | 3,190 | 4,910 |
| Navigate to November | 3,209 | 8,119 |
| Select date | 346 | 8,465 |
| Search | 529 | 8,994 |

These are tokenizer counts of the exact feedback strings, using the pinned
Qwen3-1.7B tokenizer. They exclude assistant output and message framing. The
model-free diagnostic measures an unrestricted scripted path against the existing
budget; it is not an agent rollout executed with a larger allowance. Its feedback
alone exceeds the agent's 4,096-token allowance when the calendar opens.

The 8,994-token prefix is not a proven minimum over all valid strategies. It
does not establish how many assistant turns an efficiently batched policy needs,
how much a complete workflow requires, or that every flight instance fails at
the current budget. It does establish that the ordinary inspected route is not
an adequate feasibility witness for the current agent protocol. Calendar HTML
accounts for most of its observed feedback cost.

## Hardware interpretation and remaining work

There is no flight-specific GPU-memory or training-throughput measurement yet.
More VRAM would not supply the missing scroll action or alter the configured
trajectory allowance. These results therefore do not yet justify a particular
GPU capacity or allocation request.

Next qualify scrolling within the visible interface, preserving hidden-state
exclusion, then complete correct/wrong/recovery controls and measure full paths.
Any tool or observation change must be explicit and isolated from the retained
campaigns. Budget changes still require the user's explicit approval under
decision 0025. Once the useful protocol is concrete, measure or clearly label
estimates of its resource requirements before preparing the supervisor request.
Do not switch families.

## Evidence

`flight-qualification-s4022-ops/` contains the snapshot script, plan, three raw
snapshots and completion receipt; both oracle scripts, plans, logs and partial
event journals; and `remote-manifest.json`. All 13 files listed in the remote
manifest matched their harvested SHA-256 values. Snapshot source hashes also
match the local runtime files. Both failed oracle workers exited with code 1
and shut down their environment servers. No qualification worker remained and
both GPUs remained idle at the final process/device check.
