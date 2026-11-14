# Five-family training diagnostics, 12 November 2026

This is a descriptive overview of the current five-family campaigns. It reuses
the existing plotting functions and saved reports. It does not replace the
completed per-seed scientific reviews, select another checkpoint, change a
measurement, or authorize further experiments.

## Deliverables and coverage

- [Consolidated diagnostic PDF](../../output/pdf/training-diagnostics-20261112.pdf).
- [Reproduction script](training-diagnostics-20261112/build.py),
  [machine-readable summary](training-diagnostics-20261112/summary.json) and
  [source hashes/page index](training-diagnostics-20261112/manifest.json).
- 24 complete training runs: E1, E2 and matched placebo for all five original
  families, plus Table 4017, Inbox 4240 and Checkbox 4242.
- One additional complete training log, Tic-tac-toe E1 seed 4244, captured
  read-only from L4 at 2026-11-11T23:28:00 UTC. Its evaluation/admission is
  pending in this snapshot, so it is explicitly provisional.
- The campaign snapshot is 2026-11-11T23:27:20 UTC, or 12 November just after
  midnight in Berlin. E0 has no training curve. Queued cells, the aborted L4
  checkbox E2, archived development experiments and void results are excluded.

PDF pages 1-2 explain coverage and summarize the logs; pages 3-4 show all saved
checkpoint success and successful-only length curves; pages 5-13 compare arms
within family/seed; pages 14-38 show each individual training log.

## Assessment

A blanket "no anomalies" conclusion is not supported. Technical completion
and an improving training reward do not establish a good fitted policy.

The table below uses means over updates 1-50 and 251-300. It is descriptive;
the windows are not independent samples or significance tests.

| Training signal | First 50 | Last 50 | Reading |
| --- | ---: | ---: | --- |
| Tab E2 seed 4231, raw task reward | 0.2950 | 0.0731 | Clear training deterioration |
| Tab E2 seed 4231, non-EOS completion fraction | 0.1831 | 0.3275 | More non-EOS endings; not an exact budget-cap count |
| Tab E2 seed 4231, mean failed-call fraction | 0.0000 | 0.1704 | A higher per-update failed-call fraction accompanies the decline |
| Inbox E2 seed 4240, raw task reward | 0.5750 | 0.9781 | Training performance improves despite the documented late held-out decline |
| Checkbox E2 seed 4242, raw task reward | 0.7863 | 0.9531 | Native training reward does not measure exact checkbox completion |
| Tic-tac-toe E1 seed 4244, raw task reward | 0.1519 | 0.2456 | Some training improvement; final evaluation is still pending |
| Tic-tac-toe E1 seed 4244, non-EOS completion fraction | 0.3956 | 0.1506 | Reduced but still common non-EOS endings |

Interpret these curves with the existing task-grounded findings:

- [Tabs](click_tab_2_s4231_findings.md): the completed E2 collapse is already a
  reviewed negative result. The new overview makes its reward deterioration,
  non-EOS endings and tool failures visible together.
- [Inbox seed 4240](seed_replication_email_inbox_noscroll_s4240_findings.md):
  late successful-response compression coexists with worse held-out completion.
  Its high late training reward would conceal this if considered alone.
- [Checkbox seed 4242](seed_replication_click_checkboxes_soft_s4242_findings.md)
  and [original seed](click_checkboxes_soft_s4232_findings.md): native majority
  success and exact completion remain separate endpoints. A smooth reward curve
  cannot resolve that measurement limitation.
- [Table seed 4017](seed_replication_read_table_2_s4017_findings.md): task
  learning and compression coexist with unresolved preservation/behavior bounds.
  A stable-looking curve is insufficient for a clean scientific endpoint verdict.

Low entropy during successful compression is not, by itself, collapse. Likewise,
high zero-variance fractions in saturated E1 groups are a gradient-availability
limitation, not proof of a runtime fault. The matched placebo remains necessary
when interpreting different gradient availability under shaping.

## Checks and limits

The builder checks all 7500 reward records for finite numeric values, exactly
one record per update 1-300, and agreement of logged total reward with the
configured weighted raw components to absolute tolerance 1e-6. Placebo uses
the recorded `WithinGroupShuffle` component name. Corrupt-record self-checks
exercise missing updates, non-finite rewards and incorrect reward arithmetic.
All 72 completed checkpoint reports have 200 samples and the correct run/update
identity. Source hashes are checked again after plotting.

All 25 frozen configs use `kl_beta: 0`; KL is absent from these logs. This is
not a measured zero-KL result. The pinned TRL source counts model-generated
tokens through the tool mask, excluding tool-feedback tokens, for its completion
length metrics. Its `completions/clipped_ratio` checks whether the final token
is neither EOS nor padding; it does not directly count budget-cap events.
Likewise its "terminated" length means EOS-ended, not environment success.
Tool failure frequency divides failed calls by all tool calls within each
update, logging zero when no calls occur. The summary averages those update
fractions; it is not a pooled fraction of all calls or failures per completion.
The installed TRL 1.6.0 source reconstructs weighted total rewards and logs
`nanstd(rewards)` across the update's completions. Thus `reward_std` measures
total composed reward, not task reward alone in E2/placebo, and is distinct
from mean within-prompt-group SD.
The source excerpts and full-file hash are retained in
[metric-source.json](training-diagnostics-20261112/metric-source.json).
Held-out length charts use the recorded assistant-token counts. These rulers
need not match exactly. Successful-only held-out means use each policy's own
successful subset and do not replace the existing paired token comparisons.
The length page labels each point with its successful-episode count, making
small successful subsets such as the final Tab E2 policy's five successes visible.

Overlay trends are non-overlapping ten-update means over faint individual
updates. No seed or family is pooled. Checkpoints reuse the held-out questions
and are not independent replications. The original home-host and kernel/driver
limitations remain in the linked findings. Log-level arithmetic does not
independently replay all rollouts, re-tokenize trajectories, recheck weights or
validate semantic behavior; existing hash-bound reviews retain that scope.

Regenerate this dated snapshot from the repository root:

```bash
MPLCONFIGDIR=/private/tmp/thesis-training-mpl .venv-test/bin/python \
  pipeline/runs/training-diagnostics-20261112/build.py
```

This builder intentionally binds the current run counts and frozen provisional
capture. A later refresh needs a new dated snapshot and explicit coverage
updates, rather than silently labeling this PDF current forever.

## Verification record

The plotting tests passed (39), and the full CPU suite passed (613, with seven
skips). The PDF was reopened, rendered and visually checked; its text bounds,
page/run identities, all 72 successful-count labels and required metric-label
replacements passed. Numerical summary bytes remained unchanged after the
reporting corrections. See [validation.json](training-diagnostics-20261112/validation.json).

Independent counterpart review passed after four accepted fixes and one
source-verified rejection; the requested total-reward clarification was retained
with that rejection. The [disposition](training-diagnostics-20261112/counterpart-disposition.json)
and [correction verdict](training-diagnostics-20261112/counterpart-corrections-output.txt)
preserve the evidence. The first launch returned no diagnostic; a retry reached
its tool-turn limit. Neither produced a verdict. A final layout-only adjustment separates the
success-count labels into condition rows; it changes no recorded measurements.
