# Deterministic off-target findings from the retained thesis evaluations

Historical 3 November analysis below. The later instruction makes all matched
E1/E2 questions primary and shorter shared successes supplementary. The
[4 November follow-up](offtarget_sample100_20261104_findings.md) completes the
approved 100-pair semantic sample and a full 5,400-trajectory code rerun,
including all four outcome groups and both token-change directions. It reproduces
the full-population measurements below unchanged. Statements below about a
pending semantic review or the user's subset focus describe the earlier stage.

3 November 2026. Retrospective analysis, definition version 2. Semantic LLM and human validation have not been performed. The original five-family findings retain their existing review status; these new outcome-conditioned comparisons have not received independent scientific review. Table 4017 and Inbox 4240 remain provisional.

## Conclusion

Successful compression is compatible with distinct observable outcomes: no additional detected errors in the selected table trajectories, degraded goal completion despite native success in Checkboxes, and more execution errors despite eventual success in Inbox and Tic-tac-toe. There is no single cross-family off-target response. The first Inbox replication repeats the direction of the error increase but reverses the direction of the overall success change. These observations support a thesis discussion of efficiency, outcome quality and execution behavior as separate dimensions. They do not establish a harmful reasoning mechanism or intentional reward exploitation.

The compact decision-facing version is in the [experiment status report](thesis_experiment_status_20261103.md).

## Scope and measurement

- 27 saved final evaluations, 200 episodes each: five original E0/E1/E2/placebo quartets; Table 4017's quartet; Inbox 4240's completed E0/E1/E2 trio. The unfinished Inbox placebo is excluded. Trained endpoints are update 300. No archived attempts, screens or intermediate checkpoints enter these comparisons.
- Seven E2-versus-E1 comparisons are primary for this analysis. Six E2-versus-placebo comparisons and seven E1-versus-E0 comparisons provide separate context. Every contrast has 200 matching seeds and identical initial observations, with no pairing exclusions. No families or seeds are pooled.
- The existing module ran unchanged at **definition v2**. Its outputs are separate from historical v1 results. It measures narrowly defined state/action events; the [definitions](../offtarget/README.md) remain authoritative. Same-turn actions are precommitted and cannot demonstrate a response to same-turn feedback.
- Three populations are retained for each E2 contrast: all paired questions; questions both arms solve under native scoring; and those shared successes where E2 uses strictly fewer saved `n_tokens`. The last is the user's requested focus. It uses model-output tokens, not elapsed time, total trajectory-budget consumption or deployment cost.
- This is a retrospective, outcome-conditioned analysis, not a new preregistration. The cohort rule was saved before this computation, after the overall experiment outcomes were already known. Conditioning on success and compression selects different populations across contrasts; the resulting rates are not whole-test prevalence or causal estimates.
- All 5,400 trajectories have assessable evidence under the implemented definitions: zero unknown episodes and zero unknown observations for every reported metric. This does not imply coverage of every possible harmful behavior. Zero applicable opportunities is distinct from a verified absence of a possible error.

## Primary comparisons and selection

All figures below refer to E1 versus E2. Token change is the median per-question percentage change among shared native successes, before selecting only shorter E2 responses. Lost and gained successes count changes on identical questions.

| Family / seed | Successes E1 / E2 | Lost / gained successes | Shared successes | Shared successes with shorter E2 | Median token change |
|---|---:|---:|---:|---:|---:|
| Table 4016 | 199 / 198 | 2 / 1 | 197 | 195 | -44.9% |
| Inbox 4021 | 183 / 199 | 1 / 17 | 182 | 182 | -92.0% |
| Checkboxes 4232 | 191 / 162 | 36 / 7 | 155 | 152 | -43.5% |
| Tic-tac-toe 4230 | 85 / 66 | 19 / 0 | 66 | 59 | -23.1% |
| Tabs 4231 | 189 / 5 | 184 / 0 | 5 | 3 | -15.7%, descriptive only |
| Table 4017, provisional | 197 / 192 | 6 / 1 | 191 | 191 | -92.0% |
| Inbox 4240, provisional | 199 / 168 | 31 / 0 | 168 | 168 | -74.4% |

The three shorter Tabs pairs cannot establish a stable compression benefit. The new Inbox trio has technical integrity receipts and hash-verified local evidence, but its placebo and independent scientific review are pending.

## What happens within successful compression?

### Checkboxes: native success conceals worse goal completion

On the **152** shared successes with shorter E2 responses, an inexact final selection occurs in **40 E1 episodes (26.3%) versus 91 E2 episodes (59.9%)**. There are 61 newly inexact pairs and 10 corrected pairs. All 152 episodes in each arm have an assessable submission, so this comparison has equal opportunity exposure. E2's 91 mismatches comprise 45 omissions-only cases, 26 extras-only cases and 20 with both.

The pattern does not depend on requiring strictly shorter E2: among all 155 shared successes, the counts are 41 versus 94. Across all 200 questions, inexact submissions rise from 60 to 133; native successes with inexact selections rise from 52 to 100. Thus the finding adds information beyond the already reported success loss. E2 has no recorded tool errors here: syntactically valid execution and native success do not guarantee exact goal completion.

The selected-pair difference is +33.6 percentage points, with the module's approximate conservative 95% difference interval [+15.8, +49.2]. This interval describes instance uncertainty within the selected cohort, not seed uncertainty or simultaneous confirmation across all metrics. Checkbox E1/E2 host differences remain a confound.

### Inbox: error-containing execution can still end correctly

For seed 4021, all **182** shared successes are shorter under E2. Episodes with tool errors increase from **8 to 182**. There are 8 error events in 567 answered E1 calls versus 187 in 747 E2 calls. Adjacent identical dispatched actions occur in 0 versus 54 episodes. Recorded examples include locator timeouts on click/fill attempts. The deterministic module records the errors; it does not decide whether each timeout reflects the policy, interface behavior or another mechanism.

The narrower feedback-repetition detector is **zero in both arms**: no repeated error-producing call from the immediately preceding turn, across 5 E1 and 184 E2 eligible answered calls. Adjacent duplicates therefore must not be relabeled as ignored error feedback. Both policies ultimately execute the requested operation correctly on this selected subset; there are zero commit mismatches. This final-outcome agreement is partly implied by selecting native successes.

For provisional seed 4240, all **168** shared successes are shorter under E2. Error-containing episodes increase from **2 to 33**, with 2/388 versus 38/628 error events per answered calls. Saved errors include attempts to fill non-editable elements. Neither arm repeats a previous-turn error-producing call in this subset (2 versus 38 eligible calls). The error increase therefore recurs at a smaller magnitude. Overall success, however, falls from 199 to 168, rather than improving as in seed 4021. The 31 lost E1 successes must remain part of the interpretation.

### Tables: compression without additional detected errors in the selected subset

For seed 4016's **195** selected pairs, wrong fills are 0/390 versus 0/390 opportunities, and error-containing episodes fall from 3 to 0. For seed 4017's **191** pairs, wrong fills are 0/382 in both arms and neither arm has tool errors. This is a useful negative comparison for the later semantic review, not proof that all compressed trajectories are harmless.

Over the full test sets, mismatched submissions increase from 1 to 2 in seed 4016 and from 2 to 8 in seed 4017. These failures are excluded by the successful-cohort definition. Zero mismatched final submissions within native successes adds little independent evidence about reasoning quality; the absence of wrong intermediate fills is the more relevant additional check.

### Tic-tac-toe: errors within wins, but tactical exposure is not comparable

Among the **59** selected winning pairs, error-containing episodes increase from **0 to 39** (0/186 versus 39/219 answered calls). All 39 E2 errors attempt to click the nonexistent element identifier `ttt-0`. Neither arm clicks a cell known occupied at decision time in this subset.

E2 leaves a unique immediate opponent threat unblocked in 9 of 12 assessable first-action opportunities, spread across 9 winning episodes. E1 has **zero such opportunities**, because its later moves are precommitted in action batches. Consequently, 0 E1 events versus 9 E2 events is not a fair comparison of tactical decision quality. Conversely, E1 has 6 later-batch opponent collisions in 3 selected episodes across 127 opportunities, while E2 has no later-batch opportunities. These execution styles require separate interpretation.

All 66 shared wins, including seven where E2 is not shorter, show 0 versus 42 error-containing episodes. The full-test win loss and accepted native-versus-strict-prefix budget sensitivity remain as documented in the [original findings](tic_tac_toe_s4230_findings.md). This analysis does not redefine the budget or scoring.

### Tabs: repeated errors explain the observed execution pattern, not its cause

The selected successful-compression subset has only **three** pairs: tool errors occur in all three E2 episodes and none of E1's. Two E2 episodes repeat an error-producing call after previous-turn feedback. This is descriptive evidence only.

Across all 200 questions, error-containing episodes rise from 0 to 195, and previous-turn error repetition from 0 to 179. E2 repeats such calls 501 times in 630 eligible opportunities; E1 has no error-following opportunities. E2 reaches a token/turn limit in 176 episodes versus six for E1. These observations characterize the collapse, but do not identify why training produced it. Apparent reductions in other task events can result from failing before reaching the relevant state.

## Controls and interpretation limits

Against placebo, the successful-and-shorter E2 subset also shows inexact Checkboxes selections increasing from **23 to 91 of 151 pairs**, and original Inbox error-containing episodes increasing from **37 to 195 of 195 pairs**. These are separately selected cohorts, not the same questions as the E1 comparisons. The original placebo study was exploratory, declared after original outcomes were known. Reward variance, gradient noise and host/run-date differences prevent a unique mechanism attribution.

E1-versus-E0 outputs are included separately to distinguish specialization from shaping. For example, original Inbox error-containing episodes decrease from 124 to 10 with task-only training before rising to 200 with E2. Such contrasts provide context; they do not turn the E2 contrast into a controlled test of an internal reasoning mechanism.

There is no justified aggregate "off-target score": metrics have different meanings and opportunities, and more events do not always imply worse final outcomes. Shorter answers, one-action completion and action batching are not themselves harmful-behavior labels. Counts, opportunities, directional paired changes and uncertainty are preserved for every metric, including zero and negative results.

## Next semantic review

Proceed with the planned LLM review, followed by human inspection, without launching new training. Prioritize the distinction between Checkboxes' incomplete goals and Inbox's eventual success despite errors. Include table trajectories without detected errors and efficient, correct examples from the other families as negative controls. Treat Tic-tac-toe batch decisions separately from decisions made after new feedback. Tabs provides repeated-feedback-error examples but almost no successful-compression sample.

Follow the [existing review plan](../../docs/plans/off-target-human-llm-review.md): keep a probability sample from the full paired population separate from deliberately selected diagnostic examples, hide arm identity where feasible, retain unknown/not-applicable judgments, and require evidence from the visible trace. Validate semantic labels before making broader claims about under-verification, abandoned valid plans, intent or reward exploitation. This analysis has not started that review.

## Verification and reproducibility

- The focused module tests passed: **76 tests** in `test_offtarget_analysis.py` and `test_offtarget_metrics.py`.
- The analysis driver's synthetic cohort-selection check, success partitions and all-cohort equality checks passed. Each module contrast paired exactly 200 identical initial observations.
- All seven new-seed evaluations' episode/config/environment/report/technical-review hashes match their harvest manifests; their technical integrity receipts pass. These checks do not complete their scientific review.
- A separate check verified all source/output hashes, native accuracy, saved invalid-action counts and wrong-termination indicators. Repeated analyses of the same run produce identical event records. Coverage was checked across all 27 evaluations, with no unknown measurements.
- No measurement, training or evaluation code, frozen input, original finding, presentation or experiment queue was changed. No semantic LLM judge, human validation or new GPU experiment was run.

Artifacts: [scope](offtarget-v2-thesis-20261103/measurements/scope.json), [cohort membership and complete metrics](offtarget-v2-thesis-20261103/measurements/summary.json), [hash receipt](offtarget-v2-thesis-20261103/measurements/receipt.json), and [reproduction driver](offtarget-v2-thesis-20261103/analyze_saved.py). Each `<family-seed>/<contrast>/` directory under `measurements/` contains the module's full-population Markdown/JSON report, located episode events and its own receipt. The summary retains all shared-success and strictly-shorter cohort counts, opportunities, unknowns and paired statistics. Historical v1 artifacts remain unchanged.

From `pipeline/`, reproduce into a fresh directory:

```bash
PYTHONPATH=. ../.venv-test/bin/python runs/offtarget-v2-thesis-20261103/analyze_saved.py --output runs/<new-directory>
```
