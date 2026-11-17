# Semantic sample review and full-corpus deterministic follow-up

4 November 2026. Machine review and consolidation are complete for the revised,
bounded scope. Human validation remains open. The original experiment review
status is unchanged; Table 4017 and Inbox 4240 remain scientifically provisional.

## Conclusions

The results support a task-specific account of length shaping, not a general
claim that E2 reasons less carefully. Checkboxes shows worse exact completion
even among native successes. Tabs shows repeated invalid actions after explicit
feedback, frequently with longer responses. Inbox distinguishes recoverable
execution errors from final outcome quality and differs substantially by seed.
Tic-tac-toe changes how actions are batched, which changes both verification
opportunities and the interpretation of apparent feedback failures. Tables
provide no consistent semantic process explanation for their small success loss.

The LLM review helps characterize these patterns and identify ambiguous cases.
It does not establish a causal reasoning mechanism, intent, or a validated
cross-family off-target prevalence. The compact conclusions and next steps are
in the [status report](thesis_experiment_status_20261103.md).

## Scope, coverage and provenance

The user replaced the full two-model census with approximately 100 paired tasks
to limit time and usage. The frozen corpus remains 27 final evaluations and
5,400 trajectories. The deterministic analysis still covers every trajectory,
including E0 and available placebos. No newly harvested evaluation was added.

The semantic sample contains 100 E1/E2 pairs, or 200 trajectories: 46 shared
successes, 28 E1-only successes, 13 E2-only successes and 13 shared failures.
Fourteen or fifteen pairs were selected per family/seed stratum, with round-robin
allocation across available outcome groups and random selection within groups
(seed 20261104). Selection did not use reviewer labels, review availability,
token differences or detector events. Rare outcome groups are oversampled.
Unweighted semantic rates describe this sample, not the full test population.

| Stratum | Pairs | Claude evaluated E1 / E2 | Codex evaluated E1 / E2 |
|---|---:|---:|---:|
| Table 4016 | 14 | 7 / 8 | 14 / 14 |
| Table 4017 | 14 | 7 / 10 | 14 / 14 |
| Inbox 4021 | 15 | 11 / 10 | 15 / 15 |
| Inbox 4240 | 14 | 10 / 11 | 14 / 14 |
| Checkboxes 4232 | 15 | 8 / 9 | 15 / 15 |
| Tic-tac-toe 4230 | 14 | 7 / 6 | 14 / 14 |
| Tabs 4231 | 14 | 7 / 7 | 14 / 14 |

All 400 selected reviewer/trajectory requests have a result or an explicit
unavailable record. Codex supplied 200 evaluations and Claude 118. Claude has
82 unavailable annotations: 66 from refused requests and 16 from one invalid
empty package response. The empty response is not classified as a confirmed
safeguard refusal. There was no retry or rephrasing of these requests. Only
50 pairs have evaluations from both models for both arms. Missingness affects
entire packages as well as individual trajectories and is not assumed random.

The original queues retain 1,135 evaluated Claude reviews, 186 Claude unavailable
records and 1,774 evaluated Codex reviews. These include E0/placebo and other
out-of-sample trajectories; they remain preserved, but are not treated as a
completed semantic census or pooled into the selected sample. The selected
sample reused 119 request records and added 281 requested episode annotations
in 19 packages. The separate two-package pilot also remains preserved.

Claude Opus 5.5 and Codex GPT 6.1 Sol ran independently at xhigh. Arm names,
native outcomes, token totals and detector flags were hidden; task feedback
inherently reveals some outcome information. Earlier reviews used a fresh
context per trajectory. New reviews used packages of at most 16 trajectories,
with lossless repeated-text references and one fresh context per package.
Positive/unknown labels retain evidence, harm and alternatives; negative
rationales were not elicited in the compact package format. This is a declared
annotation-protocol change. Of the evaluated selected annotations, Claude has
47 singleton and 71 package reviews; Codex has 69 singleton and 131 package
reviews. Models never saw one another's judgments.

The bounded model window, including the pilot and preparation, ran from
05:24:13 to 05:53:27 UTC, approximately 29.2 minutes. Consolidation and reporting
followed. This is additional work after the earlier census, not the total review
campaign duration. Usage is saved per request; no actual billing or remaining
subscription allowance is inferred from the CLI's list-price estimates.

## Qualitative findings and disagreements

Counts below are evidence-checked reviewer judgments. An arrow is E1 to E2 on
the same pairs with known labels, with the number of such pairs stated. Missing
annotations are never negative labels. A supported label and observed harm are
separate: a redundant action can waste effort without changing final success.

- **Tabs:** feedback-use failures increase from 3 to 13 on 14 Codex pairs and
  from 3 to 5 on six Claude pairs. The examples show retries with rejected HTML
  identifiers and repeated calls to an unavailable tool after explicit errors.
  Some eventually succeed: Codex marks 0 to 4 among all five shared successes;
  Claude marks 0 to 2 among three jointly available pairs. The full deterministic
  count of previous-turn error repetition is 0 to 179 of 200. The semantic
  evidence supports a description of failed adaptation to visible feedback;
  it does not explain why training produced it. Additional verification and
  abandoned-plan labels overlap this pattern and are less consistently assigned.
- **Checkboxes:** Codex independently rates exact goal satisfaction at 8 to 1
  among 15 pairs, although each arm has nine native successes in this selected
  sample. Seven pairs change from satisfied to partial. All three corresponding
  losses with both Claude annotations are also non-satisfied under Claude.
  Full-population exact completion is 139 to 62 of 200. Traces contain omitted
  synonyms, extra selections and input/label double toggles. These are observable
  quality failures. Whether they reflect skipped checking, mapping mistakes or
  vocabulary limitations remains disputed: verification labels move in opposite
  directions for the two reviewers on their respective known pairs.
- **Inbox 4021:** feedback-use failures decrease from 3 to 0 on 15 Codex pairs
  and 1 to 0 on nine Claude pairs, despite tool errors in all 200 full-corpus E2
  episodes. Several errors occur in precommitted action batches, followed by
  successful correction. They cannot be described as ignoring feedback received
  only after those actions were chosen. Verification is contested: Codex labels
  precommitting a fill before opening the reply form; Claude emphasizes the
  later successful check before sending. On the seven E2-only sampled successes,
  Codex feedback failures fall from 3 to 0. This contrasts with worsening
  execution-error counts and illustrates why outcome groups must stay separate.
- **Inbox 4240:** both reviewers identify repeated clicks on an already open
  email instead of the visible Reply control. Among the six sampled E1-only
  successes, feedback failures rise from 0 to 3 for Codex and from 0 to 3 on the
  five pairs with Claude coverage. Across all known pairs, both show 1 to 3,
  but this increase does not persist in the smaller same-protocol subsets.
  The full deterministic no-tool-call ending count rises from 0 to 21. Several
  sampled endings contain an unclosed `<think>` prefix and an unexecuted recovery
  plan. Even where both reviewers label premature disengagement, formatting or
  output failure remains plausible. Voluntary abandonment is not established.
- **Tic-tac-toe:** Codex labels insufficient verification in 14 E1 versus six
  E2 trajectories; Claude in six versus one on six known pairs. E1 frequently
  precommits a move sequence without observing intervening opponent moves.
  These positive labels do not imply losses: on the five sampled shared wins,
  Codex labels verification gaps in all five E1 traces but observed harm in none.
  Across all 14 Codex pairs, observed verification-related harm is 7 to 6,
  a much smaller contrast than the label count. E2 instead offers later board
  decisions, including missed wins and unblocked threats. A simple comparison
  of feedback-failure rates would confound behavior with opportunity.
- **Tables:** both seeds have no consistent increase in the six semantic
  process labels. Wrong submissions remain measurable, including 2 to 8 in
  seed 4017. A wrong value with little recorded reasoning cannot establish that
  the model deliberately skipped a check. One E1 table error explicitly confuses
  a label identifier with its input and submits within the same batch; both
  reviewers recognize that localized verification problem.

There is no consistent cross-family increase in premature disengagement,
abandoned promising reasoning or unsupported completion claims. The sole Claude
positive completion-claim example is a forecast inside pre-action reasoning,
not an unequivocal post-action success assertion. It stays disputed rather than
supporting a claim of deception or systematic false reporting.

The coordinator inspected all 103 positive reviewer records, substantive label
disagreements, unknown explanations, the evidence-invalid case and selected
original traces. This inspection adds interpretation; it does not replace human
adjudication or overwrite either reviewer's labels.

## Agreement, evidence and protocol sensitivity

For known judgments from both reviewers, feedback-label agreement is 116/118
and verification-label agreement is 103/115. Exact four-way goal-label agreement
is only 82/118, largely because one reviewer calls an incomplete result failed
while the other calls it partial. The distinction between satisfied and
non-satisfied agrees in all 118 available cases. Agreement is not ground truth;
many negative labels and shared blind spots can produce high agreement.

One Codex verification quote in Tabs is not verbatim at its cited location.
The raw label remains saved; the evidence-checked result is unknown. Earlier
compact-JSON formatting alarms were corrected by a separate validator that
accepts the exact serialization shown in package prompts, without changing raw
responses or accepting changed arguments. All remaining evidence checks pass
locational matching, which does not prove the interpretation of a quote.

Sensitivity groups retain only pairs reviewed with the same annotation protocol
for a given model. Tabs' feedback increase remains: 7 new / 0 removed labels
on ten Codex pairs and 1 / 0 on four Claude pairs. Tic-tac-toe's verification
decrease remains for Codex (0 new / 3 removed on six pairs), but Claude has only
one known pair in this sensitivity. Inbox 4240's aggregate feedback increase
does not persist: Codex has 0 new / 1 removed on six pairs; Claude 1 / 1 on six.
Small, selectively available subsets cannot isolate protocol effects. They
limit, rather than validate, broader prevalence claims. Automated intervals and
unadjusted tests in the raw summaries are not used as confirmatory population
inference for this stratified, incomplete semantic sample.

## Full-population deterministic follow-up

The code module was rerun after semantic consolidation, using unchanged
definition v2 and the same frozen inputs. All 20 contrasts reproduce the prior
full-population arm summaries and paired statistics exactly; all 40 event files
are byte-identical. Each contrast includes 200 matched questions with identical
initial observations. All 27 evaluations and 5,400 trajectories remain included,
with zero unknown measurements under the implemented definitions. This does not
cover every possible behavior. The [earlier detailed event findings](offtarget_v2_thesis_20261103_findings.md)
remain the record for those measurements; its old subset emphasis is superseded.

| Stratum | Both success | E1 only | E2 only | Both failure | E2 shorter / longer, all 200 |
|---|---:|---:|---:|---:|---:|
| Table 4016 | 197 | 2 | 1 | 0 | 198 / 2 |
| Table 4017 | 191 | 6 | 1 | 2 | 200 / 0 |
| Inbox 4021 | 182 | 1 | 17 | 0 | 200 / 0 |
| Inbox 4240 | 168 | 31 | 0 | 1 | 200 / 0 |
| Checkboxes 4232 | 155 | 36 | 7 | 2 | 193 / 7 |
| Tic-tac-toe 4230 | 66 | 19 | 0 | 115 | 153 / 47 |
| Tabs 4231 | 5 | 184 | 0 | 11 | 30 / 170 |

There are no equal-token pairs. All groups, including empty groups, are retained.
Of the 184 lost Tabs successes, 163 use more E2 tokens; of the 19 lost
Tic-tac-toe wins, 13 use more. Failure and repeated action can lengthen responses
despite a training reward that favors shorter successes. Conversely, both Inbox
seeds are shorter on all questions even though success changes in opposite
directions. These all-question token changes are not pure efficiency estimates.

Opportunity denominators remain essential. For example, Tic-tac-toe E1 has zero
assessable immediate-win decisions and E2 has 33 missed wins in 99 opportunities;
E1 instead has 170 later-batch opponent collisions in 644 opportunities, compared
with zero E2 later-batch opportunities. Neither zero establishes better decisions.
Tabs E2 repeats errors 501 times in 630 eligible calls; E1 has no error-following
opportunities. The episode count and conditional opportunity rate answer
different questions and remain separate.

E0 and six available placebo comparisons are preserved as full-population
controls, not mixed with E1/E2. For example, original Inbox error-containing
episodes are 124 in E0, 10 in E1 and 200 in E2. The incomplete out-of-sample LLM
reviews do not supply a new balanced semantic placebo comparison. Existing
post-hoc placebo, host/software, seed/question-set and native-budget limitations
remain. No causal-mechanism or intent claim follows from the observed patterns.

## Human review and next steps

Proceed with writing the results as three separate dimensions: success, tokens
and observable behavior. No additional LLM or training run is needed to inspect
the present evidence. Validate the disputed constructs before including them as
more than exploratory qualitative observations. The existing replication and
its independent scientific reviews remain separate work.

Start the human review with these auditable examples. IDs refer to the frozen
sample; the full paired seed/arm mapping is in `coordinator_key.json`, while each
review directory preserves both located evidence and original source receipts.

| Case | What to decide |
|---|---|
| [Tabs ad70576cad0233a45a8d](trajectory-review-20261104-sample100/packets/ad70576cad0233a45a8d.json) | Repeated rejected identifier and unavailable tool: feedback failure is visible; does the separate verification label add anything? |
| [Checkboxes 6490c4fb10e21c09b5ec](trajectory-review-20261104-sample100/packets/6490c4fb10e21c09b5ec.json) | Omitted rabbits and a double toggle of camouflage: distinguish wrong call assembly from an identifiable skipped check. |
| [Inbox c3a809005644e3932199](trajectory-review-20261104-sample100/packets/c3a809005644e3932199.json) | Initial invalid fill, then correctly checked reply and send: classify local wasted action separately from final goal harm. |
| [Inbox eab5a502290b998a7a79](trajectory-review-20261104-sample100/packets/eab5a502290b998a7a79.json) | Both judges label disengagement, but the final unclosed think output leaves voluntary stopping unresolved. |
| [Tic-tac-toe 658ecc7f3cd9a2c75859](trajectory-review-20261104-sample100/packets/658ecc7f3cd9a2c75859.json) | Precommitted successful play: separate missing intermediate checks, no-op clicks and demonstrated outcome harm. |
| [Checkboxes bd7334482bff86198006](trajectory-review-20261104-sample100/packets/bd7334482bff86198006.json) | A mistaken pre-action success forecast: should it count as an unsupported completion claim? |
| [Tabs 65ea613c6c7febfc8102](trajectory-review-20261104-sample100/packets/65ea613c6c7febfc8102.json) | Incorrect target is visible, but one quote is paraphrased; retain the unknown evidence-checked verification label until adjudication. |

These are diagnostic examples, not a prevalence sample. For human label
validation, also inspect randomly selected agreements and negative cases;
reviewing only disagreements would miss shared reviewer errors.

## Reproduction and checks

The [workflow](../trajectory_review/README.md) documents selection, batching,
import, consolidation and deterministic follow-up. Frozen selection, prompts,
model settings, packets, raw responses, unavailable receipts and usage remain
in the run directories. New runs must use new output directories. Existing
responses must not be rerated to obtain agreement.

- [Selection](trajectory-review-20261104-sample100/selection.json), [semantic summary](trajectory-review-20261104-sample100/consolidated/summary.json), [disagreements](trajectory-review-20261104-sample100/consolidated/disagreements.jsonl), [positive rationales](trajectory-review-20261104-sample100/consolidated/positive_rationales.jsonl).
- [Full deterministic outputs](trajectory-review-20261104-sample100/deterministic-full/summary.json) and their [receipt](trajectory-review-20261104-sample100/deterministic-full/receipt.json).
- [Completion audit](trajectory-review-20261104-sample100/completion_audit.json): 524 source, 103 output and 3,702 review-artifact hash checks; unchanged measurements; coverage and timing. Recheck with `python3 pipeline/runs/trajectory-review-20261104-sample100/verify_completion.py` from the repository root. The added extraction files have separate hashes here; the original consolidation receipt is unchanged.
- The project gate passed: 613 tests, seven skips, all 17 setup checks, formatting, lint and types. Tests verify execution/data handling, not semantic validity. No training/evaluation definitions, frozen experimental inputs, presentation or GPU queue were changed.

## Evidence-validation addendum, 14 November 2026

The commit review identified an empty-quote bypass in the alternate compact
tool-call JSON match. The corrected validator retains empty and whitespace-only
quotes as evidence issues. Its regression test fails before the fix and passes
afterward.

A read-only audit checked all 21 saved batch `normalized.json` files, containing
233 reviews and 295 tool-call evidence quotes across the pilot and sample batch
roots. None of those tool-call quotes is empty or whitespace-only. Replaying the
original validator reproduces the historical checked issues; the corrected
validator changes no issues. Therefore this fix changes no evidence-filtered
labels or reported numbers. It does not add semantic validation.

The [audit receipt](trajectory-review-empty-quote-audit-20261114/audit.json)
binds the original raw/normalized/checked files, packets, source manifests and
both validator versions. Corrected derived checks are separate files under its
`corrected-checks/` directory; every checked historical input retained its hash.
No original `checked.json`, imported review, consolidation or frozen response
was overwritten, and no model request was made. The
[audit script](trajectory-review-empty-quote-audit-20261114/audit.py) retains the
comparison and immutable-output checks. The technical trap is in `LAB_NOTES.md`.

## Package-composition addendum, 17 November 2026

A code review found that the package reviews were not blind to the pairing.
The sample key lists the E1 and E2 trajectory of each question next to each
other, in random arm order, and packages were cut from that list in order.
Claude received both arms of 53 of the 100 pairs in one package. It evaluated
both arms in 23 of them and returned no review for either arm in the other 30.
Codex received and evaluated both arms of 42 pairs in one package. Every other
pair had its two arms in different requests.

Inside a shared package the reviewer sees both trajectories of the task. They
have the same goal and start page, which the repeated-text reference makes
explicit, and the arms often differ visibly in length. The rubric asks for
separate judgments and arm labels stayed in the coordinator key, but contrast
between the two arms cannot be ruled out for these annotations.

No label, count or conclusion in this report changes. Read paired E1/E2 label
differences from package reviews as not blind to the pairing, in addition to
the declared shared-context limit of the package mode. The packager now keeps
the trajectories of one question in separate packages (see the
[review README](../trajectory_review/README.md)). The existing batch roots,
responses and imports are untouched, and no model request was made.
