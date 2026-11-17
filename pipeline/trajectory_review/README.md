# Repeatable semantic trajectory review

## Current authorized scope: 100 paired tasks

Completed on 4 November: 200 Codex evaluations, 118 Claude evaluations and
82 explicit unavailable Claude records. The full deterministic rerun covers
5,400 trajectories and reproduces the previous measurements. See the
[consolidated findings](../runs/offtarget_sample100_20261104_findings.md).
Human validation remains open. All review queues are stopped; commands below
document reproduction and do not authorize another model run.

The 4 November pilot showed that completing all remaining model reviews would
exceed the requested time and usage. The user then authorized 100 paired E1/E2
tasks across all families and native-outcome groups, with reuse of prior reviews.
The code panel remains a full-corpus analysis. Sampling is independent of LLM
judgments, availability, token changes and detector labels; seed 20261104 and
all allocation/selection details are frozen in `selection.json`.

```bash
../.venv-test/bin/python -m trajectory_review.sample prepare \
  --source runs/trajectory-review-20261103-v1 \
  --output runs/trajectory-review-20261104-sample100
../.venv-test/bin/python -m trajectory_review.sample import \
  --root runs/trajectory-review-20261104-sample100 \
  --singleton runs/trajectory-review-20261103-v1 \
  --batches runs/trajectory-review-20261104-batch-v2
../.venv-test/bin/python -m trajectory_review.batches prepare \
  --source runs/trajectory-review-20261104-sample100 \
  --output runs/trajectory-review-20261104-sample100-batches --size 16
```

For this continuation, `window.json` was copied unchanged from the pilot root
before either new queue started. The original shared 60-minute deadline remains
in force. Run `trajectory_review.batches run` on the sample batch root with
`--reviewer claude` and separately `--reviewer codex`, both `--workers 4`.
No additional singleton requests or automatic continuation are authorized.

After requests finish, import their preserved results and consolidate:

```bash
../.venv-test/bin/python -m trajectory_review.sample import \
  --root runs/trajectory-review-20261104-sample100 \
  --batches runs/trajectory-review-20261104-sample100-batches
../.venv-test/bin/python -m trajectory_review.consolidate \
  --root runs/trajectory-review-20261104-sample100 \
  --output runs/trajectory-review-20261104-sample100/consolidated
../.venv-test/bin/python -m trajectory_review.deterministic_followup \
  --root runs/trajectory-review-20261103-v1 \
  --semantic runs/trajectory-review-20261104-sample100/consolidated \
  --output runs/trajectory-review-20261104-sample100/deterministic-full
```

The last command deliberately uses the original full-corpus root. The semantic
sample has its parent manifest hash; its 200 selected trajectories must not
shrink the deterministic analysis. Report unweighted sample findings separately
from full-population events, preserve unavailable annotations, annotation modes
and paired mode-sensitivity results, and retain earlier reviews outside the
sample without pretending they form a completed two-model census.

`batch_evidence` creates a separate `checked.json` accepting compact tool-call
JSON as shown in batch prompts. It preserves raw and initially normalized
responses. Import uses those checked issues and pins source receipts/raw files;
no model is queried again to fix a technical quote-format mismatch.

The 14 November empty-quote guard changes the validator hash. Existing
`checked.json` files and imported results remain immutable: rerunning the
corrected validator in place will refuse the changed derived check. Preserve
those files and use a new output path for a retrospective check. The versioned
recheck and its impact are recorded in the
[validation addendum](../runs/offtarget_sample100_20261104_findings.md#evidence-validation-addendum-14-november-2026).

The 17 November packager change keeps the trajectories of one question (same
stratum and seed) out of a shared package. Earlier packages were cut from the
key in order, and the sample key lists both arms of a pair next to each other,
so many pairs shared a package. The counts are in the
[package-composition addendum](../runs/offtarget_sample100_20261104_findings.md#package-composition-addendum-17-november-2026).
The change alters the `batches.py` hash. `verify_batch`, and with it batch
import and `batch_evidence`, now refuses both existing batch roots, which pin
the packager of commit `20393b1`. Their packages, responses and imported
results are unchanged; reproduce their checks with that file version. A new
review needs a new batch root.

## Superseded full-corpus batching pilot

The user changed the execution protocol on 4 November to small packages, lower
usage and approximately one hour. The singleton queues below are stopped and
their heartbeat is deleted. The new, bounded workflow is:

```bash
../.venv-test/bin/python -m trajectory_review.batches prepare \
  --source runs/trajectory-review-20261103-v1 \
  --output runs/trajectory-review-20261104-batch-v2 --size 16
../.venv-test/bin/python -m trajectory_review.batches run \
  --root runs/trajectory-review-20261104-batch-v2 \
  --reviewer claude --workers 1 --limit 1
# Run the same pilot for --reviewer codex. Inspect time/usage before continuing.
```

See the new run's `OPERATIONS.md`. Existing completed reviews are retained,
including refusals; only unfinished episodes enter packages. The common 60-minute
window begins with the first pilot, and cannot be reset without authorization.
The six labels and positive/unknown evidence requirements remain; repetitive
negative rationales are omitted. Exact repeated input strings are represented
losslessly. Shared package context is a declared methodological change. The old
singleton commands below document the original protocol, not an active queue.

This CPU-side workflow reads saved evaluations and submits blinded, individual
trajectories to the existing Claude Code and Codex logins. It does not train or
evaluate the experimental policy. The user authorized both reviewers on the full
retained corpus on 3 November 2026. The [research plan](../../docs/plans/off-target-human-llm-review.md)
supplies the six constructs; the [frozen rubric](rubric.md) implements them.

The primary population is **all matched E1/E2 test questions**, including both
successes, lost successes, gained successes and both failures. Compression is
examined within all four outcome groups. Shared successes with shorter E2 are a
supplement, never the primary filter. E0 and available placebos are reviewed too.
No checkpoints, families or seeds are pooled as independent replications.

## Execution

From `pipeline/`:

```bash
# Create a new immutable corpus from a completed deterministic-analysis receipt.
../.venv-test/bin/python -m trajectory_review prepare \
  --source runs/offtarget-v2-thesis-20261103/measurements \
  --output runs/trajectory-review-20261103-v1

# Each reviewer uses independent fresh contexts, one trajectory per request.
../.venv-test/bin/python -m trajectory_review run \
  --root runs/trajectory-review-20261103-v1 --reviewer claude --workers 4
../.venv-test/bin/python -m trajectory_review run \
  --root runs/trajectory-review-20261103-v1 --reviewer codex --workers 4
```

The two reviewer commands may run concurrently. `--limit N` processes only the
next N unreviewed packets for a technical pilot; it does not change the declared
full-corpus population. Continue with the same command without `--limit`.
Reviewers are fixed to Claude Opus 5.5 and GPT 6.1 Sol, both at xhigh. Neither sees
the other's output. Each request runs in an empty temporary working directory;
custom instructions, memory, shell tools and other customizations are disabled.
Native outcome metadata, arm, seed, token totals and deterministic flags are held
in a coordinator-only key. Agent-visible feedback remains intact, including any
outcome information it inherently reveals. This is partial blinding.

Each successful response is stored with the raw CLI output, exact prompt, model
settings, CLI version, usage, hashes and elapsed time. Completed jobs are verified
and skipped on resume. A provider, parsing or schema failure stops new jobs for
that reviewer; in-flight jobs finish. Failed attempts stay intact. Inspect the
failure before explicitly supplying `--retry-failed`. Do not repeatedly rerate
valid responses to obtain agreement. There is no automatic model or paid-API
fallback and no automatic retry by this driver; the provider CLI may perform its
own transport retries, retained in its raw output.

The retained technical pilot is `runs/trajectory-review-20261103/`. Its 16 raw
responses were imported unchanged into the final v1 corpus after fixing decoded
HTML excerpt validation and recording provider non-reviews separately. Packet,
prompt and schema hashes are identical; no valid response was rerated. The
import receipt records this provenance. The final corpus also retains source
code in `runtime-snapshot/` alongside its original-file hashes.

For unattended continuation past **explicit provider refusals only**, use:

```bash
../.venv-test/bin/python -m trajectory_review.continue_census \
  --root runs/trajectory-review-20261103-v1 --reviewer claude --workers 8
```

This records the refused request as unavailable and moves to other trajectories.
It never resends that request, rephrases it or changes the model. Other provider,
capacity, quota, schema or integrity failures remain fatal. A manually inspected
non-review can be recorded with `trajectory_review.unavailable`; its labels are
null, not invented annotations. Provider refusals are distinct from semantic
unknown judgments. Missing/refused annotations prevent complete two-judge
coverage even if every request in the census has been attempted.

The two shell processes need network and existing CLI authentication. They do
not expose credentials in saved artifacts. Authentication/rate limits can leave
the census incomplete; never present a partial review as a full-corpus result.

## Evidence and consolidation

Every response contains all six construct labels, separate harm assessments,
located excerpts, alternatives, and an independent goal-alignment judgment.
All raw labels are preserved. Structural validation rejects malformed responses.
Missing or unmatched evidence is recorded separately, rather than silently
correcting the judge: affected labels must be treated as unverified/unknown in
the evidence-filtered consolidation, with raw results also retained.

Before interpreting results, require a result or explicit unavailable receipt
from each provider for every episode in the authorized scope. Distinguish this
request accounting from actual two-reviewer annotation coverage. Compare
full-population deterministic E1/E2 changes first, then the four
native-outcome groups and token differences within them. Report each judge,
agreement matrices, unknown/not-applicable coverage and disputed cases. Agreement
between models is not human validation or ground truth. Retain opportunity and
batching differences from the deterministic module. A subsequent code-module run
must use the same frozen input hashes and a new output directory.

The final report should distinguish joint support, disagreement, and insufficient
evidence. Keep disagreements and original rationales for human adjudication.
Do not infer intent, private reasoning or a causal optimization mechanism.
Fewer tokens, native success/failure and isolated tool errors are not automatic
semantic labels. The study is retrospective; existing findings informed its
question. Exact repeated outputs are not guaranteed by hosted model aliases.

Once every episode has a recorded result or explicit unavailability:

```bash
../.venv-test/bin/python -m trajectory_review.consolidate \
  --root runs/trajectory-review-20261103-v1 \
  --output runs/trajectory-review-20261103-v1/consolidated

../.venv-test/bin/python -m trajectory_review.deterministic_followup \
  --root runs/trajectory-review-20261103-v1 \
  --semantic runs/trajectory-review-20261103-v1/consolidated \
  --output runs/trajectory-review-20261103-v1/deterministic-followup
```

Consolidation refuses a census with unrecorded requests. Its summary retains raw
and evidence-checked label rates, unknown bounds, paired transitions, separate
observed harm, native-versus-independent goal alignment, agreement matrices and
per-run kappa intervals. Exact label agreement is reported separately and does
not resolve disagreements. `disagreements.jsonl` retains both rationales and
evidence for every disputed or evidence-invalid episode. The deterministic
follow-up requires the matching completed consolidation and preserves all four
native-outcome groups, including empty groups and longer E2 responses.

Read those complete outputs before updating the scientific report. Automated
counts and model agreement do not replace the coordinator's interpretation or
the subsequent human review. Token changes over all questions include outcome
changes and budget-bound failures; they are not pure efficiency estimates.

For a new corpus or a changed rubric, create a new output directory and preserve
the previous one. The driver refuses source/protocol changes during a run.

## Verification

```bash
../.venv-test/bin/python -m pytest tests/test_trajectory_review.py tests/test_trajectory_review_consolidate.py -q
```

Tests cover metadata blinding, full batch preservation, unsupported interfaces,
schema and evidence handling, immutable completed results and explicit recovery
from failed provider calls. They do not validate the semantic constructs.

CLI references: [Codex exec](https://github.com/openai/codex/blob/main/codex-rs/exec/src/cli.rs),
[Codex configuration](https://github.com/openai/codex/blob/main/codex-rs/config/src/types.rs),
[Claude structured output](https://code.claude.com/docs/en/headless), and
[Claude CLI flags](https://code.claude.com/docs/en/cli-reference).
