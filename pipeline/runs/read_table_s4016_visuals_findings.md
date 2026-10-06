# Read-table-2 seed 4016: visual guide

Created 2026-10-06 from the retained, reviewed E0/E1/E2 evidence. This is an
offline presentation of the existing comparison; no experiment or measurement
was changed.

The [five-page PDF](e0-e2-campaign-ops/visuals-20261006/read_table_2_e1_e2_behavior.pdf)
uses the same visual style and explanatory structure as the
[inbox PDF](inbox-campaign-s4021-c4096-ops/deep-review-20261005/inbox_e1_e2_behavior.pdf):

1. Success and response length across checkpoints, with the final paired result.
2. Token components and changes in actions and decision turns on jointly solved tasks.
3. Variation in table content, fixed action IDs, and batching across checkpoints.
4. One successful trace comparison and both final E2 regressions.
5. Sampled training success, reward variation and measured update duration.

Pages 1 and 5 reuse the preserved evaluation and training figures without
regeneration. The other pages derive from the existing deep-review diagnostics
and final saved trajectories. E1 and E2 start independently from the original
base; update 300 remains the primary endpoint.

The [original findings](e0_e2_s4016_findings.md) and
[deep review](e0_e2_s4016_deep_review_findings.md) contain the interpretation,
statistics and limitations. Both remain unchanged.

The [builder](e0-e2-campaign-ops/visuals-20261006/build_pdf.py) verifies the 87
unique inputs bound by the original run receipts and the prior deep-review
outputs before creating the PDF. Its new
[receipt](e0-e2-campaign-ops/visuals-20261006/analysis_receipt.json) binds the
inputs and resulting PDF. Reproduction requires the locally retained run
evidence, the existing CPU test environment, and Poppler (`pdfunite`, `pdfinfo`).
The receipt records the Python, Matplotlib, NumPy and Poppler versions. Hashes
identify the exported files; a later export can have different PDF metadata and
hashes while preserving the numbers and rendered content. Reproduction is
checked by the builder's assertions and visual inspection, not byte equality
with an earlier export.

```sh
XDG_CACHE_HOME=/tmp/table-pdf-cache MPLCONFIGDIR=/tmp/table-pdf-mpl .venv-test/bin/python pipeline/runs/e0-e2-campaign-ops/visuals-20261006/build_pdf.py
```
