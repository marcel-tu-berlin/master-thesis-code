# Offline off-target module: implementation validation

Date: 2026-10-15. Definition version: 1. Scope: the separate offline module in
`pipeline/offtarget/`, not a new experiment or a semantic substitution verdict.

## Evidence

The full local source list, SHA-256 hashes, per-family coverage and successful
comparison checks are in
[validation.json](offtarget-v1-20261015-reconciled/validation.json). Each comparison folder
also contains a receipt binding its episode inputs, owning experiment config,
stack stamp, evaluation report, analysis implementation/definitions, CPU package
versions, and generated outputs. No source episode/report was overwritten.

| Family | Episodes checked | Unknown episode measurements / observations | Prior audit agreement | Terminal-state/native-outcome agreement |
|---|---:|---:|---:|---:|
| read-table-2 | 600 | 0 / 0 | Not used | 474 completed episodes |
| email-inbox-noscroll | 600 | 0 / 0 | Not used | 521 completed episodes |
| tic-tac-toe | 600 | 0 / 0 | 600 episodes | Covered by prior campaign review |
| click-tab-2 | 400 | 0 / 0 | 400 episodes | Covered by prior campaign review |
| click-checkboxes-soft | 400 | 0 / 0 | 400 episodes | Exact selection is deliberately separate from native reward |

These are 2,600 saved E0/E1/E2 episodes for the first three families and E0/E1
episodes for tabs and checkboxes. Only completed local evaluation files were
used. No checkbox E2 result is inferred from its interrupted training run.

The previous audit comparisons were episode-by-episode, using their saved
`board_audit.json`, `tab_audit.json` and `checkbox_audit.json` records. Checked
overlaps were occupied-cell clicks, missed immediate wins, unblocked unique
threats, wrong visible links, leaving a visible target, incomplete checkbox
submissions and natively successful incomplete selections. Table and inbox final
commit checks were independently derived from saved state and compared with
native correctness. Earlier wrong inbox actions can be corrected and are not
required to equal the final outcome.

The 34 checkbox synonym groups were also compared exactly with the hashed saved
task source; group membership agrees. The existing within-group duplicate `pig`
does not change its meaning.

Zero parser unknowns does not mean every harmful behavior is observable. The
tactical/tab checks deliberately exclude later actions in a precommitted batch.
Opportunity and batching counts remain essential to interpretation.

For example, tic-tac-toe E1 has zero observed immediate-win/unique-threat
opportunities under the first-action definition, while E2 has 99/126. No pair
is exposed in both arms for these checks. Their occurrence difference therefore
cannot establish that E2 makes worse decisions at comparable tactical
opportunities. A behavior review must distinguish exposure changes from decision
quality changes in either direction.

## Reproducible reports

All five comparisons matched 200 identical initial observations, with no
unmatched seeds or missing-observation exclusions:

- [Table E2 versus E1](offtarget-v1-20261015-reconciled/read-table-2-e2-vs-e1/report.md).
- [Inbox E2 versus E1](offtarget-v1-20261015-reconciled/email-inbox-noscroll-e2-vs-e1/report.md).
- [Tic-tac-toe E2 versus E1](offtarget-v1-20261015-reconciled/tic-tac-toe-e2-vs-e1/report.md).
- [Tabs E1 versus E0](offtarget-v1-20261015-reconciled/click-tab-2-e1-vs-e0/report.md).
- [Checkboxes E1 versus E0](offtarget-v1-20261015-reconciled/click-checkboxes-soft-e1-vs-e0/report.md).

The last two measure specialization drift, not shaping drift. The reports do
not replace the original success/efficiency findings or the accepted tic-tac-toe
budget sensitivity analysis.

## Checks and limits

The focused suite has 46 CPU cases covering positive/negative event controls,
same-turn versus later-turn feedback, corrected errors, native-success/exact-goal
disagreement, absent terminal feedback, unknown states, identity changes,
malformed inputs, paired coverage, stable repeated output, hashes, and overwrite
refusal including checkpoint-parent protection. The full repository gate passed
formatting, lint, types, 561 CPU tests (7 existing skips), and all 17 setup-harness
checks. The CLI imports no
model, training engine or environment runtime.

The independent review identified checkpoint protection/provenance, missing
terminal feedback, stale state/evidence positions, sample-selection bias and
paired opportunity exposure. The implementation and review instructions address
these points, with edge controls in the focused suite.

This is implementation regression evidence. Prior audits share task rules with
the new code, and agreement with native outcomes does not validate semantic harm
or establish causality. The panel was developed retrospectively with access to
existing campaigns. Independent human/LLM annotation, detector-negative review,
and uncertainty/adjudication remain open under
[the review plan](../../docs/plans/off-target-human-llm-review.md).

## Reproduction and archive boundary

The saved source script is `pipeline/offtarget/reconcile_saved.py`. From
`pipeline/`, reproduce this fixed corpus with:

```bash
../.venv-test/bin/python -m offtarget.reconcile_saved --output runs/<new-directory>
```

The final reconciliation covers all 13 evaluations, writes all five paired
reports plus the separate E0 table/inbox/tic-tac-toe reports, and records 4,395
per-metric reference comparisons: 4,395 agreed, zero disagreed. Per-run/per-check
counts and mismatch locations are persisted, including explicit unknown coverage.
The multi-turn missing-terminal-feedback regression was also checked by removing
its guard in memory: both cases then failed, while repository files stayed intact.

The raw traces, prior audit files and generated JSON/Markdown reports are local
run-archive artifacts and remain ignored by Git under the existing policy. The
links above therefore require that archive; a source-only checkout will not
contain them. The source script and this findings note are not ignored. The
hashes below anchor the evidence in this durable write-up, so archived inputs
and results can be checked without assuming the local files stayed unchanged.
No archive, current experiment or existing report was overwritten.

Additional single-arm coverage receipts are under
`offtarget-v1-20261015-reconciled/read-table-2-e0/`,
`offtarget-v1-20261015-reconciled/email-inbox-noscroll-e0/`, and
`offtarget-v1-20261015-reconciled/tic-tac-toe-e0/`.

## SHA-256 anchors

Input and implementation paths below are relative to `pipeline/`. These are
byte hashes of the exact archived inputs and checked implementation.

| Input or implementation | SHA-256 |
|---|---|
| `offtarget/README.md` | `64d646d818f8bc32e0c4fb793ba67aff7b0f1d26cbb5c1ebf7ed120419f82b7d` |
| `offtarget/__init__.py` | `4d1b247bdc4555175a378937b7f2d068e028c93bdc0085f3a5aff8efe318aa18` |
| `offtarget/__main__.py` | `0dbac4d8bd82578eea5d46a2fb893960ee03e7d1a75c4ec85e498b86c4ecb28b` |
| `offtarget/analysis.py` | `3879126112aef563263093469009554e78c50560c8186e74315ff2a6cc02cb99` |
| `offtarget/checkbox_synonyms.json` | `1668e6e558719528b3244d95e0b817b4f3b83359c8bb80376ed79e0d7b30fab9` |
| `offtarget/reconcile_saved.py` | `92bcc3fe9846531598effb790ee1eba22eca4b81646f686cd4d8f77347a43416` |
| `offtarget/tasks.py` | `2c5f963fa583218529a4a4c5fecd9cd9ef39917210800b5bc2cde175f07e35ee` |
| `runs/e0-click-checkboxes-soft-s4232-c4096/checkbox_audit.json` | `40f7dd8fb18e43e5aaf64e4d627b9eca42d6d9efdd00e9725cac78a0eab96ef3` |
| `runs/e0-click-checkboxes-soft-s4232-c4096/episodes_held_out_click_checkboxes_soft.jsonl` | `40ab7755dc74394dbf0074a1119764c5abbdf5a342fad3cfd6ece84e938c40cb` |
| `runs/e0-click-tab-2-s4231-c4096/episodes_held_out_click_tab_2.jsonl` | `ad90afff82e38759c8e5c39773df9837a4f68e05059421a7d7caa4dd2ae99c50` |
| `runs/e0-click-tab-2-s4231-c4096/tab_audit.json` | `ecba00ddb3f5ad69893b8c93ae4fd789c7e8334fdf758164626ce8a543bcc86e` |
| `runs/e0-email-inbox-noscroll-s4021-c4096/episodes_held_out_inbox.jsonl` | `23fe4009d266e650b88c41d3a42e2f76c122f2213b04ff68ee09b3ada9556f19` |
| `runs/e0-read-table-2-s4016/episodes_held_out_table.jsonl` | `365809ea10445c042a970f060cda80d1ec277d14717a6019251d978417f20d91` |
| `runs/e0-tic-tac-toe-s4230-c4096/board_audit.json` | `42f7a4fb798cae6da7df1af2999e3e11580f2445d2a4e93e44b4d2f347e18bb2` |
| `runs/e0-tic-tac-toe-s4230-c4096/episodes_held_out_tic_tac_toe.jsonl` | `0b21de8b592a9b70896890aa770c55a45e757b66767a107c1aa210200783d910` |
| `runs/e1-click-checkboxes-soft-s4232-c4096/checkbox_audit.json` | `8dd634a19b6286d61d36c3fe3edb4303591b55ce58ba8fa7e4d855a2b585272c` |
| `runs/e1-click-checkboxes-soft-s4232-c4096/episodes_held_out_click_checkboxes_soft.jsonl` | `be6f359763ad97e7240246c79bc5c527a4d207305b7ce89f1e0999ccafe8cbd7` |
| `runs/e1-click-tab-2-s4231-c4096/episodes_held_out_click_tab_2.jsonl` | `6de0b381de2adcc700f3c74b3bcf9c5a22145e6dac81c23682cbcd8555443062` |
| `runs/e1-click-tab-2-s4231-c4096/tab_audit.json` | `ee06ca6e00ced85467a149cece67f4daaae822021a76dbdd5033a5f6178fb35c` |
| `runs/e1-email-inbox-noscroll-s4021-c4096/episodes_held_out_inbox.jsonl` | `9b68244a29651a13162ac4a340271ca956ec3652598dd6ef023d4c345cd5794a` |
| `runs/e1-read-table-2-s4016/episodes_held_out_table.jsonl` | `07019d1d8d3207f7c654ed6c114896620c58ce6b696cf36151c4ea43a937ef32` |
| `runs/e1-tic-tac-toe-s4230-c4096/board_audit.json` | `4bf1890a9d7492e7cb128b62cd23b4d770b2864ffddf9f102f87b2a2700b22ce` |
| `runs/e1-tic-tac-toe-s4230-c4096/episodes_held_out_tic_tac_toe.jsonl` | `01ed15f65eba7461e43f928467a856e6714fda2f001972c4c038c9fdaa3d107b` |
| `runs/e2-email-inbox-noscroll-s4021-c4096/episodes_held_out_inbox.jsonl` | `49b2bc3d9107b0f514cffb2bc26980e3661c98eabca88d38ce84b1edbdd1034b` |
| `runs/e2-read-table-2-s4016/episodes_held_out_table.jsonl` | `ae55ea79849f4cdfe57e8abe69ee6b0df6acabf6fd2d6e4ac9641a9113005795` |
| `runs/e2-tic-tac-toe-s4230-c4096/board_audit.json` | `0a66348d1c736db09537eb98e5010a2ba151043003c8ef9513273452eb817c22` |
| `runs/e2-tic-tac-toe-s4230-c4096/episodes_held_out_tic_tac_toe.jsonl` | `464fa9cf49d0bfec7ef5444f2d69f718c042c190a7f330ea3b90ef81d88b3538` |
| `tests/test_offtarget_analysis.py` | `c363c22ed8160b86c96e1ed103929ece66c742b431421bf2903ead0ffac86134` |

The following paths are relative to the reconciled output directory. Each
receipt also binds its report and event files to the source hashes and runtime.

| Evidence manifest | SHA-256 |
|---|---|
| `validation.json` | `945611f0f63145144f8d697978a1d7f31b5cc17ac9750d179ae09a2ff8189497` |
| `click-checkboxes-soft-e1-vs-e0/receipt.json` | `9158153a140e8dd30cd0924293af8e7b6a899bf202dc57d5b5fc8648d7b71617` |
| `click-tab-2-e1-vs-e0/receipt.json` | `1acd3f10256ee80e440ea4b1805de3ec592841d5a18e18cd9bef9f204c6e14c3` |
| `email-inbox-noscroll-e0/receipt.json` | `37159b524c46af28635d5b9fecb0c40e14887a3dc0d9b32cf8c94c69a8ea921e` |
| `email-inbox-noscroll-e2-vs-e1/receipt.json` | `94705818af5387fed55cedbcbd8e732feda03f5a7779136e5a05e8fb0c1a1792` |
| `read-table-2-e0/receipt.json` | `2f0ae1e9fbf136a4d3962adfef8ae40beffa4a64c8f5f0b865d103f8f90d8d60` |
| `read-table-2-e2-vs-e1/receipt.json` | `26961a453755cabba47d5bdb18debfa29296d65833b6d2be419b930a48f8bfa6` |
| `tic-tac-toe-e0/receipt.json` | `ac7ecbf593569bea9eaed49fba57b5542ea6669059fdc5457aa1677b1237b554` |
| `tic-tac-toe-e2-vs-e1/receipt.json` | `09be3873f4fc41039060518d56acf2e2b99b7046570afcb77f465fc28312d18d` |
