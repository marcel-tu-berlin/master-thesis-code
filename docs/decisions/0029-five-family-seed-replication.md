# Five-family seed replication

Amended 2026-11-18 by user request. Retain the original seed and the first
additional seed for every family. The second additional seed was withdrawn
for all five families before any round-2 launch. Finish the 20 round-1 cells
(five E0, five E1, five E2 and five matched placebos), then report all five
original/replication pairs with their original endpoints and limitations.
This decision was made after some first-round outcomes had been reviewed.

## Original authorization

Decided 2026-10-25 by explicit user request and scope confirmation.

Run fresh E0, E1, E2 at 0.1 and matched 0.1 placebo for two additional seeds
of each completed family: table, inbox, checkboxes, tic-tac-toe and tabs.
This authorizes 30 training runs plus 10 E0 evaluations, with automatic
advancement after technical checks. Run only on L4 GPU 1. Do not contact or
use the desktop until the user releases it; GPU 0 remains excluded.

The exact seed allocation, order, fixed recipe, comparisons, failure rules,
and monitoring live in ../plans/seed-replication-20261025.md. Preserve original
results and frozen family semantics. This supersedes earlier seed-replication
holds only for these 40 new cells. Other doses, models, families and E3 remain
deferred. Stop on an unexpected runtime/integrity/audit failure, with no retry,
overwrite, restart, sample extension or favorable-outcome selection.
