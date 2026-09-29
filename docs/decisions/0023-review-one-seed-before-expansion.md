# Review one E0-E2 comparison before expanding the study

Decided: 2026-09-24, at the user's explicit correction while E0 was running.

Automatic execution is limited to E0, E1 and E2 with relative successful-response
length weight 0.1, all at seed 4016. E1 and E2 independently start from the same
original base. Keep the existing 300 updates, 200 held-out questions, scheduled
checkpoints at 100/200/300, reward definitions, seed allocation and analysis.

After these three runs are harvested and scientifically reviewed, stop and present
the comparison for the user's decision. This stop applies to positive, adverse,
null and inconclusive outcomes alike. The initial result is meant to establish
whether the experimental setup produces interpretable evidence before spending
resources on a broader study. Integrity or runtime failures still stop progression
earlier for diagnosis.

Decision 0022's additional weights 0.05/0.2, the separate 0.2 placebo training
run, and seeds 4017/4018 are deferred. Their prepared configs are retained, but
none is authorized for automatic launch. Any expansion requires a new explicit
user decision after the initial comparison; completion alone does not admit it.
E3 remains separately deferred.

The placebo is its own training condition. It uniformly shuffles only the shaped
length-cost values within each group of eight responses to the same question,
including failure zeros, while keeping task rewards attached to their responses.
It tests whether assigning the cost to the actual response matters. It is neither
part of the ordinary E2 run nor an extra evaluation automatically attached to it.

This amendment narrows execution scope and does not alter or invalidate E0's
measurements. Preserve its original launch manifest and script when restricting
the active operations manifest. One seed provides no training-seed uncertainty
estimate; state that limitation in the initial review.
