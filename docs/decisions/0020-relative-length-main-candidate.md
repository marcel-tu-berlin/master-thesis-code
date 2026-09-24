# Continue with the relative successful-response length cost

Decided: 2026-09-24. After reviewing the matched continuation result and both
formulas, the user selected the relative design as the main research candidate.
Retain the tested weight 0.1 and successful-response-only sigmoid cost, with
within-question population standard deviation and its one-token floor. Retain
mean-only GRPO advantages (`naive_sum`, `scale_rewards: none`).

The evidence is the development comparison in
pipeline/runs/table_compression_s4009_findings.md. It supports taking this recipe
forward; it does not establish superiority over linear training, which has not
been run. Linear remains an optional future reward-design comparison.

Uniform rewards give no within-group policy-gradient signal. A single success
still gives a correctness contrast against failures; it supplies no successful
length comparison. This property does not call for changing the accepted reward.

Preserve the completed experiment and its declaration. The next research work is
confirmation on fresh evaluation data and matched seed replication, with the
question allocation and comparisons declared before execution.
