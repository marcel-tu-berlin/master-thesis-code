# Five families, two seeds each: final replication findings

Table reading gave the clearest repeatable efficiency result. Relative to the matched placebo, E2 used fewer assistant tokens on jointly solved questions and preserved success at both table seeds. The same conclusion did not hold across both references in the other families. Inbox compression repeated but success preservation changed with the seed. Checkbox compression came with fewer exact completions at both seeds. Tic-tac-toe depended on the fitted baseline and the execution boundary. On tabs, the original success collapse disappeared at the second seed, where E2 instead took substantially more tokens than either reference.

## Final checkpoints

E0 is the base model, E1 uses task reward, E2 adds the 0.1 successful-length penalty and P shuffles the matched 0.1 cost within prompt groups. The comparisons below pair all 200 held-out questions within each seed. Each reference column gives **joint successes out of 200, median paired token change, and gross losses out of 200**. Negative token change means shorter successful responses for E2. A gross loss means that the reference solved the question and E2 failed. Net success can improve even when preservation fails.

| Family | Seed | E1 / E2 / P successes | E2 against E1: joint, change, losses | E2 against P: joint, change, losses |
|---|---:|---|---|---|
| Table | 4016 | 199 / 198 / 189 | 197, -44.93%, 2 | 188, -79.64%, 1 |
| Table | 4017 | 197 / 192 / 191 | 191, -92.01%, 6 | 187, -89.32%, 4 |
| Inbox | 4021 | 183 / 199 / 196 | 182, -91.96%, 1 | 195, -83.93%, 1 |
| Inbox | 4240 | 199 / 168 / 189 | 168, -74.38%, 31 | 162, -44.24%, 27 |
| Checkboxes | 4232 | 191 / 162 / 183 | 155, -43.53%, 36 | 152, -81.01%, 31 |
| Checkboxes | 4242 | 193 / 184 / 186 | 177, -39.73%, 16 | 172, -25.54%, 14 |
| Tic-tac-toe | 4230 | 85 / 66 / 58 | 66, -23.09%, 19 | 32, -6.28%, 26 |
| Tic-tac-toe | 4244 | 5 / 58 / 70 | 5, -56.96%, 0 | 57, -69.44%, 13 |
| Tabs | 4231 | 189 / 5 / 124 | 5, -15.72%, 184 | 5, +85.09%, 119 |
| Tabs | 4246 | 180 / 181 / 164 | 164, +41.89%, 16 | 152, +75.45%, 12 |

The original preservation margin requires the one-sided exact 95% upper bound on gross loss to be at most 5%. Four losses in 200 pass (upper bound 4.52%), five fail (5.18%). The compression criterion requires the upper 95% endpoint on the median paired percentage token change to be below -10%. The question-paired intervals and gains/losses for individual contrasts are in the linked family analyses below. A failed upper-bound test leaves preservation unestablished. The ten rows are separate seed-specific observations. No cross-family effect or training-seed confidence interval is estimated.

## What replicated within each family

**Table.** Both seeds passed compression against both references. Success preservation against P also repeated. Against E1 the second seed lost six successes and gained one, leaving its 5.83% gross-loss upper bound above the margin. E1's own mean tokens per success changed from 464 to 1,772 between seeds, so the size of the compression effect depends substantially on the fitted reference. The original four generic off-target bounds passed; newly introduced wrong submissions prevented a repeat at 4017. [Original trio](e0_e2_s4016_findings.md), [original placebo](placebo_read_table_2_s4016_findings.md), [replication](seed_replication_read_table_2_s4017_findings.md).

**Inbox.** The original seed passed both primary success and compression criteria against both references. At 4240, E2 still shortened jointly successful responses, but lost 31 E1 successes without a gain and lost 27 placebo successes with six gains. Relative to P, the net success difference changed from +1.5 to -10.5 percentage points. Original E2 had invalid actions on 200/200 and repeated actions on 66/200 final episodes. At 4240 E2 had 52 invalid-action episodes and 19 repeat episodes, and the invalid/repeat bounds still failed. The forward-body audit uses an E0 reference plus each arm's saved checkpoints; its evidence covers observable recipient, action and body outcomes. [Original trio](inbox_s4021_c4096_findings.md), [original placebo](placebo_email_inbox_noscroll_s4021_findings.md), [replication](seed_replication_email_inbox_noscroll_s4240_findings.md).

**Checkboxes.** The native score permits partial matches. Exact completion requires submitting the entire correct selection. In E1 / E2 / P order the exact counts were **139 / 62 / 156** at 4232 and **134 / 96 / 142** at 4242. E2 shortened joint native and joint exact successes at both seeds while failing native preservation against both references. At 4242 the native E2-P difference was just -1 point but the exact deficit was -23 points. The original exact deficit against P was -47 points. At 4232 the wrong-submission and voluntary-stop bounds failed against both references; invalid and repeated actions passed. At 4242 wrong submission still failed against both references, while the other three bounds passed. These generic measures do not capture all omitted or irrelevant selections. Original E1 ran on L4 while E2 and P ran on the home RTX 4090. [Original trio](click_checkboxes_soft_s4232_findings.md), [original placebo](placebo_click_checkboxes_soft_s4232_findings.md), [replication](seed_replication_click_checkboxes_soft_s4242_findings.md).

**Tic-tac-toe.** Native wins are primary. At 4230 E2 compressed relative to E1 but lost 19 wins; against P it met neither primary margin. At 4244 the final E1 fell from 58 wins at step 200 to five at 300. E2 passed both criteria against this five-win reference. E2 compressed against P but lost 13 wins and gained one. The saved-prefix sensitivity stops before further actions once the budget is exhausted, allowing the final feedback to overshoot and retaining already-generated assistant tokens. At 4230 this gives E1 / E2 / P wins of 78 / 66 / 37; neither primary criterion passes against P (24 joint wins, -11.40% with upper CI -5.37%; 13 losses). Against E1 at 4230 the same cutoff gave one gain, 13 losses (upper bound 10.14%) and 65 joint wins with -23.37% median token change [-27.86%, -18.16%]. Compression passed and preservation failed. At 4244 E1 retained five wins and E2 retained 58, so the E2-E1 verdict still passes both criteria on five joint wins. At 4244 the 17 excluded P wins change the native 58 versus 70 result to 58 versus 53; five gains and zero losses then pass preservation and compression against P. These are two estimands applied to the same saved traces. Original P used home hardware, and cross-host layout and hit testing were not independently established. [Original trio](tic_tac_toe_s4230_findings.md), [original placebo](placebo_tic_tac_toe_s4230_findings.md), [replication](seed_replication_tic_tac_toe_s4244_findings.md).

**Tabs.** Neither seed established compression or preservation against either reference. At 4231 E2 regressed on 184 E1 successes and 119 P successes. Only five successes remained in either paired token comparison; the E2-E1 interval ran from -27.80% to +355.75%. At 4246 E2 solved one more question than E1 and 17 more than P, with materially longer successful responses. Original E2 repeatedly generated rejected identifiers. At the second seed, final E2 made two action errors in 540 observed opportunities and had nine wrong visible-link events in 190 attributable terminations. At 4246 E1 and P had one and eight action errors in 393 and 363 opportunities respectively, and 11/191 and 27/191 wrong visible-link events among attributable terminations. Wrong termination, voluntary stop and repeat bounds all failed for E2 against both references. Only the invalid-action bound passed. Those opportunities depend on what each policy reaches. The original catastrophic loss of success does not generalize to this second seed. [Original trio](click_tab_2_s4231_findings.md), [original placebo](placebo_click_tab_2_s4231_findings.md), [replication](seed_replication_click_tab_2_s4246_findings.md).

## Checkpoint trajectories

Native successes out of 200. Entries give steps 100 / 200 / 300 in training order. E0 uses the same question block as that seed's trained arms. Step 300 was fixed as primary before analysis. The intermediate checkpoints show when effects change direction.

| Family | Seed | E0 | E1 | E2 | P |
|---|---:|---:|---|---|---|
| Table | 4016 | 43 | 183 / 197 / 199 | 184 / 199 / 198 | 193 / 185 / 189 |
| Table | 4017 | 69 | 169 / 198 / 197 | 192 / 196 / 192 | 149 / 192 / 191 |
| Inbox | 4021 | 103 | 128 / 170 / 183 | 154 / 150 / 199 | 164 / 190 / 196 |
| Inbox | 4240 | 113 | 128 / 182 / 199 | 190 / 196 / 168 | 167 / 196 / 189 |
| Checkboxes | 4232 | 125 | 180 / 195 / 191 | 185 / 183 / 162 | 196 / 189 / 183 |
| Checkboxes | 4242 | 127 | 184 / 188 / 193 | 193 / 172 / 184 | 177 / 177 / 186 |
| Tic-tac-toe | 4230 | 8 | 61 / 76 / 85 | 71 / 66 / 66 | 66 / 71 / 58 |
| Tic-tac-toe | 4244 | 5 | 48 / 58 / 5 | 57 / 35 / 58 | 58 / 70 / 70 |
| Tabs | 4231 | 47 | 148 / 179 / 189 | 73 / 56 / 5 | 76 / 89 / 124 |
| Tabs | 4246 | 52 | 161 / 179 / 180 | 87 / 152 / 181 | 83 / 98 / 164 |

## Measurement, costs and scope

The following are recorded GPU allocation-hours. E0 is evaluation only. Each trained arm shows training plus the sum of its three evaluations as separate numbers. Comparing hours across GPUs, kernels or collection dates cannot isolate an efficiency effect. These amounts exclude downtime, analysis and storage.

| Family | Seed | E0 eval | E1 train + eval | E2 train + eval | P train + eval |
|---|---:|---:|---|---|---|
| Table | 4016 | 3.99 | 28.13 + 7.47 | 22.58 + 5.01 | 30.78 + 9.85 |
| Table | 4017 | 3.77 | 28.39 + 11.13 | 23.02 + 3.67 | 29.72 + 11.02 |
| Inbox | 4021 | 2.71 | 38.23 + 11.76 | 33.03 + 4.69 | 33.65 + 8.55 |
| Inbox | 4240 | 2.71 | 35.35 + 11.92 | 29.33 + 4.90 | 29.06 + 6.07 |
| Checkboxes | 4232 | 1.97 | 24.13 + 7.62 | 13.23 + 1.88 | 14.67 + 3.15 |
| Checkboxes | 4242 | 2.03 | 26.63 + 9.19 | 22.04 + 5.48 | 31.99 + 12.88 |
| Tic-tac-toe | 4230 | 3.93 | 37.98 + 12.05 | 41.79 + 11.38 | 16.82 + 3.26 |
| Tic-tac-toe | 4244 | 3.84 | 46.88 + 16.87 | 44.35 + 12.16 | 41.90 + 15.32 |
| Tabs | 4231 | 3.01 | 31.22 + 7.22 | 39.27 + 10.94 | 37.89 + 14.76 |
| Tabs | 4246 | 3.02 | 30.39 + 7.59 | 33.85 + 7.92 | 35.46 + 8.04 |

All four arms in each new quartet ran on the L4. Original table and inbox comparisons span kernel/run-date differences. During original tic-tac-toe E1 training on kernel 6.8.0-139, an unattended upgrade installed NVIDIA packages 580.178.04 while the loaded kernel module stayed at 580.173.02. NVML reported 580.178. E2 subsequently ran with matching driver 580.178.04 and kernel 6.8.0-146. Original checkbox E2/P and tic-tac-toe P used the home RTX 4090/Arch. Tabs had matching original L4 platform stamps. The new matched hardware improves interpretation of the second seed only.

The paired analysis conditions token counts on joint successes; lost successes have no token value in that statistic. Each successful response includes assistant content, reasoning and serialized call arguments. The budget charges interleaved feedback as well. Generic off-target margins apply a one-sided exact 98.75% upper bound of 5% for each of four new-event rates. The linked family reports contain the full off-target panels at each seed, including paired transitions and the opportunity denominators behind task-specific events. Task-grounded labels condition on observable opportunities. Unknown execution, unobserved alternatives and calls committed before feedback limit interpretation. Independent semantic validation remains a separate task. Shuffling the same cost values within a group changes their assignment but cannot hold gradient noise, future trajectories or the distribution of visited states constant.

The user withdrew the entire second additional round on November 18, after some first-round results had been reviewed and before any second-round cell launched. Every original and first-replication seed is retained. The L4 stopped at the verified withdrawal boundary after all **20 authorized cells** and all **50 scheduled 200-episode evaluations** of the replication. The original outcomes were known before replication was declared. Changing the seed also changes the training questions and held-out block. Within-seed episode intervals quantify uncertainty over questions for the fitted policies. Two full seeded experiments describe agreement and disagreement within each family and provide limited evidence about variation across training seeds. [Declared plan](../../docs/plans/seed-replication-20261025.md) and [scope decision](../../docs/decisions/0029-five-family-seed-replication.md).
