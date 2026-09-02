# e30-e36 (the menu-2 campaign) read through the training logs

Date: 2026-09-01. Same seven runs as `e30_gate_findings.md`,
`e31_e32_arms_findings.md` and `e33_e36_lambda_sweep_findings.md` - no new GPU
time. What is new is that the per-step training logs and the per-episode records
are now read by committed code (`eval.paired`, the overlay and dose figures in
`eval.plots`) instead of by throwaway snippets, so the campaign can be asked
questions the aggregate report cannot answer.

Nothing here re-scores an episode. Every number the three findings files already
published re-derives exactly from `eval.paired` (e31 14/0 losses p=1.2e-4, paired
median -1130 on 186 jointly-correct episodes; e32 54/0 p=1.1e-16; e32's shifted
click-dialog-2 18/3 p=1.5e-3), which is the point of committing it: those numbers
are now reproducible rather than remembered.

Artifacts: `runs/plots_menu_campaign/` holds `training_overlay.png`,
`dose_response.png`, `paired_deltas.png`, the per-split comparison figures, and
the two generated tables `paired_held_out.md` / `paired_shifted.md`.

## The noise floor, so the deltas can be read against something

Last 30 of 150 steps, mean +- SD across steps (each step already averages 32
rollouts):

| arm | env reward | completion length | KL | entropy | grad norm | frac steps live |
|---|---|---|---|---|---|---|
| e30 (E1) | 0.891 +- 0.095 | 1884 +- 166 | 0.096 | 0.185 +- 0.005 | 0.063 +- 0.031 | 0.880 |
| e33 (E2 w0.25) | 0.860 +- 0.095 | 488 +- 59 | 0.564 | 0.104 +- 0.007 | 0.342 +- 0.034 | 1.000 |
| e34 (E2 w0.5) | 0.908 +- 0.081 | 578 +- 98 | 0.461 | 0.089 +- 0.005 | 0.350 +- 0.036 | 1.000 |
| e31 (E2 w1.0) | 0.922 +- 0.074 | 690 +- 107 | 0.359 | 0.137 +- 0.009 | 0.263 +- 0.034 | 1.000 |
| e35 (E3 w0.25) | 0.859 +- 0.119 | 1932 +- 335 | 0.139 | 0.172 +- 0.004 | 0.074 +- 0.044 | 0.900 |
| e36 (E3 w0.5) | 0.878 +- 0.097 | 2022 +- 183 | 0.120 | 0.242 +- 0.006 | 0.070 +- 0.032 | 0.933 |
| e32 (E3 w1.0) | 0.806 +- 0.153 | 2288 +- 247 | 0.097 | 0.206 +- 0.005 | 0.066 +- 0.025 | 0.953 |

Read three things off it.

- **Task learning is not what separates the arms.** Every arm's tail env reward
  sits inside one SD of e30's. The e32 write-up's "~5% weaker env learning" is
  0.806 against 0.891 with SDs of 0.15 and 0.10 - it does not clear the run's own
  step-to-step noise, so e32's 27pt held-out cost cannot be attributed to weaker
  task learning on this evidence.
- **The length separation is not noise.** 488-690 against 1884 +- 166 is six
  SDs. The E2 compression is the one effect in this campaign that no amount of
  seed variance explains away.
- **E2 trains in a different optimization regime.** Grad norm 4-6x e30's, entropy
  collapsed to 0.09-0.14 against 0.185, KL 0.36-0.56 against 0.096, and zero
  prompt-groups without reward variance at any lambda. E3 sits on top of e30 on
  all four.

## E2 is a switch, and the mechanism is now visible

`CosineLengthReward` contributes to the advantage in exact proportion to its
weight - `contrib_l1` 0.246 / 0.489 / 0.967 for lambda 0.25 / 0.5 / 1.0, a clean
1:2:4 - while the outcome does not move with it at all:

| lambda | contrib_l1 | held-out losses | paired median dtok | 95% CI |
|---|---|---|---|---|
| 0.25 | 0.246 | 26 | -1273 | [-1298, -1224] |
| 0.5 | 0.489 | 14 | -1234 | [-1235, -1198] |
| 1.0 | 0.967 | 14 | -1130 | [-1161, -986] |

Quadrupling the shaping term's share of the reward buys nothing further, and the
largest compression comes from the smallest lambda. The three CIs barely overlap
each other but all sit ~1200 tokens below zero: the arms differ from the control
enormously and from each other marginally. That is the definition of a switch,
and it now has a mechanism attached rather than being an observation about three
end-of-run numbers.

Two more things the per-episode view adds:

- **Compression is uniform, not driven by a subset.** 100% of the jointly-correct
  episodes are shorter than their e30 counterpart in all three E2 arms (186/186,
  174/174, 186/186). No subpopulation is doing the work.
- **KL is non-monotone in lambda and largest at w0.25** (0.564 > 0.461 > 0.359).
  Whatever moves the policy is triggered by turning the cosine on, not by how
  hard it is turned; the confound flagged in `e31_e32_arms_findings.md` therefore
  applies to the whole E2 column and cannot be dialled down by using a smaller
  weight.

## E3 is a dial whose penalty goes inert early

`NonTerminationPenalty` contributes 0.272 / 0.125 / 0.070 (w1.0 / 0.5 / 0.25)
over the first ten steps and 0.034 / 0.017 / 0.008 over the last thirty. The
penalty has taught what it can within roughly the first 40 steps and is then
nearly silent - so E3's held-out cost is not a component still pressing on the
policy at step 150; it is the residue of an early behavior change.

The off-target movement is real and directional. Stop-reason transitions on the
shifted split, following individual episodes (which the aggregate stop-reason
counts cannot do):

| arm | no_tool_call -> env_done | env_done -> no_tool_call | env_done -> hit_generation_cap |
|---|---|---|---|
| e32 (w1.0) | 19 | 3 | 0 |
| e36 (w0.5) | 17 | 5 | 1 |
| e35 (w0.25) | 7 | 11 | 3 |

e32 and e36 convert stalled episodes into finished ones on families they never
trained on; e35 moves the other way, which is the global degradation the sweep
write-up flagged and could not localise.

Unlike E2, E3's token effect is mixed per episode: 52% (e32), 74% (e35) and 57%
(e36) of jointly-correct episodes are shorter, the rest longer. The pooled median
is a small net, not a uniform shift.

**A new caveat for E3.** Its arms push more completions into the generation cap
than the control does (`completions/clipped_ratio` 0.081 / 0.022 / 0.050 for
w1.0 / 0.5 / 0.25, against e30's 0.016 and E2's 0.000). Truncated completions
being mislabelled as behavior is the confound that voided e9-e21, so the E3
held-out cost carries a truncation component that has not been separated. It is
small in absolute terms and it is on the training side, not the eval side, but it
belongs next to the number.

## The gradient-share confound, per arm

`frac steps live` (steps where at least one prompt-group had reward variance) is
the yardstick LAB_NOTES prefers over the mean of `frac_reward_zero_std`:

- e30: 0.880 live, mean zero-std fraction 0.600
- E2, every lambda: 1.000 live, mean zero-std fraction 0.000
- E3: 0.900-0.953 live, mean zero-std fraction 0.51-0.53

So the "reward plus more gradient" contrast is an E2-column property at every
weight, and is essentially absent for E3, whose gradient share sits within a few
points of the control's. Any E2-vs-E1 claim in the write-up needs the sentence;
an E3-vs-E1 claim does not.

## What this still cannot settle

- **One seed.** Everything above is seed 42. The noise floor here is
  within-run step-to-step variation, which bounds a delta from below but is not a
  substitute for between-seed variance.
- **The e30 ceiling.** Held-out accuracy 1.000 means every arm can only tie or
  lose there; the flip counts stay the honest readout.
- **No trajectory text for these runs.** The eval loop now records per-turn
  reasoning, content and calls, but these seven runs were evaluated before that
  existed, so "which tokens did E2 remove" is answerable only for runs evaluated
  from here on. That is the one question in this file that needs GPU time to
  close, and it needs it only if a re-eval of e30/e31 is judged worth ~4h.
