# The cosine length reward was measuring 8% of the completion

`model_token_count` dropped Qwen3's thinking blocks. Every run that enabled
`token_length` in a multi-turn or tool-calling setting - which is every agentic
run in this project - shaped on a near-constant instead of on length. Found
2026-08-03 at step 8 of e28, confirmed retrospectively in e25.

Fixed in `training/rewards/utils.py`; the run-level consequences are below.

## What the bug was

transformers' chat parser splits a Qwen3 response into fields. On any assistant
message that carries a tool call, the think block lands in **`reasoning_content`**
and `content` is left empty. Only a final, tool-less assistant message keeps its
`<think>` inline. Dumped from a live browsergym rollout:

```
roles     : ['assistant', 'tool', 'assistant']
msg keys  : [['content', 'reasoning_content', 'role', 'tool_calls'],
             ['content', 'name', 'role'],
             ['content', 'role']]
msg 0     : content 0 chars, tool_calls True
msg 2     : content "<think>\nOkay, so the user's task was to..."
model_token_count: [18, 1024]     completion_ids: [1024, 1024]
```

`model_token_count` summed `content` plus serialized tool-call arguments and
never read `reasoning_content`. For Qwen3 the think block is the generation and
everything else is a ten-token tool call, so it counted 18 tokens of 1024.

## How it presented

Not as an error. As a reward with no variance.

e28, first six steps, with `max_len` 4096:

| step | mean_length | cosine raw_mean | cosine raw_std | env raw_mean | implied tokens |
|---|---|---|---|---|---|
| 1 | 2342 | -0.7493 | 0.661 | 0.125 | - |
| 3 | 2728 | -0.9996 | 0.0002 | 0 | ~73 |
| 4 | 508 | +0.9999 | 0.00003 | 1.0 | ~37 |
| 6 | 2202 | -0.9998 | 0.0002 | 0 | ~52 |

Every step where all eight rollouts shared a correctness label has a cosine
standard deviation near 0.0002. The steps with real variance (1 and 2) are the
mixed-correctness ones, so what looks like a working reward is the correctness
gate talking, not length.

## It affected e9-e25 too, and that null cannot stand

e25 is the strongest case: the most recent cosine run, at the 4096 budget, built
as "the cosine length reward's fair test". Its `train_log.json` has the
per-component series. Restricting to the 166 steps where correctness was uniform
and the budget was not saturated, and inverting the logged cosine value back to
the token count that produced it:

```
actual mean_length   median  859
implied by cosine    median   71      ratio 0.083
pearson r(actual, implied)  +0.671
cosine raw_std       median  0.00017   (within-group length spread: 620 tokens)
```

The reward saw 8% of the length. The correlation is real but irrelevant at that
scale: 71 tokens against `max_len` 4096 is progress 0.017, which is the flat
short endpoint of the cosine. Doubling the true length there moves the reward by
about 0.001, against correctness differences near 1.7.

This is exactly the dead-zone failure e25's own config description identified and
set `max_len: 4096` to avoid - "the error was setting max_len to a cap far above
where completions actually live, because the cosine is flat near both endpoints
and the whole distribution would sit in the dead zone". The diagnosis was right.
The token counter recreated the condition from the other direction: not a cap too
high, but a measured length twelve times too small.

**So e9-e25 did not test the cosine length reward.** The "closed null across
weight 1-16 by composer {advantage_weighted, naive_sum}" tested a reward that
could not see what it was shaping. It should not be cited as evidence about
length shaping, and the e24/e25 "first directional compression" was noise - which
its own statistics already said (sign test p=0.066, diff-of-medians CI crossing
zero).

e9's log has no usable component series; it predates the per-component logging
fix (f589e71). e25 is decisive on its own and is the run the null rested on.

## Why the tests did not catch it

`tests/test_model_token_count.py` passed throughout. It exercised the function
against message shapes this pipeline never produces - assistant messages whose
`content` holds the model's text - using a whitespace-splitting fake tokenizer.
The function was correct for its tests and wrong for its only caller.

The regression test added with the fix is built from the dumped shape instead.

## Why reasoning from the source got it wrong

TRL picks between two paths (`grpo_trainer.py:1737-1745`): `parse_response` when
the tokenizer exposes a non-null `response_schema`, otherwise a plain
`batch_decode` into `{"role": "assistant", "content": ...}`. A freshly loaded
`Qwen/Qwen3-1.7B` reports `response_schema` as None, which predicts the fallback -
and the fallback puts the whole decoded string, think block included, into
`content`, where the counter would have read it correctly.

That prediction was wrong and two passes over the source did not reveal it. One
dump of the real object did, in a five-minute smoke.

Third time in this project that a measurement was taken under conditions the run
does not use, and the third time the fix was to look at the artefact rather than
reason about the interface: five browsergym tools against the adapter's two, a
2048 budget read per-turn when TRL applies it per-trajectory, greedy probes
against sampled training. This one is the most expensive, because it invalidates
a campaign rather than a selection.

## What to re-run

`token_length` is the only affected reward. `env_reward`, `non_termination` and
the eval path are untouched - eval counts `comp_ids.shape[0]` per turn and always
tracked TRL's own `mean_length`.

- e28 (E2, browsergym cosine) relaunches on the fix; it was killed at step 8.
- e29 (E3, non-termination) is unaffected and needs no change.
- Whether to re-run any poly cosine arm is a scope decision, not a correctness
  one. The environment there is saturated and the campaign was already superseded
  by the move to browsergym; the honest minimum is to stop citing its null.

## Reproducing

The dump harness is `/workspace/dump_completion.py` on the box: it monkeypatches
`CosineLengthReward.__call__` to write the first real completion object to
`/workspace/completion_dump.json`, then runs a `--smoke` train. It writes into
`runs/e28-*` with `--overwrite`, so that directory must be cleared before a real
e28 launch.
