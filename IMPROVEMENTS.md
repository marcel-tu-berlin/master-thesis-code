# Improvements under consideration

Options for extending the campaign beyond Qwen3-1.7B, written down 2026-08-18
while the e33-e36 lambda sweep was still running. Nothing here is decided or
scheduled. Revisit after the sweep is harvested and the seed replication has
answered whether the w0.25 anomaly is seed noise (the current reading: it is -
compression saturates at the lowest lambda, KL drift is anti-monotone in
lambda, and this campaign has already seen single-seed medians swap). This
file is standalone on purpose: no entry here is tracked in taskwarrior or
BACKLOG.md until it is picked up.

The constraint behind every option: one L4 with 24 GB. Model choice is further
bounded by TRL's chat-template whitelist (qwen*, llama3_1/3_2, and three
models that do not fit the card), and the logits memory of the policy update
is vocab-bound, not model-bound: four [T, 151936] bf16 tensors, ~4.6 GiB at
T=4096, the same for every Qwen3 size.

## Model scale

- **Qwen3-4B base eval probe.** The registry uses the post-trained lineage. The
  old candidate config is archived and must be recreated from the accepted
  recipe when this work is scheduled. The eval is cheap (bf16, no vLLM).
  Decides whether click-menu-2 stays inside the 40-80% band at 4B; if it
  saturates, the band check reopens with a different task pool before any 4B
  training.
- **Qwen3-4B trained contrast.** One replication of the headline pair (E1 vs
  one shaped arm) at 4B, bf16 everywhere, own e0. Directly answers "is 1.7B
  too small". Blocked on a memory lever (next section); roughly 2-2.5x the
  1.7B step time.
- **Qwen3-0.6B down-rung.** Cheap. Three scale points (0.6 / 1.7 / 4B) make a
  trend; two make an anecdote.
- **Qwen3-8B: not now.** bf16 never fits; the only route is QLoRA plus a
  quantized vLLM copy plus Liger, which stacks two open correctness questions
  on top of a quantization confound, at 2-3x step time. Future-work section
  material.

## Memory levers (what unlocks 4B training and a 16k cap)

- **Liger fused GRPO loss.** Rejected for the current campaign by decision
  0011: the installed DAPO path changes the loss denominator and gradients.
  Revisit only after identical-input loss and gradient parity, then run the
  16k memory check.
- **Drop vLLM.** `model.use_vllm: false` already works in grpo_runner. Frees
  the duplicate weight copy, which alone makes 4B bf16 fit without Liger, and
  is scientifically cleaner: the same weights generate and train, so the
  whole sampler-mismatch/ISR machinery becomes unnecessary. Cost lands on
  generation, estimated 5-20x slower (not measured); that multiplies every
  arm and eats the seed budget, and a no-vLLM campaign differs from the 1.7B
  campaign in engine as well as scale.
- **transformers-paged backend.** TRL's third generation path
  (`use_transformers_paged`): continuous batching on the single trainer
  weight copy. Possibly the middle ground; unverified with the tool loop.
- **Decide by measurement, not estimate.** When the GPU is free, `--smoke`
  the 4B config three ways (vLLM+Liger, transformers-paged, plain generate)
  and read the s/it. Minutes each.

## Precision knobs (parked)

QLoRA nf4 on the trainer and fp8 on the vLLM copy both fit the card and both
carry the same problem: the sampler and the trainer would run different
numerics, the exact drift class the TIS audit closed. Quantization is a
treatment, never free memory - it would have to be held fixed within any
compared pair. Also unverified whether TRL's colocate weight sync works into
a quantized vLLM copy at all. Only worth revisiting if 8B ever becomes worth
its cost.

## Cross-lineage replication

Llama-3.2-1B is already in the registry (template verified byte-identical to
TRL's). One replication of the headline contrast there answers the "is this a
Qwen artefact" objection, and the model has no thinking mode, so it also
tests whether length shaping works when there is no reasoning trace to
compress. bf16 fits like 1.7B. Arguably worth more to the thesis than any 8B
number.

## Ruled out

- Switching the eval engine (HF greedy generate) - kernel numerics can flip
  greedy argmax and would orphan every harvested report.
- Lowering the completion budget to save memory - reintroduces the truncation
  confound that voided e9-e21.
- Dropping mixed precision / fp32 LM head - already a documented dead end in
  BACKLOG.md.
- 8B as a headline result - see above.

## Suggested order, if all of it happens

The final 1.7B campaign comes first, then the 4B probe, then any measured
backend comparison, one 4B contrast, and finally Llama-3.2-1B. Stop wherever
the thesis timeline says stop; each step stands alone.
