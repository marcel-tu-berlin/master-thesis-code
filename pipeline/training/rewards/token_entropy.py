import torch
import torch.nn.functional as F

from training.rewards.utils import extract_content


class TokenEntropyReward:
    """
    Mean per-token Shannon entropy H_t = -sum p log p over the policy's
    next-token distribution at each *completion* position, conditioned on
    the prompt that produced the rollout.

    Implementation:
    - Renders each prompt through the tokenizer's chat template (with
      add_generation_prompt=True).
    - Concatenates prompt + completion ids per row, pads to a common length,
      and forwards through the current policy in inference mode.
    - Picks the logits at positions [p_len-1 .. p_len+c_len-2] for each row
      (logits[t] predict token t+1), so entropy reflects the uncertainty
      over the actually emitted completion tokens *given the prompt*, not
      the unconditional distribution over re-tokenized completion text.

    Approximation note: the forward pass uses the *current* (post-update)
    policy rather than the policy that produced the rollout. Subword
    boundaries from re-decoding may also differ slightly from
    generation-time tokenization. Both effects are small in practice but
    the reward should be read as a proxy.

    fork_mask_top_frac: if >0, average only over tokens in the top fraction
    by entropy (Wang 2025, high-entropy tokens). Focuses reward on actual
    decision points. Expressed as a fraction in [0, 1] — e.g. 0.25 for
    "top 25%". Values >1 are rejected to prevent the prior pct/frac
    ambiguity that produced silent torch.quantile crashes.
    """

    def __init__(
        self,
        model,
        tokenizer,
        reward_scale: float = 0.1,
        fork_mask_top_frac: float = 0.0,
        max_seq_length: int | None = None,
        chunk_size: int = 4,
    ) -> None:
        if not 0.0 <= fork_mask_top_frac <= 1.0:
            raise ValueError(
                f"fork_mask_top_frac must be in [0, 1] (got {fork_mask_top_frac}). "
                "Pass a fraction like 0.25, not a percent like 25."
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 (got {chunk_size})")
        self.model = model
        self.tokenizer = tokenizer
        self.reward_scale = reward_scale
        self.fork_mask_top_frac = fork_mask_top_frac
        # Cap concatenated prompt+completion length to keep the forward pass
        # within the model's context window. Defaults to the tokenizer's max
        # if available; falls back to 4096.
        self.max_seq_length = max_seq_length or getattr(tokenizer, "model_max_length", 4096) or 4096
        # Rows per forward pass. Materialising (B, T, V) fp32 entropy
        # tensors blows up VRAM (Qwen-7B V≈152k → ~2.4 GB per intermediate
        # at B=8, T=500). Builder picks the default: 1 when vLLM is
        # co-resident (tight headroom), 4 otherwise. Direct instantiation
        # defaults to 4 — drop to 1 if you hit OOM.
        self.chunk_size = chunk_size

    def _prompt_to_text(self, prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        return self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if not completions:
            return []

        prompt_texts = [self._prompt_to_text(p) for p in prompts]

        prompt_ids = [
            self.tokenizer(pt, add_special_tokens=False).input_ids
            for pt in prompt_texts
        ]
        # Prefer TRL-provided completion_ids (exact ids the policy produced)
        # over re-tokenising the decoded text. Re-tokenisation drifts for
        # byte-fallback tokens and silently biases entropy on those rows.
        provided_ids = kwargs.get("completion_ids")
        if provided_ids is not None and len(provided_ids) == len(completions):
            comp_ids = [list(c) for c in provided_ids]
        else:
            comp_texts = [extract_content(c) for c in completions]
            comp_ids = [
                self.tokenizer(ct, add_special_tokens=False).input_ids
                for ct in comp_texts
            ]

        full_ids: list[list[int]] = []
        prompt_lens: list[int] = []
        comp_lens: list[int] = []
        for p, c in zip(prompt_ids, comp_ids):
            seq = (p + c)[: self.max_seq_length]
            p_len = min(len(p), len(seq))
            c_len = max(len(seq) - p_len, 0)
            full_ids.append(seq)
            prompt_lens.append(p_len)
            comp_lens.append(c_len)

        # Bail out cleanly if no completion tokens survived truncation.
        if all(c < 1 for c in comp_lens) or all(len(s) < 2 for s in full_ids):
            return [0.0] * len(completions)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        device = self.model.device
        n = len(completions)
        scores: list[float] = [0.0] * n

        # Chunked forward + entropy. Keeps peak VRAM bounded by chunk_size
        # rather than full batch. Per-chunk we slice [start:end] *before*
        # promoting to fp32, so the giant (B, T, V) fp32 intermediates the
        # old vectorised path produced never exist.
        for chunk_start in range(0, n, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n)
            chunk_ids = full_ids[chunk_start:chunk_end]
            chunk_max = max(len(s) for s in chunk_ids)

            input_ids = torch.tensor(
                [s + [pad_id] * (chunk_max - len(s)) for s in chunk_ids],
                dtype=torch.long, device=device,
            )
            attention_mask = torch.tensor(
                [[1] * len(s) + [0] * (chunk_max - len(s)) for s in chunk_ids],
                dtype=torch.long, device=device,
            )

            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            for j, abs_i in enumerate(range(chunk_start, chunk_end)):
                p_len = prompt_lens[abs_i]
                c_len = comp_lens[abs_i]
                if p_len < 1 or c_len < 1:
                    continue
                # logits[t] predict token t+1; completion tokens occupy
                # positions [p_len .. p_len+c_len-1], so we want entropy at
                # [p_len-1 .. p_len+c_len-2].
                start = p_len - 1
                end = p_len + c_len - 1
                # Slice first → only (c_len, V) reaches fp32, not (T, V).
                sl = logits[j, start:end].float()
                if sl.numel() == 0:
                    continue
                log_probs = F.log_softmax(sl, dim=-1)
                h_i = -(log_probs.exp() * log_probs).sum(dim=-1)
                del sl, log_probs
                if self.fork_mask_top_frac > 0.0:
                    threshold = torch.quantile(h_i, 1.0 - self.fork_mask_top_frac)
                    h_i = h_i[h_i >= threshold]
                if h_i.numel() > 0:
                    scores[abs_i] = self.reward_scale * h_i.mean().item()

            del logits, input_ids, attention_mask

        return scores
