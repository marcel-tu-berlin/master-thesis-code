# slug -> kwargs for the model loader (AutoModelForCausalLM + LoRA)
MODEL_REGISTRY: dict[str, dict] = {
    "qwen3-1.7b": {
        "model_name": "Qwen/Qwen3-1.7B",
        "load_in_4bit": False,   # bf16: fits 24 GB easily, plays cleanly with vLLM colocate
        "max_seq_length": 2048,
        "max_lora_rank": 32,
    },
    "qwen3-4b": {
        "model_name": "Qwen/Qwen3-4B-Base",
        "load_in_4bit": False,
        "max_seq_length": 2048,
        "max_lora_rank": 32,
    },
    "qwen-1.5b": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "max_lora_rank": 32,
    },
    "qwen-7b": {
        "model_name": "Qwen/Qwen2-7B",
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "max_lora_rank": 64,
    },
    # Second lineage, for the cross-model check that the reward effects are not
    # a Qwen artefact. Llama is not a preference, it is the only option: TRL's
    # `add_response_schema` matches chat templates by exact string equality
    # against a fixed list (glm4moe, gptoss, llama3_1/3_2, qwen*, nemotron_3)
    # and `GRPOTrainer` refuses to start on anything else. Of that list the only
    # non-Qwen entries that fit a 24 GB card are the Llama 3.x pair - gpt-oss is
    # 21.5B and GLM-4.5-Air 110B. SmolLM3 and Granite render our tools correctly
    # and TRL still rejects them, so "has a tool-calling template" is not the bar.
    #
    # The `unsloth/` mirror rather than `meta-llama/` because the official repo
    # is gated behind license approval plus an HF token readable on the box, and
    # that box is shared. The usual objection to a mirror is that a re-upload
    # could carry a modified template, which for this pipeline would be the one
    # confound that matters; this mirror's `chat_template` hashes byte-identical
    # to TRL's `llama3_2_chat_template` (sha256 5816fce10444e03c). Its only
    # deviations from stock are an `unsloth_fixed` marker in config.json and a
    # `pad_token` of `<|finetune_right_pad_id|>`, a reserved Llama token that
    # official ships unset.
    #
    # Two differences from Qwen3 matter to a length study: no thinking mode (so
    # the tool-call JSON is the whole generation and there is no reasoning trace
    # to compress), and a 128k vocab against Qwen3's 152k (a proportionally
    # smaller logits tensor in the policy update).
    "llama3.2-1b": {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "load_in_4bit": False,
        "max_seq_length": 2048,
        "max_lora_rank": 32,
    },
}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def get_model_config(slug: str) -> dict:
    if slug not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model slug: {slug!r}. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[slug]
