from __future__ import annotations

# slug → kwargs for FastLanguageModel.from_pretrained
MODEL_REGISTRY: dict[str, dict] = {
    "qwen3-4b": {
        "model_name": "unsloth/Qwen3-4B-Base",
        "load_in_4bit": False,
        "max_seq_length": 2048,
        "max_lora_rank": 32,
    },
    "qwen-1.5b": {
        "model_name": "Qwen/QwQ-1.5B",
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "max_lora_rank": 32,
    },
    "qwen-7b": {
        "model_name": "Qwen/QwQ-7B",
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "max_lora_rank": 64,
    },
    "llama-8b": {
        "model_name": "meta-llama/meta-Llama-3.1-8B-Instruct",
        "load_in_4bit": True,
        "max_seq_length": 512,
        "max_lora_rank": 32,
    },
    "llama-1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "max_lora_rank": 16,
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
