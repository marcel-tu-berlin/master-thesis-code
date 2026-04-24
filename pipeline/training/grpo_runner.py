from __future__ import annotations

import os
from typing import Callable

from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from training.registry import LORA_TARGET_MODULES, get_model_config

PatchFastRL("GRPO", FastLanguageModel)


class GRPORunner:
    """
    Thin wrapper around Unsloth + TRL GRPOTrainer.
    Keeps model loading, LoRA setup, and training config separate from business logic.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        model_cfg = get_model_config(config["model"]["slug"])

        lora_rank = config["model"].get("lora_r", model_cfg["max_lora_rank"])
        load_4bit = config["model"].get("load_in_4bit", model_cfg["load_in_4bit"])
        max_seq = config["model"].get("max_seq_length", model_cfg["max_seq_length"])
        use_vllm = config["model"].get("use_vllm", False)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["model_name"],
            max_seq_length=max_seq,
            load_in_4bit=load_4bit,
            fast_inference=use_vllm,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=config["model"].get("gpu_memory_utilization", 0.6) if use_vllm else None,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_rank,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=config.get("seed", 3407),
        )

        self._lora_rank = lora_rank
        self._max_seq = max_seq
        self._use_vllm = use_vllm

    def train(self, dataset, reward_fn: Callable, output_dir: str) -> None:
        t = self.config["training"]
        max_prompt_len = t.get("max_prompt_length", self._max_seq // 2)
        max_completion_len = self._max_seq - max_prompt_len

        grpo_args = GRPOConfig(
            use_vllm=self._use_vllm,
            temperature=t.get("temperature", 1.0),
            learning_rate=t.get("learning_rate", 5e-6),
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=t.get("weight_decay", 0.1),
            warmup_ratio=t.get("warmup_ratio", 0.1),
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=1,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=t.get("batch_size", 1),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
            num_generations=t.get("n_rollouts", 8),
            max_prompt_length=max_prompt_len,
            max_completion_length=max_completion_len,
            max_steps=t.get("max_steps", 500),
            save_steps=t.get("save_steps", 100),
            output_dir=output_dir,
            report_to="none",
            kl_coef=t.get("kl_beta", 0.001),
        )

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[reward_fn],
            args=grpo_args,
            train_dataset=dataset,
        )
        trainer.train()

    def save_lora(self, path: str) -> None:
        self.model.save_lora(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA saved to {path}")
