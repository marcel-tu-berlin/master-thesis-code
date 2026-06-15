import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from training.registry import LORA_TARGET_MODULES, get_model_config


class GRPORunner:
    """Vanilla TRL + PEFT GRPO. Loads the model (optionally 4-bit nf4), applies
    LoRA, and runs GRPOTrainer. The agentic rollout_func branch is added later.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        model_cfg = get_model_config(config["model"]["slug"])

        lora_rank = int(config["model"].get("lora_r", model_cfg["max_lora_rank"]))
        lora_alpha = int(config["model"].get("lora_alpha", lora_rank * 2))
        load_4bit = config["model"].get("load_in_4bit", model_cfg["load_in_4bit"])
        max_seq = int(config["model"].get("max_seq_length", model_cfg["max_seq_length"]))
        use_vllm = config["model"].get("use_vllm", False)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        quant_config = None
        if load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_name"],
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.config.use_cache = False

        if load_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.enable_input_require_grads()

        self._lora_rank = lora_rank
        self._max_seq = max_seq
        self._use_vllm = use_vllm

    def _grpo_config(self, output_dir: str) -> GRPOConfig:
        t = self.config["training"]
        max_prompt_len = t.get("max_prompt_length", self._max_seq // 2)
        max_completion_len = self._max_seq - max_prompt_len
        kwargs = dict(
            temperature=float(t.get("temperature", 1.0)),
            learning_rate=float(t.get("learning_rate", 5e-6)),
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=float(t.get("weight_decay", 0.1)),
            warmup_ratio=float(t.get("warmup_ratio", 0.1)),
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=1,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            # TRL 1.6 counts per_device_train_batch_size in completions, and a
            # full prompt-group is num_generations completions. Set it to
            # batch_size * n_rollouts so one optimizer step consumes whole
            # groups (1 prompt-group/step at batch_size=1) and max_steps tracks
            # the number of prompts trained on, not micro-steps within a group.
            per_device_train_batch_size=int(t.get("batch_size", 1)) * int(t.get("n_rollouts", 8)),
            gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 1)),
            num_generations=int(t.get("n_rollouts", 8)),
            max_completion_length=max_completion_len,
            max_steps=int(t.get("max_steps", 500)),
            save_steps=int(t.get("save_steps", 100)),
            output_dir=output_dir,
            report_to="none",
            beta=float(t.get("kl_beta", 0.001)),
            seed=int(self.config.get("seed", 42)),
        )
        if self._use_vllm:
            kwargs["use_vllm"] = True
            kwargs["vllm_mode"] = "colocate"
            kwargs["vllm_gpu_memory_utilization"] = float(
                self.config["model"].get("gpu_memory_utilization", 0.3)
            )
        return GRPOConfig(**kwargs)

    def train(self, dataset, reward_fn, output_dir: str, callbacks=None) -> None:
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[reward_fn],
            args=self._grpo_config(output_dir),
            train_dataset=dataset,
            callbacks=callbacks or [],
        )
        trainer.train()

    def save_lora(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA saved to {path}")
