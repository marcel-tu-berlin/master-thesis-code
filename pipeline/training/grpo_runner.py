import os
import json
import sys

# Reduce CUDA allocator fragmentation so vLLM colocate + training coexist on a
# 24 GB GPU. Must be set before torch initializes the caching allocator.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
        # vLLM colocate is the default generation backend for training. It is
        # required for the agentic rollout path and is the only tractable option
        # for GRPO on a single GPU. Set model.use_vllm: false to fall back to HF.
        use_vllm = config["model"].get("use_vllm", True)

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

        # Bound activation memory: forward/backward completions in small
        # micro-batches and accumulate gradients over a full prompt-group
        # (batch_size * n_rollouts) per optimizer step. GRPO advantages are
        # unaffected (computed per group at reward time over the full generation
        # batch), so max_steps still counts prompts, not micro-steps.
        n_rollouts = int(t.get("n_rollouts", 8))
        total_completions = int(t.get("batch_size", 1)) * n_rollouts
        micro = int(t.get("micro_batch_size", 2))
        if micro < 1 or total_completions % micro != 0:
            micro = 1
        grad_accum = total_completions // micro

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
            per_device_train_batch_size=micro,
            gradient_accumulation_steps=grad_accum,
            num_generations=n_rollouts,
            max_completion_length=max_completion_len,
            max_steps=int(t.get("max_steps", 500)),
            save_steps=int(t.get("save_steps", 100)),
            output_dir=output_dir,
            report_to="none",
            beta=float(t.get("kl_beta", 0.001)),
            seed=int(self.config.get("seed", 42)),
        )
        # Multi-turn domains cap the tool-calling loop at max_turns so a rollout
        # cannot run unbounded turns. Single-step domains leave it unset (the
        # model calls its one tool and stops). max_tool_calling_iterations is a
        # TRL 1.6 GRPOConfig field (confirmed at the L4 smoke).
        max_turns = int((t.get("env_config") or {}).get("max_turns", 0))
        if max_turns > 0:
            kwargs["max_tool_calling_iterations"] = max_turns
        if self._use_vllm:
            kwargs["use_vllm"] = True
            kwargs["vllm_mode"] = "colocate"
            kwargs["vllm_gpu_memory_utilization"] = float(
                self.config["model"].get("gpu_memory_utilization", 0.3)
            )
            # Cap vLLM's context to the training seq length. Qwen3's native 40k
            # context would demand a ~4 GiB KV cache and OOM the colocated engine.
            kwargs["vllm_max_model_length"] = self._max_seq
        return GRPOConfig(**kwargs)

    def train(self, dataset, reward_fn, output_dir: str, callbacks=None,
              *, server=None, make_factory=None) -> None:
        # Agentic path: the runner owns the env-server subprocess lifecycle.
        # `server` is an unstarted EnvServerProcess; once it is up, build the
        # TRL environment_factory against its base_url. Dataset path: both stay
        # None and the trainer runs without environments.
        environment_factory = None
        if server is not None:
            if make_factory is None:
                raise ValueError("train(server=...) requires make_factory(base_url)")
            # environment_factory is an experimental TRL feature; silence its warn.
            os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
            server.start()
            server.wait_until_ready()
            # The env client (adapter._connect) imports `reasoning_gym_env` from
            # the OpenEnv repo's envs/ dir, which is not on PyPI - put it on the
            # training process's path (the server subprocess got it via PYTHONPATH).
            if server.repo_envs_path not in sys.path:
                sys.path.insert(0, server.repo_envs_path)
            environment_factory = make_factory(server.base_url)
        try:
            kwargs = dict(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=[reward_fn],
                args=self._grpo_config(output_dir),
                train_dataset=dataset,
                callbacks=callbacks or [],
            )
            if environment_factory is not None:
                kwargs["environment_factory"] = environment_factory
            trainer = GRPOTrainer(**kwargs)
            trainer.train()
            self._save_train_log(trainer, output_dir)
        finally:
            if server is not None:
                server.stop()

    def save_lora(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA saved to {path}")

    @staticmethod
    def _save_train_log(trainer, output_dir: str) -> None:
        """Persist TRL's in-memory log_history to train_log.json.

        save_strategy is off (we save only the final LoRA), so TRL never writes
        trainer_state.json and the per-step curves - reward/kl/loss/completion
        length plus the per-component reward metrics the callback merged in -
        would vanish on exit. Dump them so eval.plots can draw training curves.
        """
        log = getattr(getattr(trainer, "state", None), "log_history", None)
        if not log:
            return
        path = os.path.join(output_dir, "train_log.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved to {path}")
