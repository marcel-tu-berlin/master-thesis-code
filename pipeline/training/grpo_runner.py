import json
import os
import sys

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from training.config_schema import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_ROLLOUTS,
    resolve_checkpoint_steps,
    resolve_max_turns,
)
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
        max_seq = int(
            config["model"].get("max_seq_length", model_cfg["max_seq_length"])
        )
        # vLLM colocate is the default generation backend for training. It is
        # required for the agentic rollout path and is the only tractable option
        # for GRPO on a single GPU. Set model.use_vllm: false to fall back to HF.
        use_vllm = config["model"].get("use_vllm", True)
        sleep_mode = bool(config["model"].get("vllm_enable_sleep_mode", False))

        # Expandable segments reduce allocator fragmentation so vLLM colocate
        # and training coexist on 24 GB. torch parses the setting at the first
        # CUDA allocation (the from_pretrained below), so it is set here rather
        # than at import. vLLM's sleep mode is the exception: it swaps the
        # engine's memory through a CUDA memory pool, which torch cannot back
        # with expandable segments (pytorch#147851; vLLM asserts on the
        # PYTORCH_CUDA_ALLOC_CONF spelling, torch honours this one too).
        if not sleep_mode:
            os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
        self._sleep_mode = sleep_mode

    def completion_budget(self) -> int:
        """Tokens one trajectory may generate, tool results included: the
        trainer's max_completion_length. Reward components that score budget
        exhaustion (E3) read the cap from here so they cannot drift from it."""
        t = self.config["training"]
        max_prompt_len = int(t.get("max_prompt_length", self._max_seq // 2))
        return self._max_seq - max_prompt_len

    def _grpo_config(self, output_dir: str) -> GRPOConfig:
        t = self.config["training"]
        checkpoint_steps = resolve_checkpoint_steps(self.config)
        max_completion_len = self.completion_budget()

        # Bound activation memory: forward/backward completions in small
        # micro-batches and accumulate gradients over `batch_size` whole
        # prompt-groups per optimizer step.
        #
        # `batch_size` defaults to 4, not 1, and the difference is not a
        # performance knob. TRL centres advantages on the per-group mean
        # (grpo_trainer.py:2177, scale_rewards="group"), so a group whose
        # rollouts all score alike yields exactly zero advantage. A binary
        # env_reward saturates that way ~46% of the time, so at batch_size 1 -
        # one group per step - roughly half of all optimizer steps applied a
        # zero update. A shaped reward that varies within the group (cosine
        # length) does not saturate, so the defect hit env-only baselines about
        # twice as hard as the treatment arms they were compared against. A step
        # is dead only when every group in it saturates: 0.46^4 = 4.5%.
        #
        # `training.scale_rewards` picks the std those advantages are divided by
        # (TRL: group | batch | none). Under `group`, a shaped reward's weight
        # cancels in every group whose task reward is constant - the std of
        # R_task - lambda*C is lambda*std(C) there, so the advantage is the same
        # at every lambda (DIET, App. B). A lambda sweep that is meant to be a
        # dose-response therefore runs under `none` or `batch`; `group` stays the
        # default so the frozen e30-e36 configs reproduce what they trained under.
        n_rollouts = int(t.get("n_rollouts", DEFAULT_N_ROLLOUTS))
        batch_size = int(t.get("batch_size", DEFAULT_BATCH_SIZE))
        total_completions = batch_size * n_rollouts
        micro = int(t.get("micro_batch_size", 2))
        if micro < 1 or total_completions % micro != 0:
            micro = 1
        grad_accum = total_completions // micro
        max_steps = int(t.get("max_steps", 500))
        # Print the resolved geometry: cross-arm comparisons are only valid
        # between runs that agree on it, and the log is where that is checked.
        print(
            f"Batch geometry: batch_size={batch_size}  n_rollouts={n_rollouts}  "
            f"micro_batch_size={micro}  grad_accum={grad_accum}  max_steps={max_steps}"
        )

        # Recipe defaults since 2026-08-24 (LAB_NOTES "Standing rule: recipe
        # defaults"). The seed-42 browsergym campaign (e30-e36) ran
        # paged_adamw_8bit / cosine / kl_beta 0.001; its configs state those
        # explicitly, so re-running them reproduces the old recipe.
        #  - adamw_torch_fused: the LoRA has ~17M params, so fp32 Adam state is
        #    ~140 MB. The 8-bit paged optimizer saved nothing and quantised the
        #    moments of exactly the parameters being trained.
        #  - constant_with_warmup: a 150-step run under cosine spent its last
        #    ~40 steps near zero LR, so the average LR was about half of the
        #    stated one. This setup is step-starved; each step should count.
        #  - kl_beta 0.0 (TRL's own default): at 0.001 the KL term was 1e-4 of
        #    the loss while the ref-model forward it requires cost a full extra
        #    pass every step.
        kwargs = dict(
            temperature=float(t.get("temperature", 1.0)),
            learning_rate=float(t.get("learning_rate", 5e-6)),
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=float(t.get("weight_decay", 0.1)),
            warmup_ratio=float(t.get("warmup_ratio", 0.1)),
            lr_scheduler_type=str(t.get("lr_scheduler_type", "constant_with_warmup")),
            optim=str(t.get("optim", "adamw_torch_fused")),
            logging_steps=1,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            # TRL 1.6 counts per_device_train_batch_size in completions, and a
            # full prompt-group is num_generations completions. Micro-batch the
            # forward pass and accumulate, so one optimizer step still consumes
            # whole groups and max_steps counts steps of `batch_size` prompts.
            per_device_train_batch_size=micro,
            gradient_accumulation_steps=grad_accum,
            num_generations=n_rollouts,
            max_completion_length=max_completion_len,
            max_steps=max_steps,
            save_steps=(
                checkpoint_steps[0]
                if checkpoint_steps
                else int(t.get("save_steps", 100))
            ),
            output_dir=output_dir,
            report_to="none",
            beta=float(t.get("kl_beta", 0.0)),
            seed=int(self.config.get("seed", 42)),
            # Phase-2 A/B knobs at TRL's defaults (LAB_NOTES "recipe defaults").
            num_iterations=int(t.get("num_iterations", 1)),
            use_liger_kernel=bool(t.get("use_liger_kernel", False)),
            scale_rewards=str(t.get("scale_rewards", "group")),
        )
        if checkpoint_steps:
            kwargs["save_total_limit"] = None
        print(
            f"Recipe: optim={kwargs['optim']}  lr_scheduler_type={kwargs['lr_scheduler_type']}  "
            f"kl_beta={kwargs['beta']}  learning_rate={kwargs['learning_rate']}  "
            f"num_iterations={kwargs['num_iterations']}  use_liger_kernel={kwargs['use_liger_kernel']}  "
            f"scale_rewards={kwargs['scale_rewards']}  vllm_enable_sleep_mode={self._sleep_mode}"
        )
        # Cap the tool-calling loop. TRL treats an unset
        # max_tool_calling_iterations as sys.maxsize, so leaving it off for
        # single-step domains left the loop unbounded: a reasoning_gym rollout
        # could call `answer` repeatedly until it filled the completion budget,
        # which is itself a mechanism for the cap-bound rollouts the e22/e23 cap
        # probes went looking for. resolve_max_turns is shared with the eval
        # loop, so an unset env_config.max_turns means the same episode process
        # (one iteration) on both sides. max_tool_calling_iterations is a TRL
        # 1.6 GRPOConfig field (confirmed at the L4 smoke).
        kwargs["max_tool_calling_iterations"] = resolve_max_turns(t.get("env_config"))
        if self._use_vllm:
            kwargs["use_vllm"] = True
            kwargs["vllm_mode"] = "colocate"
            kwargs["vllm_gpu_memory_utilization"] = float(
                self.config["model"].get("gpu_memory_utilization", 0.3)
            )
            # Cap vLLM's context to the training seq length. Qwen3's native 40k
            # context would demand a ~4 GiB KV cache and OOM the colocated engine.
            kwargs["vllm_max_model_length"] = self._max_seq
            kwargs["vllm_enable_sleep_mode"] = self._sleep_mode
            # TRL 1.6's default correction for the vLLM-vs-trainer logprob
            # mismatch is sequence_mask: the per-EPISODE weight exp(sum of
            # per-token drift) multiplies the loss, and a weight outside
            # [0, 3.0] is zeroed. The summed drift grows with completion
            # length, so long episodes lose their gradient preferentially
            # (ISR mean 0.28 on browsergym, 0.49 on poly - see
            # docs/plans/no-arm-beats-e0-audit.md). Configs name the mode
            # explicitly; "off" disables the correction, the four TRL mode
            # names pass through, absent keeps TRL's default so pre-fix
            # frozen configs retain their recorded semantics.
            is_mode = t.get("vllm_importance_sampling_mode")
            if is_mode == "off":
                kwargs["vllm_importance_sampling_correction"] = False
            elif is_mode is not None:
                kwargs["vllm_importance_sampling_mode"] = str(is_mode)
            print(
                f"vLLM importance sampling mode: "
                f"{is_mode if is_mode is not None else 'sequence_mask (TRL default)'}"
            )
        return GRPOConfig(**kwargs)

    def train(
        self,
        dataset,
        reward_fn,
        output_dir: str,
        callbacks=None,
        *,
        server=None,
        make_factory=None,
    ) -> None:
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

        TRL writes a checkpoint every `save_steps` (100 by default, since
        GRPOConfig's save_strategy is STEPS), but those carry only what the
        trainer needs to resume. The per-step curves - reward/kl/loss/completion
        length plus the per-component reward metrics the callback merged in -
        live in `state.log_history` and would vanish on exit. Dump them so
        eval.plots can draw training curves.

        This runs after `trainer.train()` returns, so a killed run loses the
        file; its curves survive only as text in the batch log.
        """
        log = getattr(getattr(trainer, "state", None), "log_history", None)
        if not log:
            return
        path = os.path.join(output_dir, "train_log.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved to {path}")
