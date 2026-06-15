# OpenEnv agentic domains: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the GRPO pipeline off unsloth onto vanilla TRL, then add an agentic multi-turn training mode backed by OpenEnv with reasoning_gym as the first domain, keeping the efficiency-reward research line intact.

**Architecture:** One training stack (vanilla TRL >= 1.x + PEFT + bitsandbytes). `train.py` dispatches on `training.mode`: the existing dataset path (single-turn GRPO over a HuggingFace `Dataset`) or a new agentic path (OpenEnv env driven through TRL's `rollout_func`, colocate vLLM). The reward composer, metrics, report, and plots are shared; the environment reward becomes one more component in the advantage-weighted composer.

**Tech stack:** vanilla TRL, transformers 5.x, PEFT, bitsandbytes (nf4), vLLM (cu130 wheel, colocate), OpenEnv (`openenv-core` + reasoning_gym client), reasoning-gym 0.1.x, Qwen3-1.7B. Validated on an NVIDIA L4 (CUDA 13.0, Python 3.12) over `ssh gpu-l4`.

---

## Pre-verified facts (dependency spike already done)

A dry-run resolution on the L4 confirmed the stack resolves with no conflict, so M0's contingency is unlikely to fire:

- `--torch-backend=auto` picks `torch==2.10.0+cu130`; resolves `trl==1.6.0`, `transformers==5.12.0`, `triton==3.6.0`, and the pinned `vllm 0.19.1+cu130` wheel together.
- `openenv-core` installs on Python 3.12 (the ">=3.13" claim was wrong for the current release).
- `reasoning-gym==0.1.25` installs on Python 3.12.

`trl==1.6.0` carries `trl.experimental.openenv`. The exact symbol names there (`generate_rollout_completions` and its signature) are still verified in-task before code is written against them (Task B6).

## Deployment and test loop (validated)

The repo already lives at `gpu-l4:/workspace/master-thesis-code` with `.venv-test` (CPU torch) holding the 86-test suite, green in ~3s.

Redeploy after local edits:

```bash
rsync -az --exclude '.venv' --exclude '.venv-test' --exclude 'runs/' \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pyc' \
  --exclude '.DS_Store' --exclude '.claude' \
  /Users/mheidebrecht/Documents/Projects/Uni/Master_Thesis/master-thesis-code/ \
  gpu-l4:/workspace/master-thesis-code/
```

Run unit tests on the box (fast TDD loop):

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code && . .venv-test/bin/activate && cd pipeline && python -m pytest tests/ -q'
```

Build the GPU env (Task A2) then smoke e2e:

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code && ./setup.sh'
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && . ../.venv/bin/activate && python -m training.train --config configs/e0-baseline-math-qwen-7b.yaml --smoke --eval'
```

The L4 is compute-slow: unit tests and `--smoke` finish quickly; a full 500-step e0 is an overnight-class background job. Always `--smoke` first, then full.

## TDD discipline (where it applies)

Pure-logic units are written test-first in `.venv-test` (registry, config_schema, mode dispatch, env reward, env domain with a FAKE OpenEnv client, cosine-length id counting, metrics). Each uses the five-step cycle: failing test -> see it fail -> minimal impl -> see it pass -> commit. GPU and external-API paths (model load, eval load, rollout_func, vLLM colocate, the reasoning_gym client) are not unit-tested; they are verified by smoke runs and explicit import/inspection checks, with uncertain external signatures pinned in-task before code is written.

After every task: redeploy (rsync) and run the full unit suite to confirm nothing regressed.

---

# Phase A: migration to vanilla TRL (M0-M1)

## Task A1: Rewrite `setup.sh` for the vanilla stack

**Files:**
- Modify: `setup.sh`

- [ ] **Step 1: Replace the install line.** Drop `unsloth`; install the vanilla stack. Full file:

```bash
#! /usr/bin/env bash

set -e

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

if [ -d .venv ]; then
    rm -rf .venv
fi

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install \
  "trl>=0.26" peft bitsandbytes accelerate \
  "https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl" \
  datasets scipy matplotlib ipywidgets \
  --torch-backend=auto
```

- [ ] **Step 2: Commit**

```bash
git add setup.sh
git commit -m "build: migrate setup.sh to vanilla TRL stack (drop unsloth)"
```

Note: openenv-core and the reasoning_gym client are added in Task B5, kept out of the migration env to isolate the swap. If `--torch-backend=auto` ever fails to resolve (it did not in the spike), the contingency is to drop `--torch-backend=auto` and let the vLLM wheel choose torch, or pick a TRL-0.26-compatible vLLM.

## Task A2: Build the env on the box (dependency spike)

**Files:** none (runs `setup.sh`)

- [ ] **Step 1: Deploy and build.** rsync (command above), then:

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code && ./setup.sh 2>&1 | tail -15'
```

- [ ] **Step 2: Verify the stack imports.**

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code && . .venv/bin/activate && python -c "import torch, trl, peft, bitsandbytes, transformers, vllm; print(torch.__version__, trl.__version__, transformers.__version__); print(torch.cuda.is_available())"'
```

Expected: prints versions (torch 2.10.0+cu130, trl 1.6.0, transformers 5.12.0) and `True`.

## Task A3: Migrate `registry.py` (TDD)

**Files:**
- Modify: `pipeline/training/registry.py`
- Test: `pipeline/tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_registry.py
import pytest
from training.registry import MODEL_REGISTRY, get_model_config


def test_qwen3_1_7b_registered():
    cfg = get_model_config("qwen3-1.7b")
    assert cfg["model_name"] == "Qwen/Qwen3-1.7B"
    assert cfg["max_seq_length"] == 2048


def test_no_unsloth_prefixed_model_names():
    for slug, cfg in MODEL_REGISTRY.items():
        assert not cfg["model_name"].startswith("unsloth/"), slug


def test_unknown_slug_raises():
    with pytest.raises(KeyError):
        get_model_config("does-not-exist")
```

- [ ] **Step 2: Run, see it fail**

Run: `ssh gpu-l4 'cd /workspace/master-thesis-code && . .venv-test/bin/activate && cd pipeline && python -m pytest tests/test_registry.py -q'`
Expected: FAIL (`qwen3-1.7b` not in registry; `unsloth/Qwen3-4B-Base` still prefixed).

- [ ] **Step 3: Edit `registry.py`.** Replace the stale comment and the two unsloth-flavored entries; add `qwen3-1.7b`:

```python
# slug -> kwargs for the model loader (AutoModelForCausalLM + LoRA)
MODEL_REGISTRY: dict[str, dict] = {
    "qwen3-1.7b": {
        "model_name": "Qwen/Qwen3-1.7B",
        "load_in_4bit": True,
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
}
```

(Keep `LORA_TARGET_MODULES` and `get_model_config` unchanged.)

- [ ] **Step 4: Run, see it pass + full suite green**

Run: `ssh gpu-l4 'cd /workspace/master-thesis-code && . .venv-test/bin/activate && cd pipeline && python -m pytest tests/ -q'`
Expected: PASS, 89 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/registry.py pipeline/tests/test_registry.py
git commit -m "refactor(registry): plain HF model names, add qwen3-1.7b"
```

## Task A4: Rewrite `GRPORunner` on vanilla TRL + PEFT

**Files:**
- Modify: `pipeline/training/grpo_runner.py`

Not unit-tested (GPU). Verified by the A6 smoke run. Preserve the `train()` and `save_lora()` signatures so `train.py` is untouched.

- [ ] **Step 1: Replace the file body.** Full new `grpo_runner.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from training.registry import LORA_TARGET_MODULES, get_model_config


class GRPORunner:
    """Vanilla TRL + PEFT GRPO. Loads the model (optionally 4-bit nf4),
    applies LoRA, and runs GRPOTrainer. The agentic rollout_func branch is
    added in Task B6."""

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
            per_device_train_batch_size=int(t.get("batch_size", 1)),
            gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 1)),
            num_generations=int(t.get("n_rollouts", 8)),
            max_prompt_length=max_prompt_len,
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
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/training/grpo_runner.py
git commit -m "refactor(runner): rewrite GRPORunner on vanilla TRL + PEFT + bitsandbytes"
```

Verification happens in Task A6 (no standalone smoke yet, eval not migrated).

## Task A5: Migrate `eval/runner.py` off unsloth

**Files:**
- Modify: `pipeline/eval/runner.py` (the model-load block, lines ~41-60)

- [ ] **Step 1: Replace the unsloth load block.** Swap:

```python
from unsloth import FastLanguageModel
from training.registry import get_model_config

model_cfg = get_model_config(config["model"]["slug"])
max_seq = config["model"].get("max_seq_length", model_cfg["max_seq_length"])

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_cfg["model_name"],
    max_seq_length=max_seq,
    load_in_4bit=config["model"].get("load_in_4bit", model_cfg["load_in_4bit"]),
)
if baseline:
    print(f"Baseline mode: assessing base model {model_cfg['model_name']} (no LoRA)")
else:
    print(f"Loading checkpoint: {checkpoint_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir)
domain.build_chat_template(tokenizer)

FastLanguageModel.for_inference(model)
```

with:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from training.registry import get_model_config

model_cfg = get_model_config(config["model"]["slug"])
load_4bit = config["model"].get("load_in_4bit", model_cfg["load_in_4bit"])
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
) if load_4bit else None

tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
model = AutoModelForCausalLM.from_pretrained(
    model_cfg["model_name"], quantization_config=quant_config,
    torch_dtype=dtype, device_map="auto",
)
if baseline:
    print(f"Baseline mode: assessing base model {model_cfg['model_name']} (no LoRA)")
else:
    print(f"Loading checkpoint: {checkpoint_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir)
domain.build_chat_template(tokenizer)

model.eval()
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/eval/runner.py
git commit -m "refactor(eval): load model via transformers + PeftModel, drop unsloth"
```

## Task A6: M0 smoke - 3-step vanilla GRPO + eval end to end

**Files:** none

- [ ] **Step 1: Deploy and smoke.** rsync, then:

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && . ../.venv/bin/activate && python -m training.train --config configs/e0-baseline-math-qwen-7b.yaml --smoke --eval 2>&1 | tail -40'
```

Expected: model loads, 3 training steps log a `reward` and `loss`, `checkpoint-final/` saves, eval runs 10 samples/split and writes `runs/e0-baseline-math-qwen-7b/eval_report.json`.

- [ ] **Step 2: Confirm artifacts.**

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && ls runs/e0-baseline-math-qwen-7b/ && python -c "import json; d=json.load(open(\"runs/e0-baseline-math-qwen-7b/eval_report.json\")); print(d.get(\"status\"), list(d.get(\"results\",{}).keys()))"'
```

Expected: lists `checkpoint-final` and `eval_report.json`; status not "error".

If the colocate vLLM path errors here, rerun without `--vllm` (default is no vLLM) and note the colocate knob for a follow-up; the smoke gate only needs the HF-generate path.

## Task A7: M1 - full e0 establishes the baseline + sanity gate

**Files:** none

- [ ] **Step 1: Baseline pass (background).** The base-model assessment over the probes:

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && . ../.venv/bin/activate && nohup python -m eval.runner --config configs/e0-baseline-math-qwen-7b.yaml --baseline > runs/_baseline.log 2>&1 &'
```

- [ ] **Step 2: Full e0 train + eval (background, overnight-class on L4).**

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && . ../.venv/bin/activate && nohup python -m training.train --config configs/e0-baseline-math-qwen-7b.yaml --eval > runs/_e0.log 2>&1 &'
```

- [ ] **Step 3: Sanity gate (after both finish).** Read both reports:

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && python -c "
import json
t=json.load(open(\"runs/e0-baseline-math-qwen-7b/eval_report.json\"))
b=json.load(open(\"runs/_baselines/qwen-7b/eval_report.json\"))
ti=t[\"results\"][\"id_split\"]; bi=b[\"results\"][\"id_split\"]
print(\"trained id acc:\", ti[\"accuracy\"], \"base id acc:\", bi[\"accuracy\"])
print(\"trained mean tokens:\", ti[\"mean_tokens\"])
"'
```

Gate passes when trained id_split accuracy is clearly above base, mean tokens are in a sane range (not collapsed to the format floor), and `training_curves.png` shows reward rising / KL bounded. This `eval_report.json` is now the project baseline.

- [ ] **Step 4: Commit a marker** (record the baseline run id in the spec or a NOTES file if desired). No code change.

### PHASE A GATE

Do not start Phase B until A7's sanity gate passes. The migrated single-turn pipeline must train and evaluate sanely on vanilla TRL.

---

# Phase B: OpenEnv agentic domain (M2-M5)

## Task B1: Config schema - agentic keys + relaxed dataset requirement (TDD)

**Files:**
- Modify: `pipeline/training/config_schema.py`
- Test: `pipeline/tests/test_agentic_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_agentic_schema.py
import pytest
from training.config_schema import validate_config


def _agentic():
    return {
        "experiment_id": "e5-agentic-rg",
        "model": {"slug": "qwen3-1.7b"},
        "training": {"mode": "agentic", "env": "reasoning_gym"},
        "rewards": {"env_reward": {"enabled": True, "weight": 1.0}},
    }


def test_agentic_valid_without_dataset():
    validate_config(_agentic())


def test_agentic_requires_env():
    cfg = _agentic()
    del cfg["training"]["env"]
    with pytest.raises(ValueError, match="env"):
        validate_config(cfg)


def test_dataset_mode_still_requires_dataset():
    cfg = {"experiment_id": "e0", "model": {"slug": "qwen-7b"}, "training": {"mode": "dataset"}}
    with pytest.raises(ValueError, match="dataset"):
        validate_config(cfg)


def test_env_reward_known_key():
    validate_config(_agentic())  # env_reward must not be rejected as unknown
```

- [ ] **Step 2: Run, see it fail**

Run: `ssh gpu-l4 '... python -m pytest tests/test_agentic_schema.py -q'`
Expected: FAIL (`training.dataset` required unconditionally; `env_reward` unknown; agentic keys unknown).

- [ ] **Step 3: Edit `config_schema.py`.**
  - Add to `_KNOWN_REWARD_KEYS`: `"env_reward"`. Add `"env_reward": _COMMON_REWARD_SUBKEYS` to `_KNOWN_REWARD_SUBKEYS`.
  - Remove `training.dataset` from the static `_REQUIRED_KEYS`. In `validate_config`, after computing `mode = (config.get("training") or {}).get("mode", "dataset")`, enforce:

```python
    mode = (config.get("training") or {}).get("mode", "dataset")
    if mode not in ("dataset", "agentic"):
        errors.append(f"training.mode={mode!r} must be 'dataset' or 'agentic'")
    if mode == "dataset" and _get_nested(config, "training.dataset") is None:
        errors.append("Missing required field: training.dataset (str) - HuggingFace dataset id")
    if mode == "agentic" and _get_nested(config, "training.env") is None:
        errors.append("Missing required field: training.env (str) - OpenEnv environment id")
```

  (Keep `experiment_id` and `model.slug` in `_REQUIRED_KEYS`. The unknown-top-level-keys and unknown-rewards-keys checks stay; `training.mode`/`training.env`/`training.env_config` live under `training`, which is not sub-key-validated, so they pass.)
  - Add `eval.agentic` handling only if you sub-validate eval; currently eval is not whitelisted at sub-key level, so no change needed there.

- [ ] **Step 4: Run, see it pass + full suite**

Run: `ssh gpu-l4 '... python -m pytest tests/ -q'`
Expected: PASS (existing dataset configs still validate; agentic validates).

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/config_schema.py pipeline/tests/test_agentic_schema.py
git commit -m "feat(config): agentic mode keys, env reward, conditional dataset/env requirement"
```

## Task B2: Mode dispatch helper (TDD)

**Files:**
- Create: `pipeline/training/mode.py`
- Test: `pipeline/tests/test_mode_dispatch.py`

- [ ] **Step 1: Failing test**

```python
# pipeline/tests/test_mode_dispatch.py
import pytest
from training.mode import select_mode


def test_default_dataset():
    assert select_mode({"training": {}}) == "dataset"


def test_agentic():
    assert select_mode({"training": {"mode": "agentic", "env": "reasoning_gym"}}) == "agentic"


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        select_mode({"training": {"mode": "bogus"}})
```

- [ ] **Step 2: Run, see it fail** (`No module named training.mode`).

- [ ] **Step 3: Implement**

```python
# pipeline/training/mode.py
def select_mode(config: dict) -> str:
    """Return 'dataset' (default) or 'agentic' from training.mode."""
    mode = (config.get("training") or {}).get("mode", "dataset")
    if mode not in ("dataset", "agentic"):
        raise ValueError(f"training.mode must be 'dataset' or 'agentic', got {mode!r}")
    return mode
```

- [ ] **Step 4: Run, see it pass.**

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/mode.py pipeline/tests/test_mode_dispatch.py
git commit -m "feat(train): pure mode-dispatch helper"
```

## Task B3: `EnvReward` component (TDD)

**Files:**
- Create: `pipeline/training/rewards/env_reward.py`
- Modify: `pipeline/training/rewards/__init__.py`
- Test: `pipeline/tests/test_env_reward.py`

- [ ] **Step 1: Failing test**

```python
# pipeline/tests/test_env_reward.py
import pytest
from training.rewards.env_reward import EnvReward


def test_passthrough():
    assert EnvReward()(["p", "p"], ["c1", "c2"], env_reward=[1.0, 0.0]) == [1.0, 0.0]


def test_missing_raises():
    with pytest.raises(ValueError):
        EnvReward()(["p"], ["c"])


def test_registry_has_env_reward():
    from training.rewards import REWARD_REGISTRY
    assert "env_reward" in REWARD_REGISTRY
    enabled, weight, builder = REWARD_REGISTRY["env_reward"]
    assert enabled is False and weight == 1.0
```

- [ ] **Step 2: Run, see it fail.**

- [ ] **Step 3: Implement `env_reward.py`**

```python
# pipeline/training/rewards/env_reward.py
class EnvReward:
    """Task-success reward computed by the OpenEnv environment. The agentic
    rollout_func attaches per-completion reward as kwargs['env_reward']; this
    surfaces it as a reward component so the composer treats it like any other."""

    def __call__(self, prompts, completions, env_reward=None, **kwargs):
        if env_reward is None:
            raise ValueError("EnvReward requires kwargs['env_reward'] from the agentic rollout_func")
        if len(env_reward) != len(completions):
            raise ValueError(f"env_reward length {len(env_reward)} != completions {len(completions)}")
        return [float(x) for x in env_reward]
```

  Add to `__init__.py`: import, builder, registry entry:

```python
from training.rewards.env_reward import EnvReward

def _build_env_reward(domain, runner, training_cfg, cfg):
    return EnvReward()

# in REWARD_REGISTRY:
    "env_reward":    (False, 1.0, _build_env_reward),
```

- [ ] **Step 4: Run, see it pass + full suite.**

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/rewards/env_reward.py pipeline/training/rewards/__init__.py pipeline/tests/test_env_reward.py
git commit -m "feat(rewards): EnvReward component, registered (default off)"
```

## Task B4: Extract shared chat-template helper (TDD-guarded refactor)

**Files:**
- Modify: `pipeline/domains/base.py`

The chat template is reused by `EnvDomain`. Extract it to a module-level function so both `Domain` and `EnvDomain` call one implementation. This touches `Domain`, so the full suite must stay green.

- [ ] **Step 1: Add the helper in `base.py`** (module level):

```python
def build_reasoning_chat_template(tokenizer, system_prompt: str, reasoning_start: str) -> None:
    sp = system_prompt.replace("\\", "\\\\").replace("'", "\\'")
    rs = reasoning_start.replace("\\", "\\\\").replace("'", "\\'")
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        f"{{{{ '{sp}' + eos_token }}}}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        f"{{% if add_generation_prompt %}}{{{{ '{rs}' }}}}{{% endif %}}"
    )
    tokenizer.chat_template = chat_template
```

- [ ] **Step 2: Make `Domain.build_chat_template` delegate**

```python
    def build_chat_template(self, tokenizer) -> None:
        build_reasoning_chat_template(tokenizer, self.system_prompt, self.reasoning_start)
```

- [ ] **Step 3: Run the full suite, confirm green** (no behavior change).

Run: `ssh gpu-l4 '... python -m pytest tests/ -q'`
Expected: PASS (unchanged output for the existing chat-template behavior).

- [ ] **Step 4: Commit**

```bash
git add pipeline/domains/base.py
git commit -m "refactor(domains): extract build_reasoning_chat_template helper"
```

## Task B5: `EnvDomain` base + reasoning_gym domain with a fake client (TDD)

**Files:**
- Create: `pipeline/domains/env_base.py`
- Create: `pipeline/domains/reasoning_gym/__init__.py`
- Create: `pipeline/domains/reasoning_gym/domain.py`
- Test: `pipeline/tests/test_env_domain.py`

The OpenEnv client import is lazy (inside `make_client`), so the domain logic is testable with a fake `StepResult`.

- [ ] **Step 1: Failing test**

```python
# pipeline/tests/test_env_domain.py
from domains.reasoning_gym.domain import ReasoningGymDomain


class _FakeStep:
    def __init__(self, reward, done=True):
        self.reward = reward
        self.done = done


class _StubTok:
    pass


def test_episode_reward():
    assert ReasoningGymDomain().episode_reward(_FakeStep(1.0)) == 1.0


def test_is_correct():
    d = ReasoningGymDomain()
    assert d.is_correct(_FakeStep(1.0)) is True
    assert d.is_correct(_FakeStep(0.0)) is False


def test_chat_template_sets_attr():
    tok = _StubTok()
    ReasoningGymDomain().build_chat_template(tok)
    assert "start_working_out" in tok.chat_template
```

- [ ] **Step 2: Run, see it fail.**

- [ ] **Step 3: Implement `env_base.py`**

```python
# pipeline/domains/env_base.py
from domains.base import build_reasoning_chat_template


class EnvDomain:
    """Base for OpenEnv-backed agentic domains. The dataset abstractions of
    `Domain` do not apply; an env domain provides a client factory and reads
    the environment-computed reward off the OpenEnv StepResult."""

    system_prompt: str = ""
    reasoning_start: str = "<start_working_out>"

    def make_client(self):
        raise NotImplementedError

    def episode_reward(self, step_result) -> float:
        return float(step_result.reward)

    def is_correct(self, step_result) -> bool:
        return float(step_result.reward) > 0.0

    def difficulty(self, task) -> float | None:
        return None

    def build_chat_template(self, tokenizer) -> None:
        build_reasoning_chat_template(tokenizer, self.system_prompt, self.reasoning_start)
```

  Implement `reasoning_gym/domain.py` (client import lazy):

```python
# pipeline/domains/reasoning_gym/domain.py
from domains.env_base import EnvDomain

SYSTEM_PROMPT = (
    "You are given a reasoning problem.\n"
    "Think about it and provide your working out.\n"
    "Place it between <start_working_out> and <end_working_out>.\n"
    "Then provide your solution between <SOLUTION></SOLUTION>"
)


class ReasoningGymDomain(EnvDomain):
    system_prompt = SYSTEM_PROMPT

    def make_client(self, env_config: dict | None = None):
        # Lazy import so unit tests need no OpenEnv install / server.
        # Exact client class/args are pinned in Task B6 against the installed package.
        from envs.reasoning_gym_env import ReasoningGymEnv  # verified in B6
        env_config = env_config or {}
        return ReasoningGymEnv.from_docker_image(
            env_config.get("image", "reasoning-gym-env:latest")
        )
```

  Add `reasoning_gym/__init__.py`:

```python
from domains.reasoning_gym.domain import ReasoningGymDomain
```

- [ ] **Step 4: Run, see it pass + full suite.**

- [ ] **Step 5: Commit**

```bash
git add pipeline/domains/env_base.py pipeline/domains/reasoning_gym/ pipeline/tests/test_env_domain.py
git commit -m "feat(domains): EnvDomain base + reasoning_gym domain (lazy client)"
```

## Task B6: Install OpenEnv + pin the rollout API, wire the agentic rollout_func

**Files:**
- Modify: `setup.sh` (append openenv install)
- Modify: `pipeline/training/grpo_runner.py` (agentic branch)
- Modify: `pipeline/domains/reasoning_gym/domain.py` (correct client import, once verified)

External-API task. Pin the real symbols before writing against them.

- [ ] **Step 1: Append OpenEnv to `setup.sh`** after the main install:

```bash
uv pip install openenv-core reasoning-gym
uv pip install "openenv-reasoning-gym-env @ git+https://huggingface.co/spaces/openenv/reasoning_gym_env"
```

(If the Space path differs, find it in Step 2.)

- [ ] **Step 2: Rebuild env and INSPECT the real API** (do not write code first):

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code && ./setup.sh 2>&1 | tail -8 && . .venv/bin/activate && \
  python -c "import trl.experimental.openenv as o; print([x for x in dir(o) if not x.startswith(\"_\")])" && \
  python -c "import envs; print([m for m in dir(envs) if \"reason\" in m.lower()])"'
```

Record the exact rollout helper name/signature and the reasoning_gym client class. Update the import in `reasoning_gym/domain.py` to match.

- [ ] **Step 3: Add the agentic branch to `GRPORunner`.** Using the pinned helper (the spike expects `generate_rollout_completions(trainer, prompts)`):

```python
    def train_agentic(self, env_domain, reward_fn, output_dir: str, dataset, callbacks=None) -> None:
        from trl.experimental.openenv import generate_rollout_completions  # pinned in B6 step 2
        client = env_domain.make_client(self.config["training"].get("env_config"))
        tok = self.tokenizer

        def rollout_func(prompts, trainer):
            outputs = generate_rollout_completions(trainer, prompts)
            env_rewards = []
            for out in outputs:
                text = tok.decode(out["completion_ids"], skip_special_tokens=True)
                step = client.step(_action_for(env_domain, text))
                env_rewards.append(env_domain.episode_reward(step))
            return {
                "prompt_ids": [o["prompt_ids"] for o in outputs],
                "completion_ids": [o["completion_ids"] for o in outputs],
                "logprobs": [o["logprobs"] for o in outputs],
                "env_reward": env_rewards,
            }

        args = self._grpo_config(output_dir)
        args.use_vllm = True
        args.vllm_mode = "colocate"
        trainer = GRPOTrainer(
            model=self.model, processing_class=tok, reward_funcs=[reward_fn],
            args=args, train_dataset=dataset, callbacks=callbacks or [],
            rollout_func=rollout_func,
        )
        trainer.train()
```

  `_action_for` and the exact `rollout_func` signature (whether it receives `trainer`) are finalized against the inspected API in Step 2. The env-reset/task-prompt source (the `dataset` of task seeds) is built in Task B7.

- [ ] **Step 4: Commit**

```bash
git add setup.sh pipeline/training/grpo_runner.py pipeline/domains/reasoning_gym/domain.py
git commit -m "feat(runner): agentic rollout_func against OpenEnv reasoning_gym"
```

## Task B7: `train.py` dispatch

**Files:**
- Modify: `pipeline/training/train.py` (`build_domain`, `main`)

- [ ] **Step 1: Add agentic dispatch.** In `build_domain`, branch on mode; in `main`, route to `train` vs `train_agentic` and inject the env-reward component. Concretely, replace `build_domain`:

```python
from training.mode import select_mode

def build_domain(config: dict):
    if select_mode(config) == "agentic":
        env = config["training"]["env"]
        if env == "reasoning_gym":
            from domains.reasoning_gym.domain import ReasoningGymDomain
            return ReasoningGymDomain()
        raise NotImplementedError(f"Env: {env}")
    name = config["training"].get("domain", "math")
    if name == "math":
        from domains.math.loader import MathDomain
        return MathDomain()
    raise NotImplementedError(f"Domain: {name}")
```

  In `main`, after building components, branch the training call:

```python
    if select_mode(config) == "agentic":
        # Build a small dataset of task prompts/seeds for env reset.
        task_dataset = _build_agentic_task_dataset(config)   # see note
        runner.train_agentic(domain, reward_fn, output_dir=run_dir,
                             dataset=task_dataset, callbacks=callbacks)
    else:
        runner.train(dataset, reward_fn, output_dir=run_dir, callbacks=callbacks)
```

  Note: `_build_agentic_task_dataset` produces the per-task prompts that seed `env.reset`. For reasoning_gym single-step this is a `Dataset` of N rows with the `prompt` chat messages (system + the task statement pulled from `reasoning-gym`). Finalize its exact shape against the env's reset contract inspected in B6.

- [ ] **Step 2: Verify config build path (CPU).** Add a quick test that `build_domain` returns `ReasoningGymDomain` for an agentic config (mock the heavy imports if needed) - or verify via the smoke run in B11.

- [ ] **Step 3: Commit**

```bash
git add pipeline/training/train.py
git commit -m "feat(train): dispatch dataset vs agentic training paths"
```

## Task B8: Efficiency rewards count model tokens (TDD)

**Files:**
- Modify: `pipeline/training/rewards/cosine_length.py`
- Test: `pipeline/tests/test_cosine_length_ids.py`

- [ ] **Step 1: Failing test**

```python
# pipeline/tests/test_cosine_length_ids.py
from training.rewards.cosine_length import CosineLengthReward


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return text.split()  # 1 "token" per word


class _Dom:
    def is_correct(self, text, truth):
        return True


def test_prefers_completion_ids_over_text():
    r = CosineLengthReward(_Tok(), _Dom(), max_len=10)
    # completion_ids says 2 tokens; text says 5 - the reward must use ids (2).
    out_ids = r(["p"], ["a b c d e"], answer=["x"], completion_ids=[[1, 2]])
    out_txt = r(["p"], ["a b c d e"], answer=["x"])
    assert out_ids != out_txt  # id path (2 tokens) differs from text path (5)
```

- [ ] **Step 2: Run, see it fail.**

- [ ] **Step 3: Edit `cosine_length.py`** `__call__` to prefer provided ids:

```python
    def __call__(self, prompts, completions, answer=None, **kwargs) -> list[float]:
        truths = _require_answers(answer, len(completions), "CosineLengthReward")
        provided_ids = kwargs.get("completion_ids")
        scores = []
        for i, (completion, truth) in enumerate(zip(completions, truths)):
            text = extract_content(completion)
            if provided_ids is not None and i < len(provided_ids):
                n_tokens = len(provided_ids[i])
            else:
                n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            correct = self.domain.is_correct(text, truth)
            scores.append(self._reward(n_tokens, correct))
        return scores
```

- [ ] **Step 4: Run, see it pass + full suite.**

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/rewards/cosine_length.py pipeline/tests/test_cosine_length_ids.py
git commit -m "feat(rewards): CosineLength counts model completion_ids when provided"
```

(`TokenEntropyReward` already prefers `kwargs['completion_ids']`; no change.)

## Task B9: Agentic eval - `_run_episodes` + steps metric

**Files:**
- Modify: `pipeline/eval/metrics.py` (optional `n_steps` on `SampleResult`/`EvalMetrics`)
- Modify: `pipeline/eval/ood_probes.py` (add `_run_episodes`)
- Test: `pipeline/tests/test_agentic_eval.py`

- [ ] **Step 1: Failing test** for the metric extension (the testable unit):

```python
# pipeline/tests/test_agentic_eval.py
from eval.metrics import SampleResult, compute_metrics


def test_mean_steps_reported():
    rs = [SampleResult(correct=True, n_tokens=10, n_steps=1),
          SampleResult(correct=False, n_tokens=20, n_steps=3)]
    m = compute_metrics(rs)
    assert abs(m.mean_steps - 2.0) < 1e-9
```

- [ ] **Step 2: Run, see it fail** (`SampleResult` has no `n_steps`; `EvalMetrics` no `mean_steps`).

- [ ] **Step 3: Extend `metrics.py`.** Add `n_steps: int | None = None` to `SampleResult`, `mean_steps: float | None = None` to `EvalMetrics`, and compute the mean over present `n_steps` in `compute_metrics`.

- [ ] **Step 4: Run, see it pass.**

- [ ] **Step 5: Add `_run_episodes` to `ood_probes.py`** (GPU-verified in B11, no unit test):

```python
def _run_episodes(model, tokenizer, env_domain, n_episodes, max_new_tokens, gen_kwargs=None):
    client = env_domain.make_client()
    results = []
    for _ in range(n_episodes):
        obs = client.reset()
        prompt = _obs_to_prompt(env_domain, obs)          # finalize vs env contract (B6)
        n_tokens, text = _generate_batch(model, tokenizer, [prompt], max_new_tokens, gen_kwargs or {"do_sample": False})
        step = client.step(_action_for(env_domain, text[0]))
        results.append(SampleResult(correct=env_domain.is_correct(step),
                                    n_tokens=n_tokens[0], n_steps=1))
    return compute_metrics(results)
```

  Wire it into `run_ood_probes` under an agentic branch (when `select_mode(config) == "agentic"`).

- [ ] **Step 6: Commit**

```bash
git add pipeline/eval/metrics.py pipeline/eval/ood_probes.py pipeline/tests/test_agentic_eval.py
git commit -m "feat(eval): agentic episode probe + mean_steps metric"
```

## Task B10: `e5` reference config (TDD-validated)

**Files:**
- Create: `pipeline/configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml`

- [ ] **Step 1: Write the config**

```yaml
experiment_id: e5-agentic-reasoning-gym-qwen3-1_7b
description: "E5: agentic GRPO on reasoning_gym via OpenEnv, env reward + cosine length, Qwen3-1.7B"
seed: 42

model:
  slug: qwen3-1.7b
  lora_r: 16
  load_in_4bit: true
  max_seq_length: 2048
  use_vllm: true
  gpu_memory_utilization: 0.3

training:
  mode: agentic
  env: reasoning_gym
  env_config:
    dataset: chain_sum        # a reasoning-gym task; finalize against installed lib
    size: 500
  n_rollouts: 8
  max_steps: 300
  learning_rate: 5e-6
  kl_beta: 0.001

rewards:
  compose_method: advantage_weighted
  env_reward:
    enabled: true
    weight: 1.0
  token_length:
    enabled: true
    weight: 1.0
    max_len: 256

eval:
  temperature: 0.0
  do_sample: false
  agentic:
    n_episodes: 100
  ood_probes:
    far: MMLU
    capability_floor: simple
```

- [ ] **Step 2: Validate (CPU)**

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && . ../.venv-test/bin/activate && python -c "
import yaml; from training.config_schema import validate_config
validate_config(yaml.safe_load(open(\"configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml\")))
print(\"valid\")"'
```

Expected: `valid`.

- [ ] **Step 3: Commit**

```bash
git add pipeline/configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml
git commit -m "feat(configs): e5 agentic reasoning_gym reference config"
```

## Task B11: M2-M4 smoke - agentic train + eval on the box

**Files:** none

- [ ] **Step 1: M2 - agentic smoke train** (after B6 pins the API):

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && . ../.venv/bin/activate && python -m training.train --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml --smoke 2>&1 | tail -40'
```

Expected: env client connects, rollouts run, `env_reward` appears in the composer metrics, 3 steps log reward, checkpoint saves.

- [ ] **Step 2: M3 - confirm efficiency component is live.** Grep the training log for the cosine-length component metric (`reward/CosineLengthReward/...`). Expected: present and non-constant.

- [ ] **Step 3: M4 - agentic eval smoke**

```bash
ssh gpu-l4 'cd /workspace/master-thesis-code/pipeline && . ../.venv/bin/activate && python -m eval.runner --config configs/e5-agentic-reasoning-gym-qwen3-1_7b.yaml --smoke 2>&1 | tail -20'
```

Expected: `eval_report.json` with an agentic id-split success rate, `mean_tokens`, and `mean_steps`.

- [ ] **Step 4: Full agentic run (background).** Drop `--smoke`, run under `nohup`. Verify a sane success rate and that the cosine-length reward shapes token counts.

## Task B12: M5 - documentation

**Files:**
- Modify: `CLAUDE.md`, `pipeline/README.md`, `pipeline/configs/_template.yaml`
- (setup.sh already updated in A1/B6)

- [ ] **Step 1: `_template.yaml`** - add the `training.mode`/`env`/`env_config`, `rewards.env_reward`, and `eval.agentic` keys with comments.
- [ ] **Step 2: `pipeline/README.md`** - new "Agentic mode" section (EnvDomain, reasoning_gym, rollout_func, env reward), add `env_reward` to the reward table, add `qwen3-1.7b` to the model registry table, document the agentic eval episode probe and `mean_steps`. Update the architecture diagram with the mode dispatch.
- [ ] **Step 3: `CLAUDE.md`** - environment setup (vanilla TRL + OpenEnv, drop unsloth), Domains section (no longer math-only; EnvDomain), the agentic training mode, the env-reward component, the agentic eval path. Reconcile the notebook sections (they describe the removed unsloth stack).
- [ ] **Step 4: Apply `humanizer` + `writing-clearly-and-concisely`** to the prose. Commit:

```bash
git add CLAUDE.md pipeline/README.md pipeline/configs/_template.yaml
git commit -m "docs: agentic mode, vanilla TRL stack, env reward, reasoning_gym"
```

---

## Self-review

**Spec coverage:** A1-A2 (M0 deps), A3-A5 + A7 (migration + baseline), A6 (M0 smoke); B1-B7 (M2 schema/dispatch/env/reward/rollout), B8 (M3 efficiency), B9 (M4 eval), B10 (e5 config), B11 (smoke gates), B12 (M5 docs). Regression gate -> Phase A gate (sanity, establish-fresh). All spec sections map to a task.

**Known in-task unknowns (honestly flagged, not placeholders):** the exact `trl.experimental.openenv` rollout helper signature, the reasoning_gym OpenEnv client class/`reset`/`step`/action contract, and the reasoning-gym task name (`env_config.dataset`). Each is pinned by an inspection step (B6 Step 2) before code depends on it. `_action_for`, `_obs_to_prompt`, and `_build_agentic_task_dataset` are finalized against that inspected contract within B6/B7/B9.

**Type/name consistency:** `select_mode`, `EnvReward`, `EnvDomain`, `ReasoningGymDomain`, `build_reasoning_chat_template`, `train_agentic`, `_run_episodes`, `mean_steps` used consistently across tasks.

**vLLM colocate:** introduced as opt-in (`model.use_vllm`); the migration smoke (A6) can fall back to HF generate; agentic (B6/B11) requires colocate and is smoke-verified on the L4.
