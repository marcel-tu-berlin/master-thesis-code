"""A new continuation stage must really start from the saved trainable adapter."""

import pytest


def test_load_trained_adapter_preserves_weights_predictions_and_trainability(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    pytest.importorskip("trl")
    from transformers import GPT2Config, GPT2LMHeadModel

    from training.grpo_runner import _attach_lora

    config = GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16)
    lora = peft.LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    base = GPT2LMHeadModel(config)
    base_weights = {k: v.clone() for k, v in base.state_dict().items()}
    trained, info = _attach_lora(base, lora)
    assert info is None
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    trained.eval()
    tokens = torch.tensor([[1, 2, 3]])
    expected = trained(tokens).logits.detach()
    trained.save_pretrained(tmp_path)
    fresh = GPT2LMHeadModel(config)
    fresh.load_state_dict(base_weights)
    reloaded, info = _attach_lora(fresh, lora, str(tmp_path))
    reloaded.eval()
    assert torch.equal(expected, reloaded(tokens).logits.detach())
    assert info["weights_equal_after_load"] and info["new_optimizer_and_scheduler"]
    assert all("lora_" in n for n, p in reloaded.named_parameters() if p.requires_grad)
    wrong = peft.LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    with pytest.raises(ValueError, match=" r differs"):
        _attach_lora(GPT2LMHeadModel(config), wrong, str(tmp_path))
