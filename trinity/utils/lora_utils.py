from typing import Optional


def create_dummy_lora(
    model_path: str,
    checkpoint_job_dir: str,
    lora_rank: int,
    lora_alpha: int,
    target_modules: str,
    exclude_modules: Optional[str] = None,
) -> str:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    if not hasattr(config, "vocab_size"):
        # for Qwen3.5, vocab_size  is not stored in the top-level config
        if not hasattr(config, "get_text_config"):
            raise ValueError(
                f"Model config loaded from {model_path!r} has neither 'vocab_size' "
                "nor 'get_text_config()', so a text config cannot be derived."
            )
        # For some models, vocab_size may only be available on the text sub-config.
        config = config.get_text_config()
        if not hasattr(config, "vocab_size"):
            raise ValueError(
                f"Text config derived from {model_path!r} does not define 'vocab_size'."
            )
    model = AutoModelForCausalLM.from_config(config)
    lora_config = {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "exclude_modules": exclude_modules,
        "bias": "none",
    }
    peft_model = get_peft_model(model, LoraConfig(**lora_config))
    peft_model.save_pretrained(f"{checkpoint_job_dir}/dummy_lora")
    del model, peft_model
    torch.cuda.empty_cache()
    return f"{checkpoint_job_dir}/dummy_lora"
