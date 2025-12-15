"""
Model utilities: load base model in quantized 4-bit safely, and apply LoRA.
"""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# -----------------------------------------------------
# Model Loading and LoRA Application
# -----------------------------------------------------
def load_base_model(model_name: str):
    """
    Load base model in 4-bit with safe offload. Returns a model instance.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
        offload_folder="offload_cache",
    )
    # recommended safety flags
    model.config.use_cache = False
    return model


# -----------------------------------------------------
# LoRA Application
# -----------------------------------------------------
def apply_lora(model, r=16, alpha=16, dropout=0.05, target_modules=None):
    """
    Prepare model for k-bit training and attach LoRA adapter.
    """
    model = prepare_model_for_kbit_training(model)
    if target_modules is None:
        # common projection names; adjust if base model uses different names
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    return peft_model
