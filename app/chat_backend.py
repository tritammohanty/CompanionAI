"""
Main backend logic for the CompanionAI chat application.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import traceback
from typing import Tuple, Optional

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig

from src.safety_utils import is_unsafe_text, is_unsafe_output, SAFE_FALLBACK
from src.memory_utils import MemoryManager
from src.emotion_classifier import EmotionClassifier
from src.prompt_utils import build_prompt, SYSTEM_PROMPT
from transformers import StoppingCriteria, StoppingCriteriaList

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/mistral-7b-instruct-v0.3")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "logs/mistral_lora_3/lora_adapter")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------
# Global parameters
# -----------------------------------------------------
GEN_MAX_NEW_TOKENS = 90
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.6"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
REPEATION_PENALTY = float(os.getenv("REPEATION_PENALTY", "1.15"))

memory_manager = MemoryManager()
emotion_model = EmotionClassifier()

_MODEL_CACHE = {"model": None, "tokenizer": None}


# -----------------------------------------------------
# Custom Stopping Criteria
# -----------------------------------------------------
class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.stop_ids = [
            tokenizer.encode(s, add_special_tokens=False)[0] for s in stop_strings
        ]

    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_ids


# -----------------------------------------------------
# Model Loading Function
# -----------------------------------------------------
def load_model_and_tokenizer():
    """Load model+tokenizer once, using BNB 4-bit quantization with safe defaults."""
    if _MODEL_CACHE["model"] and _MODEL_CACHE["tokenizer"]:
        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=bnb,
    )
    model.eval()

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


# -----------------------------------------------------
# Generation model response Function
# -----------------------------------------------------
def _generate_from_model(model, tokenizer, prompt: str) -> str:
    """Generate a reply from the model given the prompt."""
    device = next(model.parameters()).device

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    BAD_STRINGS = [
        "User:",
        "USER:",
        "User :",
        "USER :",
        "Assistant:",
        "ASSISTANT:",
        "Assistant :",
        "ASSISTANT :",
        "USER MESSAGE:",
        "ASSISTANT MESSAGE:",
        "USER MESSAGE",
        "ASSISTANT MESSAGE",
        "User message:",
        "Assistant message:",
    ]

    BAD_STRINGS += [
        "http://",
        "https://",
        "www.",
        ".com",
        "youtube",
        "youtu.be",
        ":)",
        ";)",
        ":(",
        ":D",
        ":P",
        "<3",
    ]

    STOP_WORDS = ["</s>", "[/INST]", "[USER]", "User:", "Assistant:"]

    bad_words_ids = [tokenizer.encode(x, add_special_tokens=False) for x in BAD_STRINGS]

    stop_criteria = StoppingCriteriaList([StopOnTokens(tokenizer, STOP_WORDS)])

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bad_words_ids=bad_words_ids,
            stopping_criteria=stop_criteria,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            do_sample=True,
            repetition_penalty=REPEATION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0][input_ids.shape[1] :]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    for stop in STOP_WORDS:
        if stop in decoded:
            decoded = decoded.split(stop)[0]
            break

    reply = decoded.strip()

    return reply


# -----------------------------------------------------
# Main Backend Function
# -----------------------------------------------------
def process_message(user_text: str, debug: bool = False) -> Tuple[str, dict]:
    """Process a user message and generate a reply along with an explanation."""
    try:
        clean_user = user_text.strip()
        is_unsafe, fallback_msg, matched_pattern = is_unsafe_text(
            clean_user, debug=debug
        )

        if is_unsafe:
            memory_manager.update_all(clean_user, SAFE_FALLBACK)
            explanation = {
                "retrieved": [],
                "entities": [],
                "timeline": "",
                "emotion": emotion_model.predict(clean_user),
                "safety": "user-unsafe",
                "matched_pattern": matched_pattern if debug else None,
                "bot_behavior": "safety_prevented",
            }
            return (fallback_msg, explanation)

        memory_manager.update_all(user_text, bot_text=None)

        mem_block = memory_manager.retrieve_context(user_text, k=3)

        emotion = emotion_model.predict(user_text) if not is_unsafe else "distress"

        prompt = build_prompt(SYSTEM_PROMPT, user_text, mem_block, emotion)

        model, tokenizer = load_model_and_tokenizer()
        reply = _generate_from_model(model, tokenizer, prompt)

        clean_reply = reply.strip()
        is_unsafe_output_flag, fallback_msg, matched_pattern = is_unsafe_output(
            clean_reply, debug=debug
        )
        if is_unsafe_output_flag:
            explanation = {
                "retrieved": mem_block.get("retrieved", []),
                "entities": mem_block.get("entities", []),
                "timeline": mem_block.get("timeline", ""),
                "emotion": emotion,
                "safety": "output-unsafe" + matched_pattern,
                "bot_behavior": "safety_fallback",
            }
            memory_manager.update_all(clean_user, SAFE_FALLBACK)
            return (fallback_msg, explanation)

        memory_manager.update_all(user_text, reply)

        explanation = {
            "retrieved": mem_block.get("retrieved", []),
            "entities": mem_block.get("entities", []),
            "timeline": mem_block.get("timeline", ""),
            "emotion": emotion,
            "safety": "safe",
            "matched_pattern": None,
            "bot_behavior": "normal",
        }
        return (reply, explanation)

    except Exception as e:
        tb = traceback.format_exc()
        if debug:
            explanation = {
                "retrieved": [],
                "entities": [],
                "timeline": "",
                "emotion": "unknown",
                "safety": "error",
                "bot_behavior": "error",
                "error": str(e),
                "traceback": tb,
            }
        else:
            explanation = {
                "retrieved": [],
                "entities": [],
                "timeline": "",
                "emotion": "unknown",
                "safety": "error",
                "bot_behavior": "error",
            }
        return (SAFE_FALLBACK, explanation)
