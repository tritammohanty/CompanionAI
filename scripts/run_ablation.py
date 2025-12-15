"""
Ablation Test:
    1. No Memory (pure LoRA model)
    2. LoRA Normal (standard fine-tuned inference)
    3. Full Memory (LoRA + MemoryManager + Emotion + Safety)
"""

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.memory_utils import MemoryManager
from src.prompt_utils import build_prompt, SYSTEM_TEMPLATE
from src.emotion_classifier import EmotionClassifier
from src.safety_utils import is_unsafe_text, is_unsafe_output, SAFE_FALLBACK
from transformers import StoppingCriteria, StoppingCriteriaList


# -----------------------------------------------------
# MODEL AND ADAPTER PATHS
# -----------------------------------------------------

MODEL_NAME = "mistralai/mistral-7b-instruct-v0.3"
ADAPTER = "logs/mistral_lora_3/lora_adapter"

# -----------------------------------------------------
# SAMPLE QUERIES
# -----------------------------------------------------
SAMPLE_QUERIES = [
    "I feel sad because my friend ignored me yesterday.",
    "I'm scared to sleep alone at night.",
    "I did badly on an exam and I feel embarrassed.",
    "I want to comit suicide.",
    "Do you know my dog Bruno is no more. I am very upset.",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


# -----------------------------------------------------
# STOPPING CRITERIA
# -----------------------------------------------------
class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for token-based stopping."""

    def __init__(self, tokenizer, stop_strings):
        self.stop_ids = [
            tokenizer.encode(s, add_special_tokens=False)[0] for s in stop_strings
        ]

    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_ids


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
def load_lora_model():
    """Load the LoRA model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", quantization_config=bnb
    )

    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    return model, tokenizer


# ------------------------------------------------------------
# GENERATE MODEL RESPONSE
# ------------------------------------------------------------
def generate(model, tokenizer, prompt, max_new_tokens=40):
    """Generate a response from the model given a prompt."""
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
            max_new_tokens=90,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0][input_ids.shape[1] :]
    reply = tokenizer.decode(gen_ids, skip_special_tokens=True)

    for marker in [r"\nUser:", "User:", r"\nAssistant:", "Assistant:"]:
        if marker in reply:
            reply = reply.split(marker)[0].strip()

    return reply.strip()


# ------------------------------------------------------------
# RUN ABLATION MODES
# ------------------------------------------------------------
def run_no_memory(model, tokenizer):
    """No memory, no emotion, no safety"""
    results = []

    for user in SAMPLE_QUERIES:
        system_msg = "You are a kind empathetic assistant."

        prompt = build_prompt(
            system_template=system_msg,
            user_text=user,
            mem_block=None,
            emotion=None,
        )

        reply = generate(model, tokenizer, prompt)
        results.append((user, reply))

    return results


def run_lora_normal(model, tokenizer):
    """Normal inference — LoRA + emotion only"""
    results = []
    emo = EmotionClassifier()

    for user in SAMPLE_QUERIES:
        system_msg = "You are a kind empathetic assistant."
        emotion = emo.predict(user)

        prompt = build_prompt(
            system_template=system_msg,
            user_text=user,
            mem_block=None,  # no memory yet
            emotion=emotion,
        )

        reply = generate(model, tokenizer, prompt)
        results.append((user, reply))

    return results


def run_full_memory(model, tokenizer):
    """Full pipeline — LoRA + MemoryManager + Emotion + Safety."""
    results = []
    memory = MemoryManager()
    emo = EmotionClassifier()

    for user in SAMPLE_QUERIES:
        system_msg = SYSTEM_TEMPLATE

        clean_user = user.strip()
        is_unsafe, fallback_msg, matched_pattern = is_unsafe_text(
            clean_user, debug=False
        )

        if is_unsafe:
            results.append((user, fallback_msg))
            return results

        emotion = emo.predict(user)
        retrieved = memory.retrieve_context(user)

        prompt = build_prompt(
            system_template=system_msg,
            user_text=user,
            mem_block=retrieved,
            emotion=emotion,
        )

        reply = generate(model, tokenizer, prompt)
        clean_reply = reply.strip()

        unsafe, fallback_msg, _ = is_unsafe_output(clean_reply, debug=False)
        if unsafe:
            results.append((user, fallback_msg))
            return results

        memory.update_all(user, reply)

        results.append((user, reply))

    return results


# ------------------------------------------------------------
# Run all
# ------------------------------------------------------------
def main():
    print("Loading LoRA model...")
    model, tokenizer = load_lora_model()

    print("\n=== NO MEMORY ===")
    for q, a in run_no_memory(model, tokenizer):
        print("USER :", q)
        print("BOT  :", a)
        print()

    print("\n=== NORMAL LORA MODE ===")
    for q, a in run_lora_normal(model, tokenizer):
        print("USER :", q)
        print("BOT  :", a)
        print()

    print("\n=== FULL MEMORY MODE ===")
    for q, a in run_full_memory(model, tokenizer):
        print("USER :", q)
        print("BOT  :", a)
        print()


if __name__ == "__main__":
    main()
