"""
Evaluate the trained adapter on validation/test splits (PPL + heuristics).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel, AutoPeftModelForCausalLM


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
MODEL_NAME = "mistralai/mistral-7b-instruct-v0.3"
ADAPTER = (
    "logs/mistral_lora_3/lora_adapter"  # Path to the trained adapter - Change as needed
)
PROCESSED_DIR = "data/processed/empatheticdialogues"
MAX_EVAL = 200


# ------------------------------------------------------------
# Loading model and adapter
# ------------------------------------------------------------
def load_model_adapter():
    """Load the base model and the LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            ADAPTER, quantization_config=bnb, device_map="auto"
        )
        return model, tokenizer
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="offload_cache",
        )
        model = PeftModel.from_pretrained(
            base, ADAPTER, device_map="auto", offload_folder="offload_cache"
        )
        return model, tokenizer


# ------------------------------------------------------------
# Generate Model Reply
# ------------------------------------------------------------
def generate_reply(model, tokenizer, system_msg, user_msg):
    """Generate a reply from the model given system and user messages."""
    device = next(model.parameters()).device
    prompt = f"System: {system_msg}\nUser: {user_msg}\nAssistant:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=80,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)

    if "Assistant:" in full:
        full = full.split("Assistant:")[-1].strip()
    return full


# ------------------------------------------------------------
# 1. Perplexity Score
# ------------------------------------------------------------
def compute_ppl(model, tokenizer, df, max_eval=200, seq_len=512):
    """Compute perplexity of the model on the generated responses."""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for idx, row in df.head(max_eval).iterrows():
        text = f"System: {row['system']}\nUser: {row['user']}\nAssistant: {row['assistant']}"
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        ).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        loss = out.loss.item()
        tokens = enc["input_ids"].numel()
        total_loss += loss * tokens
        total_tokens += tokens
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


# ------------------------------------------------------------
# 2. Empathy Heuristics Accuracy
# ------------------------------------------------------------
def empathy_heuristics(df, max_eval=200):
    """Compute simple empathy heuristics on the generated responses."""
    flags = {"responses": 0, "avg_len": 0.0, "questions": 0, "reflective_count": 0}
    rows = df.head(max_eval)
    total_len = 0
    for _, r in rows.iterrows():
        resp = str(r.get("assistant", "")).lower()
        total_len += len(resp.split())
        if "?" in resp:
            flags["questions"] += 1
        for ph in [
            "sounds like",
            "it seems",
            "i hear",
            "that must",
            "i understand",
            "that sounds",
        ]:
            if ph in resp:
                flags["reflective_count"] += 1
    flags["responses"] = len(rows)
    flags["avg_len"] = total_len / max(1, len(rows))
    return flags


# ------------------------------------------------------------
# 3. Emotional Alignment Score
# ------------------------------------------------------------
from src.emotion_classifier import EmotionClassifier

detector = EmotionClassifier()


def emotional_alignment(df, max_eval=200):
    """Compute emotional alignment between user and bot responses."""
    scores = []
    df = df.head(max_eval)
    for _, row in df.iterrows():
        user_text = str(row["user"])
        bot_text = str(row["assistant"])
        user_em = detector.predict(user_text)
        bot_em = detector.predict(bot_text)

        if not user_em:
            continue
        if user_em == bot_em:
            scores.append(1)
        else:
            scores.append(0)

    return np.mean(scores) if scores else 0.0


# ------------------------------------------------------------
# 4. Safety Violation Rate
# ------------------------------------------------------------
from src.safety_utils import is_unsafe_text


def safety_violation_rate(df):
    """Compute safety violation rate in the generated responses."""
    unsafe = 0
    u_list = []
    match = []
    for _, r in df.iterrows():
        text = str(r["assistant"])
        unsafe_flag, _, m = is_unsafe_text(text, debug=True)
        if unsafe_flag:
            unsafe += 1
            u_list.append(unsafe_flag)
            match.append(m)
    return unsafe / len(df), match


# ------------------------------------------------------------
# 5. Response Diversity (unique n-grams)
# ------------------------------------------------------------
def ngram_diversity(df, n=3, max_eval=200):
    """Compute n-gram diversity in the generated responses."""
    df = df.head(max_eval)

    def compute_n(df, n):
        unique_ngrams = set()
        total_ngrams = 0

        for _, row in df.iterrows():
            words = str(row["assistant"]).split()
            count = max(0, len(words) - n + 1)
            total_ngrams += count

            for i in range(len(words) - n + 1):
                unique_ngrams.add(tuple(words[i : i + n]))

        if total_ngrams == 0:
            return 0.0

        return len(unique_ngrams) / total_ngrams

    return {
        "uni": compute_n(df, 1),
        "bi": compute_n(df, 2),
        "tri": compute_n(df, 3),
    }


# ------------------------------------------------------------
# 6. Coherence score
# ------------------------------------------------------------
def coherence_score(df, max_eval=200):
    """Compute a simple coherence score based on word overlap."""
    df = df.head(max_eval)
    scores = []
    for _, row in df.iterrows():
        u = row["user"].lower()
        a = row["assistant"].lower()
        overlap = len(set(u.split()) & set(a.split()))
        scores.append(min(overlap / len(u.split()), 1.0))
    return np.mean(scores)


# ------------------------------------------------------------
# 7. Memory Recall Tests
# ------------------------------------------------------------
def memory_recall_tests(model, tokenizer):
    """Run simple memory recall tests on the model."""
    device = next(model.parameters()).device
    tests = [
        ("User: Yesterday I gave my dog Bruno a bath.", "User: What is my dog's name?"),
        ("User: Last week I scored 77 on maths.", "User: What score did I get?"),
    ]
    outputs = []
    for pre, q in tests:
        prompt = f"System: You are an empathetic assistant. {pre}\n{q}\nAssistant:"
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=40)
        resp = (
            tokenizer.decode(out[0], skip_special_tokens=True)
            .split("Assistant:")[-1]
            .strip()
        )
        outputs.append((q, resp))
    return outputs


# ------------------------------------------------------------
# Run all
# ------------------------------------------------------------
def main():
    model, tokenizer = load_model_adapter()
    for split in ["validation", "test"]:
        p = os.path.join(PROCESSED_DIR, f"{split}.csv")
        if not os.path.exists(p):
            print("Missing", p)
            continue

        df = pd.read_csv(p)

        generated = []
        for _, row in df.head(MAX_EVAL).iterrows():
            reply = generate_reply(model, tokenizer, row["system"], row["user"])
            generated.append(
                {"user": row["user"], "system": row["system"], "assistant": reply}
            )

        gen_df = pd.DataFrame(generated)

        print(f"\n---- {split.upper()} ----")
        print("Empathy Heuristics:", empathy_heuristics(gen_df))
        print("Perplexity:", compute_ppl(model, tokenizer, gen_df, max_eval=MAX_EVAL))
        print("Emotional Alignment:", emotional_alignment(gen_df))
        print("Safety Violation Rate:", safety_violation_rate(gen_df))
        print("N-gram Diversity:", ngram_diversity(gen_df))
        print("Coherence Score:", coherence_score(gen_df))

    print("\n---- Memory recall tests: ----")
    for q, r in memory_recall_tests(model, tokenizer):
        print(q, "=>", r)


if __name__ == "__main__":
    main()
