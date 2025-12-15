"""
Train a Mistral model with LoRA adapters on a processed dataset.

Usage (example):
  accelerate launch scripts/train.py --config configs/default.yaml.
"""

import argparse
import os
import yaml
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from src.model_utils import load_base_model, apply_lora
from src.dataset_utils import load_processed_dataset, tokenize_fn


# -----------------------------------------------------
# Argument parsing
# -----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML config file (optional)",
    )
    return p.parse_args()


# -----------------------------------------------------
# Load YAML config
# -----------------------------------------------------
def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------
# Main training function
# -----------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    MODEL_NAME = cfg.get("model_name", "mistralai/mistral-7b-instruct-v0.3")
    PROCESSED_DIR = cfg.get("processed_dir", "data/processed/empatheticdialogues")
    OUTPUT_DIR = cfg.get("output_dir", "checkpoints/mistral_lora_3")
    ADAPTER_DIR = cfg.get("adapter_dir", "logs/mistral_lora_3")
    SEQ_LEN = cfg.get("seq_len", 512)
    BATCH = cfg.get("per_device_train_batch_size", 1)
    GRAD_ACC = cfg.get("gradient_accumulation_steps", 4)
    EPOCHS = cfg.get("num_train_epochs", 3)
    LR = cfg.get("learning_rate", 2e-4)
    TRAIN_SUBSET = cfg.get("train_subset", 2000)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (may offload to CPU if needed)...")
    base = load_base_model(MODEL_NAME)
    model = apply_lora(base)

    print("Loading processed dataset from:", PROCESSED_DIR)
    ds = load_processed_dataset(PROCESSED_DIR)
    tokenize = lambda ex: tokenize_fn(ex, tokenizer, seq_len=SEQ_LEN)
    tok_ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

    if TRAIN_SUBSET:
        train_dataset = tok_ds["train"].select(
            range(min(TRAIN_SUBSET, len(tok_ds["train"])))
        )
    else:
        train_dataset = tok_ds["train"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=True,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    t0 = time.time()
    trainer.train()
    print("Training finished in {:.2f} min".format((time.time() - t0) / 60.0))

    print("Saving LoRA adapter...")
    adapter_path = os.path.join(ADAPTER_DIR, "lora_adapter")
    model.save_pretrained(adapter_path)
    print("Saved LoRA adapter to", adapter_path)


if __name__ == "__main__":
    main()
