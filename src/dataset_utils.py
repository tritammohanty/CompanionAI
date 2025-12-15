"""
Helpers to load processed CSVs and tokenize for Trainer.
"""

import os
from datasets import load_dataset, DatasetDict
from typing import Dict
from transformers import PreTrainedTokenizerBase


# -----------------------------------------------------
# Dataset Loading
# -----------------------------------------------------
def load_processed_dataset(processed_dir: str) -> DatasetDict:
    """
    Load the processed CSVs into a HF DatasetDict.
    Expects processed_dir/{train.csv,validation.csv,test.csv}
    """
    data_files = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(processed_dir, f"{split}.csv")
        if os.path.exists(p):
            data_files[split] = p
    if not data_files:
        raise FileNotFoundError(f"No processed CSVs found in {processed_dir}")
    ds = load_dataset("csv", data_files=data_files)

    return ds


# -----------------------------------------------------
# Tokenization
# -----------------------------------------------------
def tokenize_fn(examples: Dict, tokenizer: PreTrainedTokenizerBase, seq_len: int = 512):
    """
    Convert system/user/assistant -> single sequence for causal LM training.
    Produces input_ids, attention_mask, labels (shifted same as input for causal LM).
    """
    prompts = []
    for s, u, a in zip(
        examples.get("system", []),
        examples.get("user", []),
        examples.get("assistant", []),
    ):
        if a and str(a).strip():
            full = f"System: {s}\nUser: {u}\nAssistant: {a}"
        else:
            full = f"System: {s}\nUser: {u}\nAssistant:"
        prompts.append(full)
    enc = tokenizer(prompts, truncation=True, max_length=seq_len, padding="max_length")

    enc["labels"] = enc["input_ids"].copy()
    return enc
