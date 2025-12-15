"""
Preprocess raw empatheticdialogues CSVs (train/valid/test) into
clean user->assistant pairs with safety filtering

Saves cleaned files to:
    data/processed/empatheticdialogues/{train,validation,test}.csv

Saves removal logs to:
    data/processed/empatheticdialogues/{split}_filtered_log.csv
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safety_utils import is_unsafe_text  # unified safety system
from src.prompt_utils import SYSTEM_TEMPLATE

RAW_DIR = "data/raw/empatheticdialogues"
PROCESSED_DIR = "data/processed/empatheticdialogues"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# Text cleaning function
# ============================================================
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text)
    t = t.replace("_comma_", ",")
    t = re.sub(r"<.*?>", "", t)  # remove XML/HTML tags
    t = re.sub(r"\s+", " ", t)  # collapse whitespace
    return t.strip()


# ============================================================
# Safety check wrapper
# ============================================================
def check_safe(text: str):
    """
    Returns:
        safe: bool
        pattern: matched regex pattern or None
    """
    unsafe, _, pat = is_unsafe_text(text, debug=True)
    return (not unsafe), pat


# ============================================================
# Process a single split
# ============================================================
def process_split(split_name: str, raw_path: str) -> pd.DataFrame:
    print(f"\n--- Processing {split_name} from {raw_path} ---")

    df = pd.read_csv(raw_path, on_bad_lines="skip")
    print(f"Loaded {len(df)} rows.")

    df = df.sort_values(["conv_id", "utterance_idx"]).reset_index(drop=True)

    pairs = []
    removal_log = []

    for conv_id, conv_df in df.groupby("conv_id"):
        conv_df = conv_df.sort_values("utterance_idx").reset_index(drop=True)
        if conv_df.empty:
            continue

        emotion = (
            str(conv_df["context"].iloc[0]).strip()
            if "context" in conv_df.columns
            else ""
        )

        speakers = list(conv_df["speaker_idx"].unique())
        if len(speakers) < 1:
            continue

        user_speaker = conv_df["speaker_idx"].iloc[0]
        assistant_speaker = speakers[1] if len(speakers) > 1 else None

        utterances = conv_df[["speaker_idx", "utterance"]].values.tolist()
        found_pair = False

        for i in range(len(utterances) - 1):
            spk, ut = utterances[i]
            next_spk, next_ut = utterances[i + 1]

            if (
                assistant_speaker is not None
                and spk == user_speaker
                and next_spk == assistant_speaker
            ):
                user = clean_text(ut)
                assistant = clean_text(next_ut)

                if len(user) > 3 and len(assistant) > 3:
                    pairs.append(
                        {
                            "system": SYSTEM_TEMPLATE,
                            "user": user,
                            "assistant": assistant,
                            "emotion": emotion,
                        }
                    )
                    found_pair = True

        if not found_pair:
            first_ut = clean_text(conv_df["utterance"].iloc[0])
            if len(first_ut) > 3:
                pairs.append(
                    {
                        "system": SYSTEM_TEMPLATE,
                        "user": first_ut,
                        "assistant": "",
                        "emotion": emotion,
                    }
                )

    df_pairs = pd.DataFrame(pairs)
    if df_pairs.empty:
        print("No pairs constructed for this split.")
        return df_pairs

    # ============================================================
    # Safety filtering with logging
    # ============================================================
    def row_safe(row):
        u = row["user"]
        a = row["assistant"]

        safe_u, pat_u = check_safe(u)
        safe_a, pat_a = (True, None)

        if a:
            safe_a, pat_a = check_safe(a)

        if safe_u and safe_a:
            return True

        removal_log.append(
            {
                "user": u,
                "assistant": a,
                "emotion": row.get("emotion", None),
                "unsafe_user": not safe_u,
                "unsafe_assistant": not safe_a,
                "pattern_user": pat_u,
                "pattern_assistant": pat_a,
                "split": split_name,
            }
        )
        return False

    mask = df_pairs.apply(row_safe, axis=1)
    safe_df = (
        df_pairs[mask]
        .drop_duplicates(subset=["user", "assistant"])
        .reset_index(drop=True)
    )

    print(f"Constructed {len(df_pairs)} entries → {len(safe_df)} safe after filtering.")

    if len(safe_df):
        print("Sample safe pairs:")
        print(safe_df[["user", "assistant"]].head(3).to_string(index=False))

    if len(removal_log):
        removal_df = pd.DataFrame(removal_log)
        removal_path = os.path.join(PROCESSED_DIR, f"{split_name}_filtered_log.csv")
        removal_df.to_csv(removal_path, index=False)
        print(f"Saved removal log → {removal_path} ({len(removal_df)} rows removed)")
    else:
        print("No unsafe rows removed for this split.")

    return safe_df


# ============================================================
# Debug run
# ============================================================
def main():
    filenames = {
        "train": os.path.join(RAW_DIR, "train.csv"),
        "validation": os.path.join(RAW_DIR, "valid.csv"),
        "test": os.path.join(RAW_DIR, "test.csv"),
    }

    for split, path in filenames.items():
        if not os.path.exists(path):
            print(f"Missing {path}, skipping.")
            continue

        out = process_split(split, path)
        out_path = os.path.join(PROCESSED_DIR, f"{split}.csv")
        out.to_csv(out_path, index=False)

        print(f"Saved {split} → {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
