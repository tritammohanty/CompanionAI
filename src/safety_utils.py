"""
Safety utilities for detecting unsafe content in text inputs and outputs.
"""

from typing import Optional, Tuple
import re

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -----------------------------------------------------
# Safety fallback message
# -----------------------------------------------------

SAFE_FALLBACK = (
    "I'm really sorry you're feeling this way. I can't help with requests that may be unsafe "
    "or dangerous. Please reach out to a trusted adult or local emergency services if you are in danger."
)

# ===============================================================
# Regex patterns for unsafe content detection
# ===============================================================

SELF_HARM_PATTERNS = [
    r"\bi want to die\b",
    r"\bi want to kill myself\b",
    r"\bi'm going to kill myself\b",
    r"\bi would rather die\b",
    r"\bkill myself\b",
    r"\bcommit suicide\b",
    r"\bsuicid(e|al)\b",
    r"\bend my life\b",
    r"\bhurt myself\b",
    r"\bself[- ]?harm\b",
]

VIOLENCE_PATTERNS = [
    r"\bi will kill\b",
    r"\bi can kill\b",
    r"\bi want to kill\b",
    r"\bkill (him|her|them|someone)\b",
    r"\bmurder\b",
    r"\bstab\b",
    r"\bshoot\b",
    r"\bbeat (someone|him|her)\b",
    r"\bhurt someone\b",
]

SEXUAL_PATTERNS = [
    r"\bporn\b",
    r"\bnudity\b",
    r"\bnude\b",
    r"\bsexual (content|material|activity)\b",
    r"\bexplicit (content|material)\b",
]

DRUGS_ALCOHOL_PATTERNS = [
    r"\buse drugs\b",
    r"\buse (cocaine|heroin|meth|fentanyl)\b",
    r"\bbuy drugs\b",
    r"\balcohol\b",
    r"\bbuy (weed|cannabis)\b",
    r"\bsell drugs\b",
]

MEDICAL_CLINICAL_PATTERNS = [
    r"\bdiagnos(e|is)\b",
    r"\bshould i (take|use) (medication|pills)\b",
    r"\bwhat (medicine|pill) should i\b",
    r"\bprescribe\b",
]

ILLEGAL_PATTERNS = [
    r"\bhow to (steal|shoplift|rob)\b",
    r"\bbreak into\b",
    r"\bmake (bomb|explosive)\b",
    r"\bhow to (cover up|hide) a crime\b",
]

HATE_ABUSE_PATTERNS = [
    r"\bi hate you\b",
    r"\bi hate him\b",
    r"\bi hate her\b",
    r"\bi hate them\b",
    r"\bi hate everyone\b",
    r"\bhate speech\b",
    r"\binferior race\b",
    r"\b(nigger|chink|slur)\b",
    r"\bracist\b",
    r"\bkill (all|every)\b.*\bpeople\b",
]


# ===============================================================
# Load unsafe keywords from external file
# ===============================================================
output = []
with open("data/safety/unsafe_keywords.txt", "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        if word:
            output.append(rf'    r"\b{word}\b",')

UNSAFE_KEYWORDS = [line.strip() for line in output if line.strip()]

# ===============================================================
# Compile all patterns
# ===============================================================
ALL_PATTERNS = (
    SELF_HARM_PATTERNS
    + VIOLENCE_PATTERNS
    + SEXUAL_PATTERNS
    + DRUGS_ALCOHOL_PATTERNS
    + MEDICAL_CLINICAL_PATTERNS
    + ILLEGAL_PATTERNS
    + HATE_ABUSE_PATTERNS
    + UNSAFE_KEYWORDS
)

COMPILED_REGEXES = [re.compile(p, re.I) for p in ALL_PATTERNS]


# ===============================================================
# Internal pattern matching function
# ===============================================================
def _match_pattern(text: str) -> Optional[str]:
    if not text:
        return None

    for rx in COMPILED_REGEXES:
        if rx.search(text):
            return f"regex:{rx.pattern}"


# ===============================================================
# Public safety check functions
# ===============================================================


def is_unsafe_text(text: str, debug: bool = False) -> Tuple[bool, str, Optional[str]]:
    matched = _match_pattern(text or "")
    if matched:
        return True, SAFE_FALLBACK, (matched if debug else None)
    return False, "", None


def is_unsafe_output(text: str, debug: bool = False) -> Tuple[bool, str, Optional[str]]:
    matched = _match_pattern(text or "")
    if matched:
        return True, SAFE_FALLBACK, (matched if debug else None)
    return False, "", None


# ===============================================================
# Debug run
# ===============================================================

if __name__ == "__main__":
    tests = [
        "hello friend",
        "I will kil someone",
        "I want to sucide",
        "I want to kill myself",
        "how to buy cocane",
        "I hate everyone",
        "please help me die",
        "I will buy something",
    ]

    for t in tests:
        u, fb, pat = is_unsafe_text(t, debug=True)
        print("----")
        print("text:", t)
        print("unsafe:", u)
        print("pattern:", pat)
        print("fallback:", fb if u else "")
