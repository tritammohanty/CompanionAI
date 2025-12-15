"""
Contains prompt templates and utilities for building Mistral instruction prompts
"""

# -----------------------------------------------------------------------------------
# System Prompt Template
# -----------------------------------------------------------------------------------

SYSTEM_TEMPLATE = """
You are a gentle, patient companion for children and teenagers.
Use simple, warm, safe language.
Do not give medical, diagnostic, romantic, violent, or adult advice.
Do not diagnose any condition.
Your role is to:
1. Listen carefully.
2. Respond with empathy and kindness.
3. Help the child name or explore feelings.
4. Encourage talking to a trusted adult if something feels scary, dangerous, or confusing.

If the child expresses sadness, fear, anger, or confusion, respond calmly and supportively.
Avoid giving solutions; focus on understanding and comfort.
Always stay gentle, safe, and non-judgmental.
Follow every rule exactly.
"""

SYSTEM_PROMPT = """
You are a gentle, patient companion for children and teenagers. Use simple, warm, safe language. 
Do not provide medical, diagnostic, romantic, violent, or adult advice, and do not guess personal facts.

Your job is to:
- listen kindly,
- reflect feelings softly,
- offer comfort,
- encourage talking to a trusted adult if something feels scary or unsafe.

Stay calm, supportive, and never use sarcasm or jokes about fear.
Always follow these rules exactly."""


# -----------------------------------------------------------------------------------
# Emotion Conditioning Layer
# -----------------------------------------------------------------------------------
def emotion_condition_block(emotion: str) -> str:
    """
    Auto-generates tone guidance for ANY emotion (27–32 labels)
    using a compact cluster-based system.
    """

    emotion = emotion.lower().strip()

    EMOTION_TO_CLUSTER = {
        "afraid": "fear",
        "apprehensive": "fear",
        "anxious": "fear",
        "terrified": "fear",
        "sad": "distress",
        "disappointed": "distress",
        "lonely": "distress",
        "devastated": "distress",
        "guilty": "distress",
        "hurt": "distress",
        "angry": "anger",
        "annoyed": "anger",
        "furious": "anger",
        "jealous": "anger",
        "happy": "joy",
        "excited": "joy",
        "grateful": "joy",
        "proud": "joy",
        "embarrassed": "social",
        "ashamed": "social",
        "awkward": "social",
        "neutral": "neutral",
        "surprised": "neutral",
        "hopeful": "neutral",
    }

    CLUSTER_TONE = {
        "distress": "Respond gently, validate their feelings, and offer warm support.",
        "fear": "Provide comfort, reassurance, and a sense of safety.",
        "anger": "Stay calm, grounded, and supportive. Avoid escalating their emotions.",
        "joy": "Respond with friendly encouragement and share their positive moment.",
        "social": "Respond softly, normalize their feelings, and help them feel safe.",
        "neutral": "Maintain a warm, balanced, neutral tone.",
    }

    cluster = EMOTION_TO_CLUSTER.get(emotion, "neutral")
    tone = CLUSTER_TONE[cluster]

    return f"The user feels {emotion}. {tone}"


# -----------------------------------------------------------------------------------
# Memory Conditioning Layer
# -----------------------------------------------------------------------------------
def memory_block(mem: dict) -> str:
    """
    Convert memory dict → formatted block for prompt.
    Keys expected:
        mem["retrieved"] : list[str]
        mem["entities"]  : list[str]
        mem["timeline"]  : str
    """

    retrieved_txt = "\n".join(f"- {r}" for r in mem.get("retrieved", []))
    ent_txt = ", ".join(mem.get("entities", [])) or "(none)"

    timeline_txt = mem.get("timeline", "").strip()
    if not timeline_txt:
        timeline_txt = "(no conversation history available)"

    return f"""
MEMORY CONTEXT:
  Retrieved Relevant Messages:
{retrieved_txt if retrieved_txt else "- (none)"}

  Important Keywords/Entities:
    {ent_txt}

  Recent Conversation Summary:
    {timeline_txt}
"""


# -----------------------------------------------------------------------------------
# Build Final Prompt
# -----------------------------------------------------------------------------------
def build_prompt(
    system_template: str, user_text: str, mem_block: dict, emotion: str
) -> str:
    """
    Builds an optimal Mistral instruction prompt.
    """

    memory_txt = ""
    if mem_block:
        memory_txt = memory_block(mem_block)
        memory_txt = f"[PAST_MEMORY]\n{memory_txt}\n[/PAST_MEMORY]\n"

    emotion_txt = ""
    if emotion:
        emo_guideline = emotion_condition_block(emotion)
        emotion_txt = (
            f"[EMOTION]\nDetected: {emotion}\nGuideline: {emo_guideline}\n[/EMOTION]\n"
        )

    final_prompt = (
        f"<s>[INST] <<SYS>>\n"
        f"{system_template}\n"
        f"<</SYS>>\n\n"
        f"{memory_txt}"
        f"{emotion_txt}"
        f"[USER]\n{user_text}\n[/USER]\n"
        f"[/INST]"
    )

    return final_prompt
