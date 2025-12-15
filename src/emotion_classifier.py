"""
Uses a pretrained RoBERTa emotion classifier

predict(text) -> emotion label
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


# -----------------------------------------------------
# EmotionClassifier class
# -----------------------------------------------------
class EmotionClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.labels = self.model.config.id2label  # dynamic label map
        self.model.to("cpu")

    def predict(self, text: str) -> str:
        if not text:
            return "neutral"

        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**tokens).logits
            probs = F.softmax(logits, dim=1)
            class_id = torch.argmax(probs, dim=1).item()

        return self.labels[class_id]


# -----------------------------------------------------
# Debug run
# -----------------------------------------------------
def main():
    clf = EmotionClassifier()

    test_cases = {
        # ---- Normal emotional cases ----
        "I am very happy today!": "joy",
        "I feel really sad and lonely.": "sadness",
        "I am scared to go to school tomorrow.": "fear",
        "I am angry at my friend.": "anger",
        "That was disgusting!": "disgust",
        "Wow, I didn't expect that at all!": "surprise",
        # ---- Child-style / informal text ----
        "idk i just feel bad": "sadness",
        "im scared": "fear",
        "yayyy!!!": "joy",
        "ugh whatever": "anger",
        # ---- Neutral / ambiguous ----
        "Okay.": "neutral",
        "I don't know.": "neutral",
        "Hmm.": "neutral",
        # ---- Safety-relevant edge cases ----
        "I want to die": "sadness",
        "I hate everyone": "anger",
        "Nobody cares about me": "sadness",
        # ---- Noisy / malformed input ----
        "": "neutral",
        "     ": "neutral",
        ".....": "neutral",
        "asdfghjkl": "neutral",
    }

    print("\n=== Emotion Classifier Test ===\n")

    for text, expected in test_cases.items():
        pred = clf.predict(text)
        print(f"TEXT: {repr(text)}")
        print(f"PREDICTED: {pred} | EXPECTED (approx): {expected}")
        print("-" * 60)


if __name__ == "__main__":
    main()
