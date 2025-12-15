"""
Memory Utilities Module for Chatbot
Contains:
  - EntityMemory
  - TimelineMemory
  - SemanticMemory
  - MemoryManager

Backend receives a compact dictionary:
{
    "retrieved": [...],
    "entities": [...],
    "timeline": "...",
}
"""

import os
import json
import re
import torch
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util


# -----------------------------------------------------
# Entity Memory
# -----------------------------------------------------


class EntityMemory:
    def __init__(self):
        self.entities = []

        # lightweight keyword extraction (no spaCy needed)
        self.stopwords = {
            "i",
            "am",
            "is",
            "are",
            "the",
            "a",
            "an",
            "and",
            "to",
            "of",
            "for",
            "but",
            "you",
            "in",
            "on",
            "it",
            "was",
            "today",
            "have",
            "had",
            "that",
            "this",
            "with",
            "got",
        }

    def extract(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        filtered = [t for t in tokens if len(t) > 2 and t not in self.stopwords]
        return filtered[:6]

    def update(self, text: str):
        new_entities = self.extract(text)
        for ent in new_entities:
            if ent not in self.entities:
                self.entities.append(ent)
        if len(self.entities) > 40:  # avoid memory overflow
            self.entities = self.entities[-40:]

    def get(self, n=8):
        return self.entities[-n:]


# -----------------------------------------------------
# Timeline Memory
# -----------------------------------------------------


class TimelineMemory:
    def __init__(self):
        self.timeline = []

    def update(self, user_text: str, bot_text: Optional[str]):
        if bot_text:
            self.timeline.append(f"User: {user_text}")
            self.timeline.append(f"Assistant: {bot_text}")
        else:
            self.timeline.append(f"User: {user_text}")

        # keep only last N messages
        if len(self.timeline) > 30:
            self.timeline = self.timeline[-30:]

    def summary(self, n=8):
        """Return last few turns stitched together."""
        return " ".join(self.timeline[-n:])


# -----------------------------------------------------
# Semantic Memory
# -----------------------------------------------------


class SemanticMemory:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.entries = []  # list of {text, embedding(list)}

    def update(self, text: str):
        emb = self.model.encode(text).tolist()
        self.entries.append({"text": text, "embedding": emb})

        if len(self.entries) > 200:  # memory bound
            self.entries = self.entries[-200:]

    def search(self, query: str, k: int = 3) -> List[str]:
        if not self.entries:
            return []

        query_emb = self.model.encode(query)
        corpus_embs = torch.tensor([e["embedding"] for e in self.entries])

        scores = util.cos_sim(torch.tensor(query_emb), corpus_embs)[0]
        top_idx = torch.topk(scores, k=min(k, len(scores))).indices.tolist()

        return [self.entries[i]["text"] for i in top_idx]


# -----------------------------------------------------
# Memory Manager
# -----------------------------------------------------


class MemoryManager:
    def __init__(self, storage_path="data/memory/memory_store.json"):
        self.storage_path = storage_path

        self.entity_memory = EntityMemory()
        self.timeline_memory = TimelineMemory()
        self.semantic_memory = SemanticMemory()

        self.load()

    def load(self):
        if not os.path.exists(self.storage_path):
            self.save()
            return

        try:
            data = json.load(open(self.storage_path, "r"))
            self.entity_memory.entities = data.get("entities", [])
            self.timeline_memory.timeline = data.get("timeline", [])
            self.semantic_memory.entries = data.get("semantic", [])
        except Exception:
            pass

    def save(self):
        data = {
            "entities": self.entity_memory.entities,
            "timeline": self.timeline_memory.timeline,
            "semantic": self.semantic_memory.entries,
        }
        json.dump(data, open(self.storage_path, "w"), indent=2)

    def update_all(self, user_text: str, bot_text: Optional[str]):
        self.entity_memory.update(user_text)
        self.timeline_memory.update(user_text, bot_text)
        self.semantic_memory.update(user_text)
        if bot_text:
            self.semantic_memory.update(bot_text)
        self.save()

    def retrieve_context(self, user_text: str, k: int = 3) -> Dict:
        retrieved = self.semantic_memory.search(user_text, k=k)
        ents = self.entity_memory.get()
        timeline = self.timeline_memory.summary()

        return {
            "retrieved": retrieved,
            "entities": ents,
            "timeline": timeline,
        }

    def reset(self):
        """Clear all memories."""
        self.entity_memory = EntityMemory()
        self.timeline_memory = TimelineMemory()
        self.semantic_memory = SemanticMemory()
        self.save()


# -----------------------------------------------------
# Debug run
# -----------------------------------------------------
if __name__ == "__main__":
    mm = MemoryManager("data/memory/debug_memory.json")

    mm.update_all("I am very happy today because I got free ice-cream!", None)
    mm.update_all("My friend cried today", "Oh no, what happened?")

    print(mm.retrieve_context("ice-cream"))
