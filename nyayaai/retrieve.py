"""Local TF-IDF retrieval. No embeddings API is needed, so nothing leaves the server."""

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .chunk import Chunk

SUFFIXES = ("ations", "ation", "ating", "ated", "ates", "ate", "ions", "ion", "ing", "ed", "es", "s", "e")
SYNONYMS = {
    "cancel": "terminate",
    "end": "terminate expire",
    "quit": "terminate vacate",
    "leave": "terminate vacate",
    "refund": "deposit return",
    "pay": "rent payment",
    "cost": "rent payment charges",
    "fine": "penalty",
}


def _stem(word: str) -> str:
    for s in SUFFIXES:
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[: -len(s)]
    return word


def analyze_text(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        try:
            self.vec = TfidfVectorizer(analyzer=analyze_text, sublinear_tf=True)
            self.matrix = self.vec.fit_transform([c.text for c in chunks])
        except ValueError:  # empty vocabulary
            self.vec = None

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        if self.vec is None:
            return []
        extra = " ".join(SYNONYMS.get(w, "") for w in re.findall(r"[a-z]+", query.lower()))
        scores = (self.matrix @ self.vec.transform([f"{query} {extra}"]).T).toarray().ravel()
        return [(self.chunks[i], float(scores[i])) for i in np.argsort(-scores)[:k] if scores[i] >= min_score]
