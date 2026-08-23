import os
import re
import zlib

import numpy as np
from sentence_transformers import SentenceTransformer

_WORD_RE = re.compile(r"[a-z0-9]+")


def _cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Embedder:
    """Real embedder — use on your local machine."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def cosine_score(self, a: np.ndarray, b: np.ndarray) -> float:
        return _cosine_score(a, b)


class MockEmbedder:
    """
    Drop-in replacement when HuggingFace is unavailable. Builds a deterministic
    bag-of-words hashing-trick vector: each word hashes to a dimension, so
    texts sharing vocabulary produce genuinely more similar vectors than texts
    that don't. (Pure random noise per string can't do that — random vectors
    confined to the positive orthant have high cosine similarity with each
    other regardless of content, which is a geometry artifact, not a signal.)
    Uses zlib.crc32 rather than Python's built-in hash() so vectors are
    reproducible across separate runs, not just within one process (built-in
    hash() is randomized per-process for strings unless PYTHONHASHSEED is set).
    Replace with Embedder() on your local machine.
    """
    def __init__(self, dim=384):
        self.dim = dim

    def _text_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            idx = zlib.crc32(word.encode()) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._text_vector(t) for t in texts])

    def cosine_score(self, a: np.ndarray, b: np.ndarray) -> float:
        return _cosine_score(a, b)


def get_embedder():
    """Reads AXIOM_EMBEDDER (default: real) and returns the matching embedder."""
    which = os.environ.get("AXIOM_EMBEDDER", "real").lower()
    if which == "real":
        return Embedder()
    if which == "mock":
        return MockEmbedder()
    raise ValueError(f"Unknown AXIOM_EMBEDDER: {which!r}")

