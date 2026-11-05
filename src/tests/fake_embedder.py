"""Proveedor de embeddings simulado y determinista para los tests (no es un módulo de tests)."""

from __future__ import annotations

import hashlib
import re

DIM = 64


class FakeEmbedder:
    """Bolsa de palabras con hash: mismas palabras, mismo vector; sin red ni modelo."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts, timeout=None):
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIM
        body = re.sub(r"^search_(?:document|query): ", "", text)
        for word in re.findall(r"[a-z0-9_]+", body.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        return vector
