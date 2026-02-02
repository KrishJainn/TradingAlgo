"""
AQTIS Embedding Provider.

Uses sentence-transformers for local embeddings (no API key needed).
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Local embedding generation using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embeddings. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class ChromaEmbeddingFunction:
    """Wrapper to use EmbeddingProvider as a ChromaDB embedding function."""

    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._provider.embed_batch(input)
