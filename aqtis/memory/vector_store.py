"""
AQTIS Vector Store (ChromaDB).

Provides semantic search for research papers and trade pattern embeddings.
Falls back gracefully to a no-op store when chromadb is not installed.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check once at import time
_HAS_CHROMADB = False
try:
    import chromadb as _chromadb
    _HAS_CHROMADB = True
except ImportError:
    _chromadb = None


class _NoOpCollection:
    """Stub collection that silently discards writes when chromadb is unavailable."""

    def __init__(self, name: str):
        self.name = name
        self._data: Dict[str, dict] = {}

    def count(self) -> int:
        return len(self._data)

    def upsert(self, ids, documents=None, metadatas=None, **kw):
        for i, doc_id in enumerate(ids):
            self._data[doc_id] = {
                "document": documents[i] if documents else "",
                "metadata": metadatas[i] if metadatas else {},
            }

    def query(self, query_texts=None, n_results=5, **kw):
        # Return empty results structure
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    def delete(self, ids=None, **kw):
        if ids:
            for doc_id in ids:
                self._data.pop(doc_id, None)


class _NoOpClient:
    """Stub client that uses in-memory dicts when chromadb is unavailable."""

    def __init__(self):
        self._collections: Dict[str, _NoOpCollection] = {}

    def get_or_create_collection(self, name, **kw):
        if name not in self._collections:
            self._collections[name] = _NoOpCollection(name)
        return self._collections[name]

    def delete_collection(self, name):
        self._collections.pop(name, None)


class VectorStore:
    """ChromaDB-backed vector store for semantic search."""

    def __init__(
        self,
        persist_dir: str = "aqtis_vectors",
        embedding_fn: Callable = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = embedding_fn
        self._client = None
        self._collections: Dict[str, Any] = {}

    @property
    def client(self):
        if self._client is None:
            if _HAS_CHROMADB:
                try:
                    self._client = _chromadb.PersistentClient(path=str(self.persist_dir))
                except Exception as e:
                    logger.warning(f"ChromaDB PersistentClient failed ({e}), using in-memory")
                    try:
                        self._client = _chromadb.Client()
                    except Exception:
                        logger.warning("ChromaDB Client also failed, using no-op fallback")
                        self._client = _NoOpClient()
            else:
                logger.info("chromadb not installed, using in-memory fallback store")
                self._client = _NoOpClient()
        return self._client

    def _get_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        if name not in self._collections:
            kwargs = {"name": name}
            if self._embedding_fn is not None:
                kwargs["embedding_function"] = self._embedding_fn
            self._collections[name] = self.client.get_or_create_collection(**kwargs)
        return self._collections[name]

    # ─────────────────────────────────────────────────────────────────
    # RESEARCH OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def add_research(self, document: Dict) -> str:
        """
        Add a research paper/insight to the vector store.

        Args:
            document: Dict with keys: title, text, metadata (authors, url, key_findings, etc.)

        Returns:
            Document ID.
        """
        collection = self._get_collection("trading_research")
        doc_id = document.get("id", str(uuid.uuid4()))

        metadata = document.get("metadata", {})
        # ChromaDB metadata values must be str, int, float, or bool
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif isinstance(v, list):
                clean_metadata[k] = json.dumps(v)
            elif v is not None:
                clean_metadata[k] = str(v)

        # ChromaDB requires non-empty metadata or None
        upsert_kwargs = {
            "ids": [doc_id],
            "documents": [document.get("text", "")],
        }
        if clean_metadata:
            upsert_kwargs["metadatas"] = [clean_metadata]

        collection.upsert(**upsert_kwargs)
        return doc_id

    def search_research(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search through research database.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of matching documents with metadata and distance scores.
        """
        collection = self._get_collection("trading_research")
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        documents = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = {
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
                documents.append(doc)

        return documents

    # ─────────────────────────────────────────────────────────────────
    # TRADE PATTERN OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def add_trade_pattern(self, trade: Dict) -> str:
        """
        Add a trade pattern embedding for similarity search.

        Args:
            trade: Dict with keys: trade_id, text (natural language description),
                   metadata (strategy_id, outcome, return, market_regime, etc.)

        Returns:
            Document ID.
        """
        collection = self._get_collection("trade_patterns")
        doc_id = trade.get("trade_id", str(uuid.uuid4()))

        metadata = trade.get("metadata", {})
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif isinstance(v, list):
                clean_metadata[k] = json.dumps(v)
            elif v is not None:
                clean_metadata[k] = str(v)

        upsert_kwargs = {
            "ids": [doc_id],
            "documents": [trade.get("text", "")],
        }
        if clean_metadata:
            upsert_kwargs["metadatas"] = [clean_metadata]

        collection.upsert(**upsert_kwargs)
        return doc_id

    def find_similar_trades(self, description: str, top_k: int = 10) -> List[Dict]:
        """
        Find historically similar trade setups using semantic search.

        Args:
            description: Natural language description of current trade setup.
            top_k: Number of similar trades to return.

        Returns:
            List of similar trade patterns with metadata.
        """
        collection = self._get_collection("trade_patterns")
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[description],
            n_results=min(top_k, collection.count()),
        )

        trades = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                trade = {
                    "trade_id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
                trades.append(trade)

        return trades

    # ─────────────────────────────────────────────────────────────────
    # KNOWLEDGE BASE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def add_knowledge(self, document: Dict) -> str:
        """
        Add a document to the knowledge_base collection.

        Args:
            document: Dict with keys: id, text, metadata.

        Returns:
            Document ID.
        """
        collection = self._get_collection("knowledge_base")
        doc_id = document.get("id", str(uuid.uuid4()))

        metadata = document.get("metadata", {})
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif isinstance(v, list):
                clean_metadata[k] = json.dumps(v)
            elif v is not None:
                clean_metadata[k] = str(v)

        upsert_kwargs = {
            "ids": [doc_id],
            "documents": [document.get("text", "")],
        }
        if clean_metadata:
            upsert_kwargs["metadatas"] = [clean_metadata]

        collection.upsert(**upsert_kwargs)
        return doc_id

    def search_knowledge(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Semantic search through the knowledge base.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of matching documents with metadata and distance scores.
        """
        collection = self._get_collection("knowledge_base")
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        documents = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = {
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
                documents.append(doc)

        return documents

    # ─────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        stats = {}
        for name in ["trading_research", "trade_patterns", "knowledge_base"]:
            try:
                collection = self._get_collection(name)
                stats[name] = collection.count()
            except Exception:
                stats[name] = 0
        return stats

    def delete_collection(self, name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            self._collections.pop(name, None)
        except Exception as e:
            logger.warning(f"Failed to delete collection {name}: {e}")
