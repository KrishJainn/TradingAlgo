"""
RAG Query Interface for Trading Knowledge Base.

Provides semantic search with category filtering and MMR diversity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "knowledge_layer" / "config" / "settings.yaml"


def load_config() -> Dict:
    """Load configuration from settings.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_chroma_client():
    """Get ChromaDB client."""
    try:
        import chromadb
        from chromadb.config import Settings

        config = load_config()
        persist_dir = PROJECT_ROOT / config["chroma"]["persist_directory"]

        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        return client
    except ImportError:
        logger.error("chromadb not installed")
        raise


def get_embedding_function():
    """Get the embedding function."""
    try:
        from chromadb.utils import embedding_functions

        config = load_config()
        model_name = config["embedding"]["model"]

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    except ImportError:
        logger.error("sentence-transformers not installed")
        raise


def get_collection():
    """Get the ChromaDB collection."""
    config = load_config()
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    try:
        collection = client.get_collection(
            name=config["chroma"]["collection_name"],
            embedding_function=embedding_fn
        )
        return collection
    except Exception:
        logger.warning("Collection not found. Run ingestion first.")
        return None


def query_knowledge(
    query: str,
    categories: List[str] = None,
    top_k: int = None,
    use_mmr: bool = None,
    mmr_diversity: float = None
) -> List[Dict]:
    """
    Query the knowledge base for relevant chunks.

    Args:
        query: The search query
        categories: Filter to specific categories (optional)
        top_k: Number of results to return
        use_mmr: Use Maximum Marginal Relevance for diversity
        mmr_diversity: MMR diversity parameter (0-1)

    Returns:
        List of results with content, source, category, relevance_score, metadata
    """
    config = load_config()

    # Apply defaults from config
    if top_k is None:
        top_k = config["retrieval"]["default_top_k"]
    if use_mmr is None:
        use_mmr = config["retrieval"]["use_mmr"]
    if mmr_diversity is None:
        mmr_diversity = config["retrieval"]["mmr_diversity"]

    collection = get_collection()
    if collection is None:
        logger.warning("No collection found. Returning empty results.")
        return []

    # Build where clause for category filtering
    where_clause = None
    if categories:
        if len(categories) == 1:
            where_clause = {"category": categories[0]}
        else:
            where_clause = {"category": {"$in": categories}}

    # Query with more results if using MMR (we'll filter down)
    n_results = top_k * 3 if use_mmr else top_k

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        return []

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    # Process results
    processed = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert distance to similarity score (ChromaDB uses L2 distance)
        # Smaller distance = more similar
        relevance_score = 1 / (1 + dist)

        processed.append({
            "content": doc,
            "source": meta.get("source", "Unknown"),
            "category": meta.get("category", "unknown"),
            "relevance_score": round(relevance_score, 4),
            "metadata": {
                "chunk_index": meta.get("chunk_index"),
                "total_chunks": meta.get("total_chunks"),
                "file_type": meta.get("file_type"),
                "ingested_at": meta.get("ingested_at"),
            }
        })

    # Apply MMR if requested
    if use_mmr and len(processed) > top_k:
        processed = apply_mmr(processed, top_k, mmr_diversity)

    return processed[:top_k]


def apply_mmr(
    results: List[Dict],
    top_k: int,
    diversity: float = 0.3
) -> List[Dict]:
    """
    Apply Maximum Marginal Relevance to diversify results.

    MMR balances relevance with diversity by penalizing results
    that are too similar to already selected results.

    Args:
        results: List of search results
        top_k: Number of results to return
        diversity: Diversity weight (0 = pure relevance, 1 = pure diversity)

    Returns:
        Reranked list of results
    """
    if len(results) <= top_k:
        return results

    # Simple MMR implementation using content overlap
    selected = [results[0]]  # Start with most relevant
    candidates = results[1:]

    while len(selected) < top_k and candidates:
        best_score = float("-inf")
        best_idx = 0

        for i, candidate in enumerate(candidates):
            # Relevance score
            relevance = candidate["relevance_score"]

            # Compute max similarity to already selected (simple word overlap)
            max_sim = 0
            candidate_words = set(candidate["content"].lower().split())

            for sel in selected:
                sel_words = set(sel["content"].lower().split())
                if candidate_words and sel_words:
                    overlap = len(candidate_words & sel_words)
                    sim = overlap / max(len(candidate_words), len(sel_words))
                    max_sim = max(max_sim, sim)

            # MMR score: (1 - diversity) * relevance - diversity * max_similarity
            mmr_score = (1 - diversity) * relevance - diversity * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(candidates[best_idx])
        candidates.pop(best_idx)

    return selected


def query_by_category(category: str, query: str, top_k: int = 5) -> List[Dict]:
    """Convenience function to query a specific category."""
    return query_knowledge(query, categories=[category], top_k=top_k)


def get_all_sources() -> List[Dict]:
    """Get list of all ingested sources."""
    collection = get_collection()
    if collection is None:
        return []

    try:
        # Get unique sources
        results = collection.get(include=["metadatas"])
        sources = {}

        for meta in results["metadatas"]:
            source = meta.get("source", "Unknown")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "category": meta.get("category"),
                    "file_type": meta.get("file_type"),
                    "chunks": 0
                }
            sources[source]["chunks"] += 1

        return list(sources.values())
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        return []


def search_by_source(source_name: str, query: str, top_k: int = 5) -> List[Dict]:
    """Search within a specific source document."""
    collection = get_collection()
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"source": source_name},
            include=["documents", "metadatas", "distances"]
        )

        processed = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                processed.append({
                    "content": doc,
                    "source": meta.get("source"),
                    "category": meta.get("category"),
                    "relevance_score": round(1 / (1 + dist), 4),
                    "metadata": meta
                })

        return processed
    except Exception as e:
        logger.error(f"Error searching source: {e}")
        return []


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Query the trading knowledge base")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("-c", "--category", nargs="+", help="Filter by categories")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("-l", "--list-sources", action="store_true", help="List all sources")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR diversity")

    args = parser.parse_args()

    if args.list_sources:
        sources = get_all_sources()
        print(f"\n=== Ingested Sources ({len(sources)}) ===")
        for src in sources:
            print(f"  [{src['category']}] {src['source']} ({src['chunks']} chunks)")
    elif args.query:
        results = query_knowledge(
            args.query,
            categories=args.category,
            top_k=args.top_k,
            use_mmr=not args.no_mmr
        )

        print(f"\n=== Results for: '{args.query}' ===\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['category']}] {r['source']} (score: {r['relevance_score']})")
            print(f"   {r['content'][:200]}...")
            print()
    else:
        parser.print_help()
