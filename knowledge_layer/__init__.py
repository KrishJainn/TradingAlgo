"""
RAG Knowledge Layer for 5-Player Coach Trading System.

Provides retrieval-augmented generation capabilities using:
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Smart chunking with boundary respect

Usage:
    from knowledge_layer import KnowledgeContext, get_trading_context

    # Quick context retrieval
    context = get_trading_context("How to size positions in volatile markets?")

    # Full interface
    knowledge = KnowledgeContext()
    player_context = knowledge.get_context_for_player(
        player_name="aggressive",
        market_context={"volatility": "high"},
        query="Should I increase position size?"
    )
"""

__version__ = "1.0.0"

from .context_layer import KnowledgeContext, get_trading_context

__all__ = ["KnowledgeContext", "get_trading_context"]
