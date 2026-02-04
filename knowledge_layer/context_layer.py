"""
Knowledge Context Layer for 5-Player Coach Trading System.

This is the main interface the coach and players use to retrieve
relevant knowledge for their prompts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .scripts.query import query_knowledge, get_all_sources
from .scripts.ingest import get_ingestion_status

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "knowledge_layer" / "config" / "settings.yaml"


def load_config() -> Dict:
    """Load configuration from settings.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class KnowledgeContext:
    """
    Interface for the coach system to retrieve relevant knowledge.

    Usage:
        knowledge = KnowledgeContext()
        context = knowledge.get_context_for_query("Should I buy RELIANCE?")
        # Inject context into your prompt
    """

    def __init__(self):
        """Initialize the knowledge context layer."""
        self.config = load_config()
        self._player_categories = self.config.get("player_categories", {})

    def get_context_for_query(
        self,
        user_query: str,
        player_role: str = None,
        max_tokens: int = 2000,
        categories: List[str] = None
    ) -> str:
        """
        Get formatted context string ready to inject into prompts.

        Args:
            user_query: The user's question or trading query
            player_role: Optional player role to tailor retrieval
            max_tokens: Maximum tokens for context (approximate)
            categories: Specific categories to filter (overrides player_role)

        Returns:
            Formatted context string ready for prompt injection
        """
        # Determine categories to search
        if categories:
            search_categories = categories
        elif player_role and player_role.lower() in self._player_categories:
            search_categories = self._player_categories[player_role.lower()]
        else:
            search_categories = None  # Search all

        # Query the knowledge base
        results = query_knowledge(
            query=user_query,
            categories=search_categories,
            top_k=10,  # Get more, then trim by token limit
            use_mmr=True
        )

        if not results:
            return self._format_no_results()

        # Build context string with token limit
        context_parts = []
        estimated_tokens = 0
        tokens_per_char = 0.25  # Rough estimate

        for result in results:
            chunk_text = self._format_chunk(result)
            chunk_tokens = int(len(chunk_text) * tokens_per_char)

            if estimated_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            estimated_tokens += chunk_tokens

        return self._format_context(context_parts, user_query)

    def get_context_for_player(
        self,
        player_name: str,
        market_context: Dict,
        query: str,
        max_tokens: int = 1500
    ) -> str:
        """
        Get tailored context for a specific player based on their specialty.

        Args:
            player_name: Name of the player (aggressive, conservative, etc.)
            market_context: Current market state (volatility, trend, etc.)
            query: The trading query to answer
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context tailored to the player's specialty
        """
        # Get player's specialty categories
        player_key = player_name.lower()
        if player_key not in self._player_categories:
            # Default to all categories
            categories = None
        else:
            categories = self._player_categories[player_key]

        # Enhance query with market context
        enhanced_query = self._enhance_query_with_context(query, market_context)

        # Get relevant knowledge
        results = query_knowledge(
            query=enhanced_query,
            categories=categories,
            top_k=8,
            use_mmr=True
        )

        if not results:
            return self._format_no_results_for_player(player_name)

        # Build context with player context
        context_parts = []
        estimated_tokens = 0
        tokens_per_char = 0.25

        for result in results:
            chunk_text = self._format_chunk(result)
            chunk_tokens = int(len(chunk_text) * tokens_per_char)

            if estimated_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            estimated_tokens += chunk_tokens

        return self._format_player_context(
            context_parts, player_name, market_context, query
        )

    def get_context_for_coach(
        self,
        situation: str,
        players_analysis: Dict[str, str] = None,
        max_tokens: int = 2500
    ) -> str:
        """
        Get context for the coach to orchestrate players.

        Args:
            situation: Description of the current trading situation
            players_analysis: Dict of player name -> their analysis (optional)
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context for coach decision-making
        """
        # Search across all categories for coach
        results = query_knowledge(
            query=situation,
            categories=None,  # All categories
            top_k=12,
            use_mmr=True
        )

        if not results:
            return self._format_no_results()

        # Build context
        context_parts = []
        estimated_tokens = 0
        tokens_per_char = 0.25

        for result in results:
            chunk_text = self._format_chunk(result)
            chunk_tokens = int(len(chunk_text) * tokens_per_char)

            if estimated_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            estimated_tokens += chunk_tokens

        return self._format_coach_context(context_parts, situation, players_analysis)

    def search_specific_topic(
        self,
        topic: str,
        category: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for a specific topic and return raw results.

        Useful for deep dives into specific concepts.
        """
        return query_knowledge(
            query=topic,
            categories=[category] if category else None,
            top_k=top_k,
            use_mmr=False  # Pure relevance for specific searches
        )

    def get_risk_management_context(self, position_info: Dict) -> str:
        """
        Get risk management specific context.

        Args:
            position_info: Dict with position details (size, entry, stop, etc.)

        Returns:
            Risk management focused context
        """
        query = f"position sizing risk management stop loss {position_info.get('volatility', '')} {position_info.get('trend', '')}"

        results = query_knowledge(
            query=query,
            categories=["risk_management"],
            top_k=5,
            use_mmr=True
        )

        if not results:
            return ""

        context_parts = [self._format_chunk(r) for r in results[:3]]
        return "\n\n".join(context_parts)

    def get_technical_analysis_context(self, setup: str) -> str:
        """
        Get technical analysis context for a specific setup.

        Args:
            setup: Description of the technical setup

        Returns:
            Technical analysis focused context
        """
        results = query_knowledge(
            query=setup,
            categories=["technical_analysis"],
            top_k=5,
            use_mmr=True
        )

        if not results:
            return ""

        context_parts = [self._format_chunk(r) for r in results[:3]]
        return "\n\n".join(context_parts)

    def get_available_sources(self) -> List[Dict]:
        """Get list of all available knowledge sources."""
        return get_all_sources()

    def get_status(self) -> Dict:
        """Get knowledge base status."""
        return get_ingestion_status()

    # ==================== Private Helper Methods ====================

    def _format_chunk(self, result: Dict) -> str:
        """Format a single result chunk."""
        category = result.get("category", "unknown").replace("_", " ").title()
        source = result.get("source", "Unknown Source")
        content = result.get("content", "").strip()

        # Truncate very long chunks
        if len(content) > 800:
            content = content[:800] + "..."

        return f"[{category} - {source}]\n\"{content}\""

    def _format_context(self, chunks: List[str], query: str) -> str:
        """Format the full context block."""
        if not chunks:
            return self._format_no_results()

        context = "---\nRELEVANT KNOWLEDGE:\n\n"
        context += "\n\n".join(chunks)
        context += "\n---"

        return context

    def _format_player_context(
        self,
        chunks: List[str],
        player_name: str,
        market_context: Dict,
        query: str
    ) -> str:
        """Format context for a specific player."""
        if not chunks:
            return self._format_no_results_for_player(player_name)

        market_summary = self._summarize_market_context(market_context)

        context = f"---\nKNOWLEDGE FOR {player_name.upper()} PLAYER:\n"
        context += f"Market Context: {market_summary}\n\n"
        context += "\n\n".join(chunks)
        context += "\n---"

        return context

    def _format_coach_context(
        self,
        chunks: List[str],
        situation: str,
        players_analysis: Dict = None
    ) -> str:
        """Format context for the coach."""
        if not chunks:
            return self._format_no_results()

        context = "---\nCOACH KNOWLEDGE BASE:\n\n"
        context += "\n\n".join(chunks)

        if players_analysis:
            context += "\n\nPLAYER ANALYSES:\n"
            for player, analysis in players_analysis.items():
                context += f"\n{player}: {analysis}"

        context += "\n---"

        return context

    def _format_no_results(self) -> str:
        """Format message when no results found."""
        return "---\nNo relevant knowledge found in the knowledge base.\n---"

    def _format_no_results_for_player(self, player_name: str) -> str:
        """Format no results message for a player."""
        return f"---\nNo specific knowledge found for {player_name} player's specialty.\n---"

    def _enhance_query_with_context(self, query: str, market_context: Dict) -> str:
        """Enhance query with market context for better retrieval."""
        enhancements = []

        if market_context.get("volatility"):
            enhancements.append(f"{market_context['volatility']} volatility")

        if market_context.get("trend"):
            enhancements.append(f"{market_context['trend']} trend")

        if market_context.get("regime"):
            enhancements.append(f"{market_context['regime']} market")

        if enhancements:
            return f"{query} {' '.join(enhancements)}"
        return query

    def _summarize_market_context(self, market_context: Dict) -> str:
        """Create a brief summary of market context."""
        parts = []

        if market_context.get("volatility"):
            parts.append(f"volatility: {market_context['volatility']}")

        if market_context.get("trend"):
            parts.append(f"trend: {market_context['trend']}")

        if market_context.get("regime"):
            parts.append(f"regime: {market_context['regime']}")

        return ", ".join(parts) if parts else "normal conditions"


# Convenience function for quick context retrieval
def get_trading_context(query: str, max_tokens: int = 2000) -> str:
    """
    Quick function to get trading knowledge context.

    Usage:
        context = get_trading_context("Should I buy RELIANCE after pullback to 50 EMA?")
    """
    knowledge = KnowledgeContext()
    return knowledge.get_context_for_query(query, max_tokens=max_tokens)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create knowledge context
    knowledge = KnowledgeContext()

    # Check status
    print("\n=== Knowledge Base Status ===")
    status = knowledge.get_status()
    print(f"Documents: {status['total_documents']}")
    print(f"Chunks: {status['total_chunks']}")

    # Example query
    print("\n=== Example Query ===")
    context = knowledge.get_context_for_query(
        "How should I size my position when volatility is high?",
        max_tokens=1000
    )
    print(context)

    # Player-specific context
    print("\n=== Player Context Example ===")
    player_context = knowledge.get_context_for_player(
        player_name="conservative",
        market_context={"volatility": "high", "trend": "bearish"},
        query="Should I reduce position size?",
        max_tokens=800
    )
    print(player_context)
