"""
AQTIS Research Agent.

Monitors external sources for new trading insights, academic research,
and market intelligence. Stores findings in the vector store.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Continuously monitors and integrates external research.

    Capabilities:
    - Scrape arXiv quantitative finance papers
    - Summarize papers using LLM
    - Assess relevance to current strategies
    - Search research database for solutions to specific problems
    """

    def __init__(self, memory, llm=None, relevance_threshold: float = 0.6):
        super().__init__(name="researcher", memory=memory, llm=llm)
        self.relevance_threshold = relevance_threshold

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research action."""
        action = context.get("action", "scan")

        if action == "scan":
            return self.daily_research_scan()
        elif action == "search":
            return self.search_for_solution(context.get("query", ""))
        elif action == "add_paper":
            return self.add_paper(context.get("paper", {}))
        else:
            return {"error": f"Unknown action: {action}"}

    # ─────────────────────────────────────────────────────────────────
    # DAILY SCAN
    # ─────────────────────────────────────────────────────────────────

    def daily_research_scan(self) -> Dict:
        """
        Run daily scan of research sources.

        Fetches recent arXiv q-fin papers, summarizes them, and stores
        relevant ones in the vector database.
        """
        papers = self._fetch_arxiv_papers()

        new_papers = []
        for paper in papers:
            summary = self._summarize_paper(paper)
            relevance = summary.get("relevance_score", 0)

            if relevance >= self.relevance_threshold:
                doc_id = self.memory.store_research({
                    "id": paper.get("id"),
                    "text": f"{paper.get('title', '')}. {paper.get('abstract', '')}",
                    "metadata": {
                        "title": paper.get("title", ""),
                        "authors": json.dumps(paper.get("authors", [])),
                        "url": paper.get("url", ""),
                        "published": paper.get("published", ""),
                        "relevance_score": relevance,
                        "source": "arxiv",
                    },
                })
                new_papers.append({
                    "title": paper.get("title"),
                    "relevance": relevance,
                    "doc_id": doc_id,
                    "summary": summary,
                })

        return {
            "papers_scanned": len(papers),
            "relevant_papers": len(new_papers),
            "papers": new_papers,
        }

    def _fetch_arxiv_papers(self, max_results: int = 20) -> List[Dict]:
        """Fetch recent quantitative finance papers from arXiv."""
        try:
            import arxiv
            search = arxiv.Search(
                query="cat:q-fin",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            papers = []
            client = arxiv.Client()
            for result in client.results(search):
                papers.append({
                    "id": result.entry_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [a.name for a in result.authors],
                    "url": result.pdf_url,
                    "published": result.published.isoformat() if result.published else "",
                    "categories": [c for c in result.categories],
                })

            self.logger.info(f"Fetched {len(papers)} papers from arXiv")
            return papers

        except ImportError:
            self.logger.warning("arxiv package not installed: pip install arxiv")
            return []
        except Exception as e:
            self.logger.error(f"arXiv fetch error: {e}")
            return []

    def _summarize_paper(self, paper: Dict) -> Dict:
        """Use LLM to extract key insights from a paper."""
        if not self.llm or not self.llm.is_available():
            # Rule-based relevance scoring
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()

            keywords = [
                "trading", "momentum", "mean reversion", "volatility",
                "prediction", "machine learning", "deep learning",
                "portfolio", "risk", "alpha", "factor", "signal",
                "backtest", "strategy", "market microstructure",
            ]
            keyword_hits = sum(1 for k in keywords if k in title or k in abstract)
            relevance = min(keyword_hits / 5, 1.0)

            return {
                "key_findings": [],
                "relevance_score": relevance,
                "method": "keyword_matching",
            }

        prompt = f"""Analyze this quantitative finance research paper and extract actionable insights.

Title: {paper.get('title')}
Abstract: {paper.get('abstract', '')[:1500]}

Extract:
1. Key findings relevant to algorithmic trading
2. Applicable trading strategies
3. Market conditions tested
4. Relevance score (0-1) for a quantitative trader

Respond in JSON:
{{
    "key_findings": ["finding1", "finding2"],
    "applicable_strategies": ["strategy1"],
    "market_conditions": ["condition1"],
    "implementation_notes": "How to implement",
    "limitations": ["limitation1"],
    "relevance_score": 0.7
}}"""

        return self.llm.generate_json(prompt)

    # ─────────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────────

    def search_for_solution(self, problem: str) -> Dict:
        """
        Search research database for solutions to a specific trading problem.
        """
        if not problem:
            return {"error": "No query provided"}

        relevant = self.memory.search_research(problem, top_k=10)

        if not relevant:
            return {
                "problem": problem,
                "papers_found": 0,
                "message": "No relevant research found",
            }

        # Synthesize if LLM available
        synthesis = None
        if self.llm and self.llm.is_available():
            papers_text = []
            for r in relevant[:5]:
                papers_text.append(r.get("text", "")[:500])

            prompt = f"""Problem: {problem}

Relevant research:
{json.dumps(papers_text, indent=2)}

Synthesize actionable recommendations in JSON:
{{
    "recommended_approaches": ["approach1", "approach2"],
    "key_techniques": ["technique1"],
    "implementation_steps": ["step1", "step2"],
    "risks_and_caveats": ["caveat1"]
}}"""
            synthesis = self.llm.generate_json(prompt)

        return {
            "problem": problem,
            "papers_found": len(relevant),
            "papers": [
                {
                    "title": r.get("metadata", {}).get("title", ""),
                    "relevance": r.get("distance"),
                    "text": r.get("text", "")[:200],
                }
                for r in relevant[:5]
            ],
            "synthesis": synthesis,
        }

    # ─────────────────────────────────────────────────────────────────
    # MANUAL PAPER ADD
    # ─────────────────────────────────────────────────────────────────

    def add_paper(self, paper: Dict) -> Dict:
        """Manually add a research paper to the knowledge base."""
        title = paper.get("title", "")
        text = paper.get("text", paper.get("abstract", ""))

        if not text:
            return {"error": "Paper text/abstract required"}

        summary = self._summarize_paper(paper)

        doc_id = self.memory.store_research({
            "text": f"{title}. {text}",
            "metadata": {
                "title": title,
                "authors": json.dumps(paper.get("authors", [])),
                "url": paper.get("url", ""),
                "source": paper.get("source", "manual"),
                "relevance_score": summary.get("relevance_score", 0.5),
            },
        })

        return {
            "doc_id": doc_id,
            "title": title,
            "summary": summary,
        }
