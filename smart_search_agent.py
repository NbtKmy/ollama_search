"""Smart Search Agent - Combines Web Search with LLM Intelligence

This agent integrates:
- Web search with Playwright
- Query generation with LLM
- Result evaluation and ranking
- Intelligent summarization
"""

import asyncio
from pathlib import Path
from typing import List, Optional
import json

from pydantic import BaseModel

from search_agent import WebSearchAgent, SearchReport
from llm_agent import LLMAgent, RankedResult, SearchSummary


class SmartSearchResult(BaseModel):
    """Complete smart search result with all enhancements."""
    original_intent: str
    queries_used: List[str]
    all_reports: List[SearchReport]
    top_ranked_results: List[RankedResult]
    summary: SearchSummary


class SmartSearchAgent:
    """Intelligent search agent combining web search and LLM capabilities."""

    def __init__(
        self,
        llm_model: str = "gpt-oss:20b",
        headless: bool = True,
        ollama_base_url: Optional[str] = None
    ):
        """
        Initialize Smart Search Agent.

        Args:
            llm_model: Ollama model to use (default: gpt-oss:20b)
            headless: Run browser in headless mode
            ollama_base_url: Optional custom Ollama server URL
        """
        self.search_agent = WebSearchAgent(headless=headless)
        self.llm_agent = LLMAgent(model=llm_model, base_url=ollama_base_url)

    async def smart_search(
        self,
        user_intent: str,
        num_queries: int = 3,
        results_per_query: int = 5,
        top_k: int = 5
    ) -> SmartSearchResult:
        """
        Perform an intelligent search with query generation, evaluation, and summarization.

        Args:
            user_intent: User's search intent in natural language
            num_queries: Number of search queries to generate
            results_per_query: Number of results per query
            top_k: Number of top results to return after ranking

        Returns:
            SmartSearchResult with all search intelligence
        """
        print(f"\n{'='*60}")
        print(f"Smart Search: {user_intent}")
        print(f"{'='*60}\n")

        # Step 1: Generate search queries from user intent
        print("🔍 Schritt 1: Generiere Suchanfragen...")
        suggestions = self.llm_agent.generate_queries(user_intent, num_queries)

        print(f"\nGenerierte Anfragen:")
        for i, query in enumerate(suggestions.queries, 1):
            print(f"  {i}. {query}")
        print(f"\nBegründung: {suggestions.reasoning}\n")

        # Step 2: Perform searches for each generated query
        print("🌐 Schritt 2: Führe Web-Suchen durch...")
        all_reports = []
        all_results = []

        for i, query in enumerate(suggestions.queries, 1):
            print(f"  [{i}/{len(suggestions.queries)}] Suche: {query}")
            report = await self.search_agent.search_duckduckgo(query, results_per_query)
            all_reports.append(report)
            all_results.extend(report.results)
            print(f"      → {report.result_count} Ergebnisse gefunden")

            # Small delay between searches
            if i < len(suggestions.queries):
                await asyncio.sleep(1)

        print(f"\nInsgesamt: {len(all_results)} Ergebnisse gefunden\n")

        # Step 3: Evaluate and rank all results
        print("⭐ Schritt 3: Bewerte und ranke Ergebnisse...")
        ranked_results = self.llm_agent.evaluate_results(
            user_intent,
            all_results,
            top_k=top_k
        )

        print(f"Top {len(ranked_results)} Ergebnisse nach Relevanz:\n")
        for i, ranked in enumerate(ranked_results, 1):
            print(f"  {i}. {ranked.result.title}")
            print(f"     Relevanz: {ranked.relevance_score:.2f} - {ranked.reasoning}")

        # Step 4: Create summary from best results
        print(f"\n📝 Schritt 4: Erstelle Zusammenfassung...")

        # Use the first report for summary (or could combine multiple)
        best_report = all_reports[0] if all_reports else None
        summary = self.llm_agent.summarize_results(best_report)

        print(f"\nZusammenfassung:\n{summary.summary}\n")

        return SmartSearchResult(
            original_intent=user_intent,
            queries_used=suggestions.queries,
            all_reports=all_reports,
            top_ranked_results=ranked_results,
            summary=summary
        )

    def save_smart_result(self, result: SmartSearchResult, output_path: Path):
        """Save complete smart search result to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                result.model_dump(mode='json'),
                f,
                indent=2,
                ensure_ascii=False,
                default=str
            )

        print(f"\n✅ Ergebnisse gespeichert: {output_path}")

    async def quick_answer(self, question: str) -> str:
        """
        Get a quick answer to a question using search + LLM.

        Args:
            question: Question to answer

        Returns:
            Answer string
        """
        # Generate optimized search query
        suggestions = self.llm_agent.generate_queries(question, num_queries=1)
        query = suggestions.queries[0]

        # Search
        report = await self.search_agent.search_duckduckgo(query, max_results=3)

        # Summarize
        summary = self.llm_agent.summarize_results(report)

        return summary.summary


async def demo():
    """Demonstration of Smart Search Agent."""
    agent = SmartSearchAgent(llm_model="gpt-oss:20b", headless=True)

    # Example 1: Comprehensive smart search
    print("\n" + "="*60)
    print("DEMO: Smart Search Agent mit gpt-oss:20b")
    print("="*60)

    result = await agent.smart_search(
        user_intent="Wie erstelle ich einen Web-Scraper mit Python und Playwright?",
        num_queries=2,
        results_per_query=5,
        top_k=5
    )

    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG DER ERGEBNISSE")
    print("="*60)
    print(f"\nUrsprüngliche Absicht: {result.original_intent}")
    print(f"\nVerwendete Suchanfragen:")
    for i, q in enumerate(result.queries_used, 1):
        print(f"  {i}. {q}")

    print(f"\n📊 Zusammenfassung:")
    print(f"{result.summary.summary}\n")

    print("🔑 Wichtige Punkte:")
    for point in result.summary.key_points:
        print(f"  • {point}")

    print(f"\n⭐ Top {len(result.top_ranked_results)} relevante Ergebnisse:")
    for i, ranked in enumerate(result.top_ranked_results, 1):
        print(f"\n  {i}. {ranked.result.title}")
        print(f"     URL: {ranked.result.url}")
        print(f"     Relevanz: {ranked.relevance_score:.2f}")
        print(f"     → {ranked.reasoning}")

    # Save results
    output_path = Path("output/smart_search_demo.json")
    agent.save_smart_result(result, output_path)

    # Example 2: Quick answer
    print("\n\n" + "="*60)
    print("DEMO: Quick Answer")
    print("="*60)

    question = "Was ist asyncio in Python?"
    print(f"\nFrage: {question}")
    print("\nAntwort:")

    answer = await agent.quick_answer(question)
    print(f"{answer}")


if __name__ == "__main__":
    asyncio.run(demo())
