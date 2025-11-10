"""LLM Agent using Ollama

This agent provides:
- Query generation from user intent
- Search result summarization
- Search result evaluation and ranking
"""

from typing import List, Optional, Dict, Any
import json

import ollama
from pydantic import BaseModel

from search_agent import SearchResult, SearchReport


class RankedResult(BaseModel):
    """A search result with relevance score."""
    result: SearchResult
    relevance_score: float
    reasoning: str


class QuerySuggestions(BaseModel):
    """Generated search query suggestions."""
    original_intent: str
    queries: List[str]
    reasoning: str


class SearchSummary(BaseModel):
    """Summary of search results."""
    query: str
    summary: str
    key_points: List[str]
    result_count: int


class LLMAgent:
    """Agent using Ollama LLM for intelligent search operations."""

    def __init__(self, model: str = "gpt-oss:20b", base_url: Optional[str] = None):
        """
        Initialize LLM Agent.

        Args:
            model: Ollama model name (default: gpt-oss:20b)
            base_url: Optional custom Ollama server URL
        """
        self.model = model
        self.client = ollama.Client(host=base_url) if base_url else ollama

    def _chat(self, prompt: str, system: Optional[str] = None) -> str:
        """Send a chat message to the LLM."""
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages
        )

        return response['message']['content']

    def generate_queries(self, user_intent: str, num_queries: int = 3) -> QuerySuggestions:
        """
        Generate search queries based on user intent.

        Args:
            user_intent: User's search intent in natural language
            num_queries: Number of queries to generate

        Returns:
            QuerySuggestions with generated queries
        """
        system_prompt = """Du bist ein Experte für Suchmaschinenoptimierung.
Deine Aufgabe ist es, effektive Suchanfragen zu generieren, die die Absicht des Benutzers am besten erfassen."""

        prompt = f"""Basierend auf dieser Benutzerabsicht: "{user_intent}"

Generiere {num_queries} verschiedene, effektive Suchanfragen, die verschiedene Aspekte des Themas abdecken.

Antworte im folgenden JSON-Format:
{{
    "queries": ["query1", "query2", "query3"],
    "reasoning": "Kurze Erklärung, warum diese Queries gewählt wurden"
}}"""

        response = self._chat(prompt, system_prompt)

        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            return QuerySuggestions(
                original_intent=user_intent,
                queries=data.get("queries", []),
                reasoning=data.get("reasoning", "")
            )
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            # Fallback: use original intent as query
            return QuerySuggestions(
                original_intent=user_intent,
                queries=[user_intent],
                reasoning="Fallback: using original intent"
            )

    def summarize_results(self, report: SearchReport) -> SearchSummary:
        """
        Summarize search results using LLM.

        Args:
            report: SearchReport to summarize

        Returns:
            SearchSummary with summary and key points
        """
        system_prompt = """Du bist ein Experte für das Zusammenfassen von Informationen.
Erstelle prägnante, informative Zusammenfassungen von Suchergebnissen."""

        # Build context from results
        results_text = ""
        for i, result in enumerate(report.results, 1):
            results_text += f"\n{i}. {result.title}\n"
            results_text += f"   URL: {result.url}\n"
            results_text += f"   {result.snippet}\n"

        prompt = f"""Suchanfrage: "{report.query}"

Suchergebnisse:
{results_text}

Erstelle eine Zusammenfassung dieser Suchergebnisse. Antworte im folgenden JSON-Format:
{{
    "summary": "Eine prägnante Zusammenfassung (2-3 Sätze) der wichtigsten Informationen",
    "key_points": [
        "Wichtiger Punkt 1",
        "Wichtiger Punkt 2",
        "Wichtiger Punkt 3"
    ]
}}"""

        response = self._chat(prompt, system_prompt)

        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            return SearchSummary(
                query=report.query,
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                result_count=report.result_count
            )
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return SearchSummary(
                query=report.query,
                summary="Fehler beim Erstellen der Zusammenfassung",
                key_points=[],
                result_count=report.result_count
            )

    def evaluate_results(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[RankedResult]:
        """
        Evaluate and rank search results by relevance.

        Args:
            query: Original search query
            results: List of search results to evaluate
            top_k: Number of top results to return

        Returns:
            List of RankedResult sorted by relevance
        """
        system_prompt = """Du bist ein Experte für die Bewertung der Relevanz von Suchergebnissen.
Bewerte jedes Ergebnis auf einer Skala von 0.0 bis 1.0 basierend auf seiner Relevanz zur Suchanfrage."""

        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n{i}. Titel: {result.title}\n"
            results_text += f"   URL: {result.url}\n"
            results_text += f"   Snippet: {result.snippet}\n"

        prompt = f"""Suchanfrage: "{query}"

Bewerte die Relevanz dieser Suchergebnisse:
{results_text}

Antworte im folgenden JSON-Format mit einem Array von Bewertungen:
{{
    "evaluations": [
        {{
            "index": 1,
            "score": 0.95,
            "reasoning": "Sehr relevant weil..."
        }},
        {{
            "index": 2,
            "score": 0.75,
            "reasoning": "Teilweise relevant weil..."
        }}
    ]
}}"""

        response = self._chat(prompt, system_prompt)

        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Create ranked results
            ranked = []
            for eval_data in data.get("evaluations", []):
                idx = eval_data.get("index", 0) - 1  # Convert to 0-based
                if 0 <= idx < len(results):
                    ranked.append(RankedResult(
                        result=results[idx],
                        relevance_score=eval_data.get("score", 0.5),
                        reasoning=eval_data.get("reasoning", "")
                    ))

            # Sort by relevance score
            ranked.sort(key=lambda x: x.relevance_score, reverse=True)

            return ranked[:top_k]

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Fallback: return results in original order with neutral scores
            return [
                RankedResult(
                    result=result,
                    relevance_score=0.5,
                    reasoning="Fallback ranking"
                )
                for result in results[:top_k]
            ]


async def main():
    """Example usage of LLM Agent."""
    from search_agent import WebSearchAgent

    print("=== LLM-Enhanced Search Agent Demo ===\n")

    llm_agent = LLMAgent(model="gpt-oss:20b")
    search_agent = WebSearchAgent(headless=True)

    # 1. Generate queries from user intent
    print("1. Query Generation")
    print("-" * 50)
    user_intent = "Ich möchte lernen, wie man asynchrone Web-Scraper in Python erstellt"
    print(f"Benutzerabsicht: {user_intent}\n")

    suggestions = llm_agent.generate_queries(user_intent, num_queries=3)
    print("Generierte Suchanfragen:")
    for i, query in enumerate(suggestions.queries, 1):
        print(f"  {i}. {query}")
    print(f"\nBegründung: {suggestions.reasoning}\n")

    # 2. Search using first generated query
    print("\n2. Suche durchführen")
    print("-" * 50)
    query = suggestions.queries[0]
    print(f"Suche nach: {query}\n")

    report = await search_agent.search_duckduckgo(query, max_results=5)
    print(f"Gefunden: {report.result_count} Ergebnisse\n")

    # 3. Evaluate and rank results
    print("\n3. Ergebnisse bewerten und ranken")
    print("-" * 50)
    ranked_results = llm_agent.evaluate_results(query, report.results, top_k=3)

    for i, ranked in enumerate(ranked_results, 1):
        print(f"{i}. {ranked.result.title}")
        print(f"   Relevanz: {ranked.relevance_score:.2f}")
        print(f"   URL: {ranked.result.url}")
        print(f"   Begründung: {ranked.reasoning}\n")

    # 4. Summarize results
    print("\n4. Ergebnisse zusammenfassen")
    print("-" * 50)
    summary = llm_agent.summarize_results(report)
    print(f"Zusammenfassung:\n{summary.summary}\n")
    print("Wichtige Punkte:")
    for point in summary.key_points:
        print(f"  • {point}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
