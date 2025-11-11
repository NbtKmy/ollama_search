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


class ChunkSummary(BaseModel):
    """Summary of a document chunk."""
    chunk_index: int
    section_title: Optional[str]
    summary: str
    key_points: List[str]


class PDFSummaryResult(BaseModel):
    """Complete PDF summarization result."""
    overall_summary: str
    key_takeaways: List[str]
    chunk_summaries: List[ChunkSummary]
    strategy_used: str  # 'structure' or 'size'
    total_chunks: int


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

    def summarize_chunk(self, chunk_text: str, chunk_index: int,
                       section_title: Optional[str] = None,
                       detail_level: str = "medium") -> ChunkSummary:
        """
        Summarize a single chunk of text.

        Args:
            chunk_text: Text content to summarize
            chunk_index: Index of this chunk
            section_title: Optional title of the section this chunk belongs to
            detail_level: Level of detail ('brief', 'medium', 'detailed')

        Returns:
            ChunkSummary with summary and key points
        """
        # Adjust prompt based on detail level
        if detail_level == "brief":
            length_instruction = "1-2 Sätze"
            num_points = 2
        elif detail_level == "detailed":
            length_instruction = "4-5 Sätze"
            num_points = 5
        else:  # medium
            length_instruction = "2-3 Sätze"
            num_points = 3

        system_prompt = """Du bist ein Experte für das Zusammenfassen von Dokumenten.
Erstelle prägnante, informative Zusammenfassungen, die die wichtigsten Informationen erfassen."""

        section_info = f"\nAbschnitt: {section_title}" if section_title else ""

        prompt = f"""Fasse den folgenden Textabschnitt zusammen:{section_info}

Text:
{chunk_text[:3000]}

Erstelle eine Zusammenfassung mit {length_instruction} und {num_points} wichtigen Punkten.
Antworte im folgenden JSON-Format:
{{
    "summary": "Eine prägnante Zusammenfassung ({length_instruction})",
    "key_points": [
        "Wichtiger Punkt 1",
        "Wichtiger Punkt 2"
    ]
}}"""

        response = self._chat(prompt, system_prompt)

        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            return ChunkSummary(
                chunk_index=chunk_index,
                section_title=section_title,
                summary=data.get("summary", ""),
                key_points=data.get("key_points", [])
            )
        except Exception as e:
            print(f"Error parsing chunk summary: {e}")
            return ChunkSummary(
                chunk_index=chunk_index,
                section_title=section_title,
                summary="Fehler beim Erstellen der Zusammenfassung",
                key_points=[]
            )

    def create_overall_summary(self, chunk_summaries: List[ChunkSummary],
                             detail_level: str = "medium") -> tuple[str, List[str]]:
        """
        Create an overall summary from chunk summaries.

        Args:
            chunk_summaries: List of chunk summaries to combine
            detail_level: Level of detail for final summary

        Returns:
            Tuple of (overall_summary, key_takeaways)
        """
        if detail_level == "brief":
            length_instruction = "2-3 Sätze"
            num_takeaways = 3
        elif detail_level == "detailed":
            length_instruction = "Ein ausführlicher Absatz (6-8 Sätze)"
            num_takeaways = 8
        else:  # medium
            length_instruction = "4-5 Sätze"
            num_takeaways = 5

        system_prompt = """Du bist ein Experte für das Zusammenfassen von Dokumenten.
Erstelle eine kohärente Gesamtzusammenfassung aus mehreren Teilzusammenfassungen."""

        # Build combined text from all chunk summaries
        combined_text = ""
        for i, chunk_summary in enumerate(chunk_summaries, 1):
            section_label = f" ({chunk_summary.section_title})" if chunk_summary.section_title else ""
            combined_text += f"\n\nAbschnitt {i}{section_label}:\n"
            combined_text += f"{chunk_summary.summary}\n"
            if chunk_summary.key_points:
                combined_text += "Wichtige Punkte:\n"
                for point in chunk_summary.key_points:
                    combined_text += f"  - {point}\n"

        prompt = f"""Basierend auf diesen Teilzusammenfassungen eines Dokuments:

{combined_text[:5000]}

Erstelle eine Gesamtzusammenfassung des gesamten Dokuments mit:
- Einer kohärenten Zusammenfassung ({length_instruction})
- {num_takeaways} wichtigsten Erkenntnissen aus dem gesamten Dokument

Antworte im folgenden JSON-Format:
{{
    "overall_summary": "Eine kohärente Gesamtzusammenfassung ({length_instruction})",
    "key_takeaways": [
        "Wichtigste Erkenntnis 1",
        "Wichtigste Erkenntnis 2"
    ]
}}"""

        response = self._chat(prompt, system_prompt)

        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            return (
                data.get("overall_summary", ""),
                data.get("key_takeaways", [])
            )
        except Exception as e:
            print(f"Error parsing overall summary: {e}")
            return (
                "Fehler beim Erstellen der Gesamtzusammenfassung",
                []
            )

    def summarize_pdf(self, chunks: List, strategy: str = "size",
                     detail_level: str = "medium") -> PDFSummaryResult:
        """
        Summarize a PDF document using hybrid strategy.

        Args:
            chunks: List of PDFChunk objects from pdf_processor
            strategy: Chunking strategy used ('structure' or 'size')
            detail_level: Level of detail ('brief', 'medium', 'detailed')

        Returns:
            PDFSummaryResult with complete summarization
        """
        print(f"Summarizing {len(chunks)} chunks with {strategy} strategy and {detail_level} detail level...")

        # Step 1: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
            chunk_summary = self.summarize_chunk(
                chunk_text=chunk.content,
                chunk_index=i,
                section_title=chunk.section_title,
                detail_level=detail_level
            )
            chunk_summaries.append(chunk_summary)

        # Step 2: Create overall summary from chunk summaries
        print("  Creating overall summary...")
        overall_summary, key_takeaways = self.create_overall_summary(
            chunk_summaries, detail_level
        )

        return PDFSummaryResult(
            overall_summary=overall_summary,
            key_takeaways=key_takeaways,
            chunk_summaries=chunk_summaries,
            strategy_used=strategy,
            total_chunks=len(chunks)
        )


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
