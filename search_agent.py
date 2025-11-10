"""Web Search Agent using HTTP requests (no browser automation)

This agent can:
- Search the web using DuckDuckGo Lite (HTML-only version)
- Extract search results with titles, URLs, and snippets
- Save results to JSON files
- Load search queries from files
- No need for Playwright/browser automation (faster and more reliable)
"""

import asyncio
import json
import httpx
from pathlib import Path
from typing import List
from datetime import datetime
from bs4 import BeautifulSoup

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    snippet: str


class SearchReport(BaseModel):
    """A complete search report."""
    query: str
    timestamp: datetime
    results: List[SearchResult]
    result_count: int


class WebSearchAgent:
    """Agent for performing web searches using HTTP requests."""

    def __init__(self, headless: bool = True):
        # headless parameter kept for backward compatibility but not used
        self.client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                },
                timeout=30.0,
                follow_redirects=True
            )
        return self.client

    async def search_duckduckgo(self, query: str, max_results: int = 10) -> SearchReport:
        """
        Search DuckDuckGo and return results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            SearchReport with results
        """
        client = await self._get_client()

        try:
            # Use DuckDuckGo Lite (HTML-only version, bot-friendly)
            params = {
                'q': query,
                'kl': 'us-en',  # Region/language
            }

            response = await client.post(
                'https://lite.duckduckgo.com/lite/',
                data=params
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            results = await self._extract_results(soup, max_results)

            return SearchReport(
                query=query,
                timestamp=datetime.now(),
                results=results,
                result_count=len(results)
            )

        except Exception as e:
            print(f"Search error: {e}")
            return SearchReport(
                query=query,
                timestamp=datetime.now(),
                results=[],
                result_count=0
            )

    async def _extract_results(self, soup: BeautifulSoup, max_results: int) -> List[SearchResult]:
        """Extract search results from DuckDuckGo Lite HTML."""
        results = []

        # Find all result links (each result has a link with class='result-link')
        # Skip sponsored links by looking for organic results
        all_links = soup.find_all('a', class_='result-link')

        i = 0
        for link in all_links:
            if i >= max_results:
                break

            try:
                title = link.get_text(strip=True)
                url = link.get('href', '')

                # Skip if it's a "more info" link or empty
                if not url or not title or title == "more info":
                    continue

                # Skip DuckDuckGo redirect links (ads)
                if url.startswith('https://duckduckgo.com/y.js'):
                    continue

                # Find the corresponding snippet
                # The snippet is in a following row with class='result-snippet'
                snippet = ""
                parent_tr = link.find_parent('tr')
                if parent_tr:
                    # Look for next tr with result-snippet
                    next_trs = parent_tr.find_next_siblings('tr', limit=3)
                    for tr in next_trs:
                        snippet_td = tr.find('td', class_='result-snippet')
                        if snippet_td:
                            snippet = snippet_td.get_text(strip=True)
                            break

                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet
                ))
                i += 1

            except Exception as e:
                print(f"Error extracting result: {e}")
                continue

        return results

    def save_results(self, report: SearchReport, output_path: Path):
        """Save search results to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                report.model_dump(mode='json'),
                f,
                indent=2,
                ensure_ascii=False,
                default=str
            )

        print(f"Results saved to {output_path}")

    def load_queries(self, queries_path: Path) -> List[str]:
        """Load search queries from a text file (one per line)."""
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        return queries

    async def batch_search(self, queries: List[str], output_dir: Path, max_results: int = 10):
        """
        Perform multiple searches and save results.

        Args:
            queries: List of search queries
            output_dir: Directory to save results
            max_results: Maximum results per query
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Searching: {query}")

            try:
                report = await self.search_duckduckgo(query, max_results)

                # Create filename from query
                safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
                safe_filename = safe_filename[:50]  # Limit length
                output_path = output_dir / f"{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                self.save_results(report, output_path)
                print(f"  Found {report.result_count} results")

                # Small delay between searches
                await asyncio.sleep(2)

            except Exception as e:
                print(f"  Error searching '{query}': {e}")

        # Close client
        if self.client:
            await self.client.aclose()


async def main():
    """Example usage of WebSearchAgent."""
    agent = WebSearchAgent(headless=True)

    # Single search example
    print("=== Single Search Example ===")
    report = await agent.search_duckduckgo("Python async programming", max_results=5)

    print(f"\nQuery: {report.query}")
    print(f"Found {report.result_count} results:\n")

    for i, result in enumerate(report.results, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   {result.snippet[:100]}...")
        print()

    # Save results
    agent.save_results(report, Path("output/search_results.json"))

    # Close client
    if agent.client:
        await agent.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
