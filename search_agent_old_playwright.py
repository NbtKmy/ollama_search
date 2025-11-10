"""Web Search Agent using Playwright

This agent can:
- Search the web using DuckDuckGo
- Extract search results with titles, URLs, and snippets
- Save results to JSON files
- Load search queries from files
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from playwright.async_api import async_playwright, Page
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
    """Agent for performing web searches and handling results."""

    def __init__(self, headless: bool = True):
        self.headless = headless

    async def search_duckduckgo(self, query: str, max_results: int = 10) -> SearchReport:
        """
        Search DuckDuckGo and return results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            SearchReport with results
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=['--disable-blink-features=AutomationControlled']
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US'
            )
            page = await context.new_page()

            # Additional stealth measures
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            try:
                # Navigate to DuckDuckGo
                await page.goto("https://duckduckgo.com", wait_until="domcontentloaded")

                # Enter search query
                await page.fill('input[name="q"]', query)
                await page.press('input[name="q"]', 'Enter')

                # Wait for page to load completely
                print("Waiting for search results to load...")
                await page.wait_for_load_state("networkidle", timeout=20000)

                # Wait additional time for JavaScript to render results
                await asyncio.sleep(3)

                print("Page loaded, extracting results...")

                # Debug: save HTML and screenshot
                html = await page.content()
                with open("actual_page.html", "w", encoding="utf-8") as f:
                    f.write(html)
                await page.screenshot(path="actual_screenshot.png")
                print("Debug: HTML and screenshot saved")

                # Extract results
                results = await self._extract_results(page, max_results)

                await browser.close()

                return SearchReport(
                    query=query,
                    timestamp=datetime.now(),
                    results=results,
                    result_count=len(results)
                )
            except Exception as e:
                await browser.close()
                raise Exception(f"Search failed: {str(e)}")

    async def _extract_results(self, page: Page, max_results: int) -> List[SearchResult]:
        """Extract search results from the page."""
        results = []

        # Try multiple selectors as DuckDuckGo's structure varies
        selectors_to_try = [
            '[data-testid="result"]',
            'article[data-testid="result"]',
            'li[data-layout="organic"]',
            'article',
            '.result',
            '[data-nrn="result"]'
        ]

        result_elements = []
        for selector in selectors_to_try:
            result_elements = await page.query_selector_all(selector)
            if result_elements:
                print(f"Found {len(result_elements)} results using selector: {selector}")
                break

        if not result_elements:
            # Fallback: try to find any links in the results area
            print("Warning: Could not find results with standard selectors, trying fallback...")
            result_elements = await page.query_selector_all('#links article, #web_content_wrapper article')

        for element in result_elements[:max_results]:
            try:
                # Extract title - try multiple approaches
                title = "No title"
                title_element = await element.query_selector('h2, h3, [data-testid="result-title-a"]')
                if title_element:
                    title = await title_element.inner_text()

                # Extract URL - try multiple approaches
                url = ""
                link_element = await element.query_selector('a[data-testid="result-title-a"], h2 a, h3 a, a[href]')
                if link_element:
                    url = await link_element.get_attribute('href')

                # Extract snippet - try multiple approaches
                snippet = "No snippet"
                snippet_element = await element.query_selector('[data-result="snippet"], [data-testid="result-snippet"], div[data-result="snippet"]')
                if not snippet_element:
                    # Try to get any text content that's not the title
                    snippet_element = await element.query_selector('div:not(h2):not(h3)')
                if snippet_element:
                    snippet_text = await snippet_element.inner_text()
                    if snippet_text and snippet_text != title:
                        snippet = snippet_text

                # Only add if we have at least a URL
                if url:
                    results.append(SearchResult(
                        title=title.strip(),
                        url=url,
                        snippet=snippet.strip()
                    ))
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


if __name__ == "__main__":
    asyncio.run(main())
