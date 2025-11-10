"""
Smart Search Agent mit LLM-Integration (gpt-oss:20b)

Verwendung:
    # Einfache Suche (ohne LLM)
    uv run python main.py search "your query"

    # Intelligente Suche mit LLM (Query-Generierung, Bewertung, Zusammenfassung)
    uv run python main.py smart "Ich möchte lernen wie man..."

    # Schnelle Antwort auf eine Frage
    uv run python main.py ask "Was ist asyncio?"

    # Batch-Suche aus Datei
    uv run python main.py batch queries.txt

    # Demo ausführen
    uv run python main.py demo

Hinweis: Stelle sicher, dass Ollama läuft und das gpt-oss:20b Modell verfügbar ist:
    ollama pull gpt-oss:20b
    ollama serve
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

from search_agent import WebSearchAgent
from smart_search_agent import SmartSearchAgent


async def single_search(query: str):
    """Führe eine einfache Suche durch (ohne LLM)."""
    agent = WebSearchAgent(headless=True)
    print(f"Suche nach: {query}\n")

    report = await agent.search_duckduckgo(query, max_results=5)

    print(f"Gefunden: {report.result_count} Ergebnisse:\n")
    for i, result in enumerate(report.results, 1):
        print(f"{i}. {result.title}")
        print(f"   {result.url}")
        print(f"   {result.snippet[:150]}...")
        print()

    # Speichere Ergebnisse
    output_path = Path("output") / f"search_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    agent.save_results(report, output_path)


async def smart_search(user_intent: str, model: str = "gpt-oss:20b"):
    """Führe intelligente Suche mit LLM durch."""
    print(f"\n🤖 Verwende Modell: {model}")

    agent = SmartSearchAgent(llm_model=model, headless=True)

    result = await agent.smart_search(
        user_intent=user_intent,
        num_queries=3,
        results_per_query=5,
        top_k=5
    )

    # Zeige finale Zusammenfassung
    print("\n" + "="*60)
    print("📋 FINALE ZUSAMMENFASSUNG")
    print("="*60)

    print(f"\n💡 Zusammenfassung:\n{result.summary.summary}\n")

    print("🔑 Wichtige Punkte:")
    for point in result.summary.key_points:
        print(f"  • {point}")

    print(f"\n⭐ Top {len(result.top_ranked_results)} Ergebnisse:")
    for i, ranked in enumerate(result.top_ranked_results, 1):
        print(f"\n  {i}. {ranked.result.title} (Relevanz: {ranked.relevance_score:.2f})")
        print(f"     {ranked.result.url}")

    # Speichere Ergebnisse
    safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in user_intent)
    safe_filename = safe_filename[:50]
    output_path = Path("output/smart") / f"{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    agent.save_smart_result(result, output_path)


async def quick_answer(question: str, model: str = "gpt-oss:20b"):
    """Beantworte eine Frage schnell."""
    print(f"\n🤖 Verwende Modell: {model}")
    print(f"❓ Frage: {question}\n")

    agent = SmartSearchAgent(llm_model=model, headless=True)
    answer = await agent.quick_answer(question)

    print("💬 Antwort:")
    print(f"{answer}\n")


async def batch_search(queries_file: str):
    """Führe Batch-Suche aus Datei durch."""
    agent = WebSearchAgent(headless=True)
    queries_path = Path(queries_file)

    if not queries_path.exists():
        print(f"Fehler: Datei '{queries_file}' nicht gefunden")
        return

    queries = agent.load_queries(queries_path)
    print(f"Geladen: {len(queries)} Anfragen aus {queries_file}\n")

    await agent.batch_search(queries, Path("output/batch"), max_results=5)
    print("\nBatch-Suche abgeschlossen!")


async def run_demo():
    """Führe Demo-Präsentation aus."""
    print("\n" + "="*60)
    print("🚀 SMART SEARCH AGENT DEMO")
    print("="*60)

    agent = SmartSearchAgent(llm_model="gpt-oss:20b", headless=True)

    # Demo: Smart Search
    result = await agent.smart_search(
        user_intent="Was sind die besten Methoden für Web Scraping mit Python?",
        num_queries=2,
        results_per_query=5,
        top_k=3
    )

    print("\n" + "="*60)
    print("✅ DEMO ABGESCHLOSSEN")
    print("="*60)

    print(f"\n📊 Statistiken:")
    print(f"  • Anfragen generiert: {len(result.queries_used)}")
    print(f"  • Gesamt Ergebnisse: {sum(r.result_count for r in result.all_reports)}")
    print(f"  • Top Ergebnisse: {len(result.top_ranked_results)}")

    output_path = Path("output/demo_result.json")
    agent.save_smart_result(result, output_path)


def main():
    """Haupteinstiegspunkt."""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    # Optional: Modell als letztes Argument mit --model
    model = "gpt-oss:20b"
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            model = sys.argv[model_idx + 1]
            sys.argv.pop(model_idx)  # Remove --model
            sys.argv.pop(model_idx)  # Remove model name

    try:
        if command == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            asyncio.run(single_search(query))

        elif command == "smart" and len(sys.argv) > 2:
            intent = " ".join(sys.argv[2:])
            asyncio.run(smart_search(intent, model))

        elif command == "ask" and len(sys.argv) > 2:
            question = " ".join(sys.argv[2:])
            asyncio.run(quick_answer(question, model))

        elif command == "batch" and len(sys.argv) > 2:
            queries_file = sys.argv[2]
            asyncio.run(batch_search(queries_file))

        elif command == "demo":
            asyncio.run(run_demo())

        else:
            print(__doc__)

    except KeyboardInterrupt:
        print("\n\n⚠️  Unterbrochen durch Benutzer")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
