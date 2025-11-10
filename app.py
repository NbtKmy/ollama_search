"""Gradio Web UI for Smart Search Agent

This app provides:
- Web-based UI for asking questions
- Display answers with source citations
- Visual presentation of search results
"""

import gradio as gr
import asyncio
from datetime import datetime
from pathlib import Path

from smart_search_agent import SmartSearchAgent
from search_agent import SearchReport


class SearchApp:
    """Gradio app for Smart Search Agent."""

    def __init__(self, model: str = "gpt-oss:20b"):
        self.agent = SmartSearchAgent(llm_model=model, headless=True)
        self.model = model

    def quick_answer_with_sources(self, question: str, progress=gr.Progress()):
        """
        Get a quick answer with source citations.

        Returns:
            Tuple of (answer, sources_html)
        """
        if not question.strip():
            return "Bitte geben Sie eine Frage ein.", ""

        try:
            progress(0, desc="🔍 Suchanfrage wird generiert...")

            # Generate optimized search query
            suggestions = self.agent.llm_agent.generate_queries(question, num_queries=1)
            query = suggestions.queries[0]

            progress(0.3, desc="🌐 Web-Suche läuft...")

            # Search - run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                report = loop.run_until_complete(
                    self.agent.search_agent.search_duckduckgo(query, max_results=5)
                )
            finally:
                loop.close()

            # If no results found
            if report.result_count == 0:
                return "Keine Suchergebnisse gefunden.", ""

            progress(0.7, desc="🤖 Antwort wird generiert...")

            # Summarize
            summary = self.agent.llm_agent.summarize_results(report)

            progress(0.9, desc="✨ Ergebnisse werden formatiert...")

            # Build answer text
            answer = f"**Antwort:**\n\n{summary.summary}\n\n"

            if summary.key_points:
                answer += "**Wichtige Punkte:**\n"
                for point in summary.key_points:
                    answer += f"- {point}\n"

            # Build sources HTML
            sources_html = "<div style='margin-top: 20px;'>"
            sources_html += "<h3>📚 Verwendete Quellen:</h3>"

            for i, result in enumerate(report.results, 1):
                sources_html += f"""
                <div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>
                    <div style='font-weight: bold; color: #1a73e8; margin-bottom: 5px;'>
                        {i}. <a href='{result.url}' target='_blank' style='color: #1a73e8; text-decoration: none;'>{result.title}</a>
                    </div>
                    <div style='font-size: 0.9em; color: #5f6368; margin-bottom: 5px;'>
                        🔗 {result.url}
                    </div>
                    <div style='font-size: 0.85em; color: #333;'>
                        {result.snippet}
                    </div>
                </div>
                """

            sources_html += "</div>"

            progress(1.0, desc="✅ 完了!")

            return answer, sources_html

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"エラー詳細:\n{error_detail}")
            return f"❌ Fehler: {str(e)}", f"<pre style='color: red;'>{error_detail}</pre>"

    def smart_search_full(self, user_intent: str, num_queries: int, results_per_query: int, progress=gr.Progress()):
        """
        Perform full smart search with query generation and ranking.

        Returns:
            Tuple of (answer, sources_html, queries_used)
        """
        if not user_intent.strip():
            return "Bitte geben Sie eine Suchabsicht ein.", "", ""

        try:
            progress(0, desc="🔍 Suchanfragen werden generiert...")

            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.agent.smart_search(
                        user_intent=user_intent,
                        num_queries=num_queries,
                        results_per_query=results_per_query,
                        top_k=5
                    )
                )
            finally:
                loop.close()

            progress(0.8, desc="📊 Ergebnisse werden formatiert...")

            # Build answer text
            answer = f"**Zusammenfassung:**\n\n{result.summary.summary}\n\n"

            if result.summary.key_points:
                answer += "**Wichtige Punkte:**\n"
                for point in result.summary.key_points:
                    answer += f"- {point}\n"

            # Build ranked results with sources
            sources_html = "<div style='margin-top: 20px;'>"
            sources_html += "<h3>⭐ Top Ergebnisse (nach Relevanz sortiert):</h3>"

            for i, ranked in enumerate(result.top_ranked_results, 1):
                sources_html += f"""
                <div style='margin: 10px 0; padding: 15px; border: 2px solid #4285f4; border-radius: 8px; background-color: #f8f9fa;'>
                    <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;'>
                        <div style='font-weight: bold; font-size: 1.1em; color: #1a73e8; flex: 1;'>
                            {i}. <a href='{ranked.result.url}' target='_blank' style='color: #1a73e8; text-decoration: none;'>{ranked.result.title}</a>
                        </div>
                        <div style='background-color: #4285f4; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold; margin-left: 10px;'>
                            {ranked.relevance_score:.0%}
                        </div>
                    </div>
                    <div style='font-size: 0.9em; color: #5f6368; margin-bottom: 8px;'>
                        🔗 {ranked.result.url}
                    </div>
                    <div style='font-size: 0.85em; color: #333; margin-bottom: 8px;'>
                        {ranked.result.snippet}
                    </div>
                    <div style='font-size: 0.85em; color: #555; font-style: italic; background-color: #e8f0fe; padding: 8px; border-radius: 4px;'>
                        💡 {ranked.reasoning}
                    </div>
                </div>
                """

            sources_html += "</div>"

            # Build queries used
            queries_text = "**Verwendete Suchanfragen:**\n"
            for i, query in enumerate(result.queries_used, 1):
                queries_text += f"{i}. {query}\n"

            progress(1.0, desc="✅ 完了!")

            return answer, sources_html, queries_text

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"エラー詳細:\n{error_detail}")
            return f"❌ Fehler: {str(e)}", f"<pre style='color: red;'>{error_detail}</pre>", ""


def create_ui():
    """Create Gradio UI."""
    app = SearchApp(model="gpt-oss:20b")

    with gr.Blocks(title="Smart Search Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔍 Smart Search Agent mit gpt-oss:20b

            Stellen Sie Fragen und erhalten Sie KI-gestützte Antworten mit Quellenangaben.
            """
        )

        with gr.Tabs():
            # Tab 1: Quick Answer
            with gr.Tab("💬 Schnelle Antwort"):
                gr.Markdown(
                    """
                    ### Stellen Sie eine Frage
                    Erhalten Sie eine schnelle Antwort basierend auf aktuellen Web-Suchergebnissen.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        question_input = gr.Textbox(
                            label="Ihre Frage",
                            placeholder="z.B. Was ist asyncio in Python?",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        ask_button = gr.Button("🔍 Suchen", variant="primary", size="lg")

                answer_output = gr.Markdown(label="Antwort")
                sources_output = gr.HTML(label="Quellen")

                ask_button.click(
                    fn=app.quick_answer_with_sources,
                    inputs=[question_input],
                    outputs=[answer_output, sources_output]
                )

                # Examples
                gr.Examples(
                    examples=[
                        ["Was ist asyncio in Python?"],
                        ["Wie funktioniert Machine Learning?"],
                        ["Was sind die Vorteile von TypeScript?"],
                        ["Erkläre mir Docker in einfachen Worten"],
                    ],
                    inputs=[question_input]
                )

            # Tab 2: Advanced Search
            with gr.Tab("🚀 Erweiterte Suche"):
                gr.Markdown(
                    """
                    ### Intelligente Suche mit Query-Generierung
                    Nutzen Sie KI-generierte Suchanfragen und Relevanz-Ranking für bessere Ergebnisse.
                    """
                )

                intent_input = gr.Textbox(
                    label="Ihre Suchabsicht",
                    placeholder="z.B. Ich möchte lernen, wie man Web-Scraper mit Python erstellt",
                    lines=3
                )

                with gr.Row():
                    num_queries_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        label="Anzahl Suchanfragen"
                    )
                    results_per_query_slider = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Ergebnisse pro Anfrage"
                    )

                search_button = gr.Button("🔍 Intelligente Suche", variant="primary", size="lg")

                queries_output = gr.Markdown(label="Generierte Suchanfragen")
                advanced_answer_output = gr.Markdown(label="Zusammenfassung")
                advanced_sources_output = gr.HTML(label="Top Ergebnisse")

                search_button.click(
                    fn=app.smart_search_full,
                    inputs=[intent_input, num_queries_slider, results_per_query_slider],
                    outputs=[advanced_answer_output, advanced_sources_output, queries_output]
                )

                # Examples
                gr.Examples(
                    examples=[
                        ["Wie erstelle ich einen Web-Scraper mit Python?"],
                        ["Was sind die besten Praktiken für React Hooks?"],
                        ["Erkläre mir Kubernetes und Container-Orchestrierung"],
                    ],
                    inputs=[intent_input]
                )

            # Tab 3: Info
            with gr.Tab("ℹ️ Info"):
                gr.Markdown(
                    """
                    ## Über diese App

                    Diese App kombiniert:
                    - **Web-Suche**: DuckDuckGo Lite API für aktuelle Informationen
                    - **KI-Modell**: gpt-oss:20b via Ollama für intelligente Verarbeitung
                    - **Smart Features**:
                        - Automatische Query-Generierung
                        - Relevanz-Bewertung der Ergebnisse
                        - KI-gestützte Zusammenfassungen
                        - Quellenangaben mit Links

                    ### Verwendung

                    **Schnelle Antwort:**
                    - Ideal für direkte Fragen
                    - Schneller und einfacher
                    - Zeigt Top-5 Quellen

                    **Erweiterte Suche:**
                    - Nutzt mehrere optimierte Suchanfragen
                    - Bewertet und rankt Ergebnisse nach Relevanz
                    - Detailliertere Informationen

                    ### Technologie-Stack
                    - Python 3.11+
                    - Gradio für UI
                    - httpx + BeautifulSoup4 für Web-Suche
                    - Ollama für LLM
                    - Pydantic für Datenvalidierung
                    """
                )

        gr.Markdown(
            """
            ---
            💡 **Tipp**: Die Qualität der Antworten hängt von den Suchergebnissen ab.
            Formulieren Sie Ihre Fragen klar und spezifisch.
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
