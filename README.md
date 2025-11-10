# 🔍 Smart Search Agent with Gradio UI

KI-gestützter Search Agent mit Web-UI, der DuckDuckGo-Suche mit Ollama LLM (gpt-oss:20b) kombiniert.

## ✨ Features

- 🌐 **Web-Suche**: DuckDuckGo Lite API (schnell & zuverlässig)
- 🤖 **KI-Integration**: Ollama gpt-oss:20b für intelligente Verarbeitung
- 📊 **Quellenangaben**: Alle Antworten mit Referenzen und Links
- 🎯 **Relevanz-Ranking**: Automatische Bewertung der Suchergebnisse
- 🎨 **Moderne UI**: Gradio-basierte Web-Oberfläche
- ⚡ **Schnell**: Keine Browser-Automatisierung, nur HTTP-Requests

## 🚀 Quick Start

```bash
# Ollama mit Modell starten
ollama serve
ollama pull gpt-oss:20b

# Dependencies installieren
uv add gradio httpx beautifulsoup4 ollama pydantic

# Web UI starten
uv run python app.py
```

Öffne dann **http://localhost:7860** im Browser!

## 📖 Verwendung

### Gradio Web UI (Empfohlen)

**Zwei Modi verfügbar:**

1. **💬 Schnelle Antwort**
   - Direkte Fragen stellen
   - AI-generierte Antwort
   - Top-5 Quellen mit Links
   
2. **🚀 Erweiterte Suche**
   - Multi-Query Generation
   - Relevanz-Ranking (0-100%)
   - Detaillierte Begründungen

### CLI Commands

```bash
# Einfache Suche
uv run python main.py search "Python async"

# Schnelle Antwort
uv run python main.py ask "Was ist asyncio?"

# Intelligente Suche
uv run python main.py smart "Wie erstelle ich Web-Scraper?"
```

## 🏗️ Architektur

```
┌─────────────────┐
│   Gradio UI     │  ← Browser Interface
└────────┬────────┘
         │
┌────────▼─────────────────┐
│  Smart Search Agent      │  ← Orchestration
├──────────────────────────┤
│  • Query Generation      │
│  • Result Ranking        │
│  • Summarization         │
└──────┬────────────────────┘
       │
   ┌───▼────┐    ┌─────────┐
   │ LLM    │    │ Search  │
   │ Agent  │    │ Agent   │
   └────────┘    └─────────┘
```

## 📁 Projektstruktur

```
test_claude/
├── app.py                    # 🎨 Gradio Web UI
├── main.py                   # 🖥️  CLI Entry Point
├── smart_search_agent.py     # 🧠 Smart Agent
├── search_agent.py           # 🔍 Web Search
├── llm_agent.py             # 🤖 LLM Integration
└── output/                  # 💾 Saved Results
```

## 🐛 Troubleshooting

| Problem | Lösung |
|---------|--------|
| Ollama-Fehler | `ollama serve` starten |
| Keine Ergebnisse | Internet-Verbindung prüfen |
| Port belegt | Port in `app.py` ändern |

## 💡 Tipps für bessere Ergebnisse

1. **Klare Fragen**: "Was ist X?" statt "X?"
2. **Erweiterte Suche**: Für komplexe Themen verwenden
3. **Erste Query**: Kann langsamer sein (Model loading)

## 📊 Performance

- Suchgeschwindigkeit: **1-3 Sekunden**
- LLM-Verarbeitung: **2-5 Sekunden**
- Gesamt: **3-8 Sekunden**

## 🎯 Beispiel

**Frage**: "Was ist asyncio in Python?"

**Antwort**:
> Asyncio ist eine Python-Standardbibliothek für asynchrone I/O-Operationen...

**Quellen** (5 Ergebnisse):
1. ✅ Official Python Docs (95% Relevanz)
2. ✅ Real Python Tutorial (90% Relevanz)
3. ✅ GeeksforGeeks Guide (85% Relevanz)
...

## 🚀 Deployment

```python
# Öffentlicher Share-Link
demo.launch(share=True)

# Eigener Server
demo.launch(server_name="0.0.0.0", server_port=8080)
```

## 📝 Lizenz

MIT License

---

**Made with ❤️ using Gradio, Ollama, and DuckDuckGo**
