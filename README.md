# MADRA: Multi-Agent Deep Research Architecture 🧠

MADRA is a highly advanced, autonomous multi-agent research system built with **LangGraph** and **FastAPI**. It leverages the speed of Groq LLMs to perform parallel web scraping, critical analysis, and self-refining report generation.

![MADRA UI Preview](https://img.shields.io/badge/UI-Loquix%20Web%20Components-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange?style=for-the-badge)

## ✨ Key Features
- **Concurrent Web Research**: The `Researcher` agent uses `asyncio` to scrape multiple URLs and extract facts in parallel, bypassing the slowness of sequential scraping.
- **Actor-Critic Refinement Loop**: An internal `Critic` agent acts as an expert editor, automatically reviewing draft reports for factual consistency and clarity, and forcing a rewrite before the final output.
- **Academic Citation System**: MADRA automatically injects inline markdown footnotes (e.g. `[1]`) into factual claims and generates a comprehensive `## References` section.
- **Real-Time Streaming UI**: Built with FastAPI Server-Sent Events (SSE) and **Loquix Web Components**, the frontend displays exactly which agents are working and how long they take in a beautiful, expanding dropdown.
- **Smart Caching**: Implements `diskcache` to store previously extracted web claims locally, saving API tokens and massively speeding up repeated queries.

## 🏗️ Agent Architecture
1. **Planner Node**: Breaks down complex user queries into smaller, highly specific sub-questions.
2. **Researcher Node**: Searches DuckDuckGo, scrapes HTML, and uses LLMs to extract verified factual claims.
3. **Reflector Node**: Grades the quality of the research (0-10). If the score is low, it loops back to the Planner.
4. **Writer Node**: Drafts a comprehensive report using only the verified evidence, enforcing inline citations.
5. **Critic Node**: Reviews the Writer's draft for redundancy and clarity, and sends it back for one final polish.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mansi091/MADRA.git
   cd MADRA
   ```

2. **Install dependencies:**
   We recommend using `uv` for lightning-fast package management:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set your Environment Variables:**
   You must set your Groq API key in your environment or a `.env` file.
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

4. **Run the Application:**
   Start the FastAPI server:
   ```bash
   uv run uvicorn app:app --reload
   ```
   Then, open `http://127.0.0.1:8000` in your web browser to access the beautiful chat interface.

## 🛡️ Rate Limits & Customization
MADRA is currently optimized for free-tier LLM providers. In `nodes/researcher.py`, the async scraper utilizes `time.sleep()` to prevent throwing HTTP 429 Rate Limit errors. If you are using a paid API tier, you can safely remove these delays for near-instant research!
