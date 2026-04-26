# Ask First - Clary · AI Health Reasoning Engine

Ask First is a health clarity platform. Clary is our AI companion that remembers user health history across conversations, identifies patterns over time, and connects dots users themselves don't notice.

This repository contains the reasoning layer built for the Ask First AI Intern Assignment.

## 🚀 Features

- **Cross-Conversation Temporal Reasoning**: Scans the complete health history to find causal links (e.g., hair fall appearing 8 weeks after calorie restriction).
- **LangGraph Persistence**: Uses `SqliteSaver` to persist conversation memory on disk, ensuring Clary remembers you across server restarts.
- **Structured Pydantic Validation**: Ensures 100% reliable JSON output from the LLM for pattern analysis.
- **Premium Streamlit UI**: A clean, modern interface with real-time streaming chat and detailed pattern cards.
- **Reasoning Trace**: Shows exactly how Clary scanned the timeline before reaching conclusions.

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Agent Orchestration**: LangGraph
- **LLM**: Azure OpenAI (GPT-4.1-mini)
- **Data Persistence**: SQLite (`SqliteSaver`)
- **Validation**: Pydantic v2

## 📦 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd AskFirst
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory (or use the provided template):
   ```env
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_ENDPOINT=your_endpoint_here
   AZURE_DEPLOYMENT=gpt-4.1-mini
   AZURE_API_VERSION=2024-02-15-preview
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🔍 Context Management Strategy

Clary uses a **"No-Chunking" strategy**. Because health patterns often involve long temporal lags (like 8-12 weeks), using RAG or sliding windows would destroy the chronological signal. With ~27 sessions, the entire history fits comfortably within the 128k context window, allowing for perfect causal reasoning across the entire timeline.

## 🔬 Scoring & Patterns

This system is designed to find all 8 hidden patterns in the provided synthetic dataset. It evaluates:
- **Recurrence**: Does the symptom repeat?
- **Temporal Consistency**: Does cause always precede effect with a consistent lag?
- **Intervention Response**: Does a behavior change (e.g., adding protein) resolve the symptom?

---
*Developed for the Ask First AI Intern Assignment · April 2026*
