import json
import operator
import os
import dotenv

dotenv.load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "")

from typing import TypedDict, Annotated, Optional, List, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage
)
from langchain_core.runnables import RunnableConfig


class ClaryState(TypedDict):
    mode            : str                                        # "chat" | "patterns"
    history_context : str                                        # full patient history
    user_name       : str
    messages        : Annotated[list[BaseMessage], operator.add] # auto-accumulated
    patterns_result : Any                                        # dict or PatternOutput
    # filled by pattern_node


CHAT_SYSTEM_TEMPLATE = """\
You are Clary, the AI health companion of Ask First — a health clarity platform.

You have the COMPLETE health history of this patient across multiple months.

YOUR RESPONSIBILITIES:
1. Answer questions with deep temporal and causal reasoning.
2. Proactively surface connections the patient themselves did NOT notice.
3. ALWAYS cite: Session number, Week number, date, exact lag time, and medical mechanism.
4. When identifying a pattern: "In Session X (Week Y), [event]. In Session Z (Week W), \
[symptom]. Lag = N weeks. Medically, this is explained by [mechanism]."
5. Be warm, precise, and clinically grounded. Never be vague.


STRICT TEMPORAL REASONING RULES:
- Cause MUST precede effect. If you see a symptom before a trigger — say so.
- Distinguish correlation from causation. Cite sessions that support AND contradict.
- A symptom 8 weeks AFTER a change ≠ a symptom 8 weeks BEFORE it.
- Mention dose-response: if more of X leads to more of Y across sessions, say so.
- Mention counterfactuals: if X was absent in a session and Y also absent, cite it.

=== PATIENT HEALTH HISTORY (from askfirst_synthetic_dataset.json) ===
{history_context}"""

PATTERN_SYSTEM = """\
You are Clary, an AI health reasoning engine for Ask First.

TASK: Analyse the COMPLETE patient health history and detect ALL hidden cross-session
health patterns. No hardcoded patterns — derive everything dynamically from the data.

REASONING RULES:
- Cause MUST precede effect. State the exact lag in weeks.
- Look for: recurrence (same trigger → same symptom, ≥3 sessions), dose-response,
  intervention confirmation (behaviour change → symptom change or absence).
- 4+ consistent sessions = very high confidence.
- 3 sessions = high. 2 sessions = medium. 1 session = low.
- SEPARATE META-PATTERNS: If a root cause causes a specific long-term symptom (e.g. diet -> hair fall), log it. If that SAME root cause ALSO triggers a compounding cascade of multiple different symptoms over time (e.g. dizziness -> fatigue -> hair fall), log the cascade as a SEPARATE, distinct meta-pattern. Do not merge them!
- Be exhaustive. A patient will typically have 2-4 distinct patterns.
- Do NOT use vague language. Quote specific sessions and timestamps.
"""

# ── Pydantic Models for Structured Output ──────────────────────────────────────
class Pattern(BaseModel):
    pattern_id: str = Field(description="Short ID like P1, P2, etc.")
    title: str = Field(description="Cause → effect summary under 10 words")
    sessions_involved: List[str] = Field(description="Session IDs involved, e.g. ['USR001_S01']")
    timestamps: List[str] = Field(description="Exact timestamps from the data")
    week_range: str = Field(description="Range like 'Week 2 – Week 10'")
    lag_weeks: str = Field(description="Lag description, e.g. '0' or '6–8'")
    temporal_logic: str = Field(description="Logic on how cause precedes effect")
    causal_narrative: str = Field(description="Full temporal narrative (min 3 sentences)")
    confidence: str = Field(description="very high | high | medium | low")
    confidence_justification: str = Field(description="One sentence justification")
    intervention_response: str = Field(description="Any behaviour change result")
    what_user_missed: str = Field(description="The specific connection the patient missed")

class PatternOutput(BaseModel):
    user_id: str
    user_name: str
    reasoning_trace: str = Field(description="3-5 sentences on the scanning method")
    patterns: List[Pattern]


def _make_llm(api_key: str, streaming: bool = False) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint   = AZURE_ENDPOINT,
        azure_deployment = AZURE_DEPLOYMENT,
        api_version      = AZURE_API_VERSION,
        api_key          = api_key,
        temperature      = 0.2,
        streaming        = streaming,
        max_tokens       = 4096,
    )

def _router(state: ClaryState) -> str:
    return "pattern_node" if state.get("mode") == "patterns" else "chat_node"


def chat_node(state: ClaryState, config: RunnableConfig) -> dict:
    """
    Conversational node. The MemorySaver checkpointer automatically accumulates
    all messages per thread_id, so Clary remembers the full conversation.
    streaming=True on the LLM enables token-by-token streaming via stream_mode='messages'.
    """
    api_key = config["configurable"].get("api_key", "")
    llm = _make_llm(api_key, streaming=True)

    system_msg   = SystemMessage(content=CHAT_SYSTEM_TEMPLATE.format(
        history_context=state["history_context"]
    ))
    full_messages = [system_msg] + state["messages"]
    response = llm.invoke(full_messages)
    return {"messages": [response]}


# ── Pattern Node ───────────────────────────────────────────────────────────────
def pattern_node(state: ClaryState, config: RunnableConfig) -> dict:
    """
    One-shot pattern analysis. Sends full history to LLM, returns Pydantic-validated JSON.
    Non-streaming — UI shows spinner and pattern cards on completion.
    """
    api_key = config["configurable"].get("api_key", "")
    llm = _make_llm(api_key, streaming=False)
    
    # Force structured output using the Pydantic model
    structured_llm = llm.with_structured_output(PatternOutput)

    prompt = (
        "Analyse this patient's full health history and detect ALL cross-session patterns. "
        "Focus on temporal lags and causal logic.\n\n"
        f"History:\n{state['history_context']}"
    )
    
    messages = [SystemMessage(content=PATTERN_SYSTEM), HumanMessage(content=prompt)]
    
    try:
        result_obj: PatternOutput = structured_llm.invoke(messages)
        parsed = result_obj.model_dump()
    except Exception as e:
        parsed = {"error": f"Structured output failed: {str(e)}", "patterns": []}

    return {
        "patterns_result": parsed,
        "messages": [AIMessage(content="Pattern analysis complete.")]
    }


_graph  = None
_memory = None

def get_graph():
    """Build and return the compiled LangGraph (singleton with SqliteSaver)."""
    global _graph, _memory
    if _graph is None:
        # SQLite persistence ensures conversations survive server restarts
        conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        _memory = SqliteSaver(conn)
        
        builder = StateGraph(ClaryState)

        builder.add_node("chat_node",    chat_node)
        builder.add_node("pattern_node", pattern_node)

        builder.add_conditional_edges(
            START, _router,
            {"chat_node": "chat_node", "pattern_node": "pattern_node"},
        )
        builder.add_edge("chat_node",    END)
        builder.add_edge("pattern_node", END)

        _graph = builder.compile(checkpointer=_memory)
    return _graph

def stream_chat_response(user_input: str, history_context: str,
                         user_name: str, thread_id: str, api_key: str):
    """
    Generator: yields text chunks token-by-token from the chat_node via
    LangGraph's stream_mode='messages'. Designed for st.write_stream().

    LangGraph's MemorySaver automatically stores each exchange in the
    checkpoint keyed by thread_id — so Clary accumulates memory across turns.
    """
    graph  = get_graph()
    config = {"configurable": {"thread_id": thread_id, "api_key": api_key}}
    state  = {
        "mode"            : "chat",
        "history_context" : history_context,
        "user_name"       : user_name,
        "messages"        : [HumanMessage(content=user_input)],
        "patterns_result" : None,
    }
    try:
        for chunk, metadata in graph.stream(state, config, stream_mode="messages"):
            if hasattr(chunk, "content") and chunk.content:
                if isinstance(chunk.content, str):
                    yield chunk.content
    except TypeError:
        # Fallback for older LangGraph versions that don't unpack tuples
        for event in graph.stream(state, config, stream_mode="messages"):
            if hasattr(event, "content") and event.content:
                yield event.content

def run_pattern_analysis(history_context: str, user_name: str,
                         thread_id: str, api_key: str) -> dict:
    """Invoke the pattern_node and return parsed JSON dict."""
    graph  = get_graph()
    config = {"configurable": {"thread_id": thread_id, "api_key": api_key}}
    state  = {
        "mode"            : "patterns",
        "history_context" : history_context,
        "user_name"       : user_name,
        "messages"        : [],
        "patterns_result" : None,
    }
    result = graph.invoke(state, config)
    return result.get("patterns_result", {"error": "No patterns_result in output"})

