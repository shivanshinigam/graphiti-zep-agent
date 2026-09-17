"""
3_langgraph_agent.py  —  TASK 3: LangGraph Agent with Graphiti Memory

WHAT THIS DOES:
  Builds a LangGraph agent that uses Graphiti as its LONG-TERM MEMORY.

  Unlike a stateless LLM call, this agent:
    1. Stores every conversation turn into the knowledge graph
    2. Retrieves relevant facts from the graph before answering
    3. Answers with temporally-aware context ("as of March 2022...")
    4. Remembers facts ACROSS sessions (persistent graph memory)

THE GRAPH:

  [START]
     |
     v
  [retrieve_context_node]   ← search Graphiti for relevant facts
     |
     v
  [generate_node]           ← LLM answers using graph facts as context
     |
     v
  [save_to_graph_node]      ← persist this conversation turn to Graphiti
     |
     v
  [END]

STATE:
  class AgentState(TypedDict):
    messages    : list of conversation messages (Human/AI)
    query       : current user question
    graph_context: facts retrieved from Graphiti
    answer      : LLM's response

ZEP CLOUD vs GRAPHITI (SELF-HOSTED):
  Both are shown in this file:
    - Section A: Self-hosted Graphiti + Neo4j (what we set up today)
    - Section B: Zep Cloud (add ZEP_API_KEY to .env for this section)

RUN:
  python 3_langgraph_agent.py "Who is the current CTO?"
  python 3_langgraph_agent.py "What happened to NovaTech in 2022?"
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import TypedDict, Annotated
from dotenv import load_dotenv

import google.generativeai as genai
from langgraph.graph import StateGraph, START, END

from utils.graphiti_client import get_graphiti_client

load_dotenv()

# Configure Gemini for the generate node
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
llm = genai.GenerativeModel("gemini-2.0-flash")


# ══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Shared state flowing through all nodes in the agent graph.

    messages      : full conversation history (Human + AI turns)
    query         : the current question being processed
    graph_context : facts retrieved from Graphiti for this query
    answer        : the LLM's final answer
    """
    query         : str
    graph_context : list[str]   # list of fact strings from Graphiti
    answer        : str
    session_id    : str         # identifies this conversation session


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: RETRIEVE CONTEXT FROM GRAPHITI
# ══════════════════════════════════════════════════════════════════════════════

async def retrieve_context_node(state: AgentState) -> dict:
    """
    Node 1: Search the Graphiti knowledge graph for facts relevant to the query.

    - Reads  : state["query"]
    - Writes : state["graph_context"]

    Graphiti's search returns EDGES (facts with timestamps), not text chunks.
    We format them as "Fact [valid: date → date]" strings for the LLM.

    The key advantage: if the query asks about CTO, Graphiti will return
    the CURRENT CTO fact (Aisha), not Leon — because Leon's fact has been
    temporally invalidated in the graph.
    """
    print(f"\n[Node 1: RETRIEVE FROM GRAPHITI]")
    print(f"  Query: {state['query']}")

    graphiti = get_graphiti_client()

    results = await graphiti.search(
        query          = state["query"],
        reference_time = datetime.now(timezone.utc),   # "as of right now"
        num_results    = 5,
    )

    await graphiti.close()

    # Format facts for LLM context
    facts = []
    for edge in results:
        fact       = edge.fact if hasattr(edge, "fact") else str(edge)
        valid_at   = edge.valid_at.strftime("%Y-%m-%d") if hasattr(edge, "valid_at") and edge.valid_at else "unknown"
        invalid_at = edge.invalid_at.strftime("%Y-%m-%d") if hasattr(edge, "invalid_at") and edge.invalid_at else "present"
        facts.append(f"- {fact}  [valid: {valid_at} → {invalid_at}]")

    print(f"  Retrieved {len(facts)} facts from Graphiti:")
    for f in facts:
        print(f"    {f}")

    return {"graph_context": facts}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: GENERATE ANSWER
# ══════════════════════════════════════════════════════════════════════════════

def generate_node(state: AgentState) -> dict:
    """
    Node 2: Generate an answer using Graphiti facts as grounded context.

    - Reads  : state["query"], state["graph_context"]
    - Writes : state["answer"]

    The prompt explicitly tells the LLM to use the temporal facts provided,
    and to acknowledge when facts are historical vs current.

    COMPARISON WITH RAG PIPELINE:
      RAG (Task 3 from yesterday): passes raw text chunks to LLM
      Graphiti agent             : passes structured FACTS with timestamps
        → LLM is less likely to hallucinate because facts are explicit
        → LLM can reason about time: "as of July 2024, Marcus is CEO"
    """
    print(f"\n[Node 2: GENERATE ANSWER]")
    print(f"  Context facts: {len(state['graph_context'])}")

    if not state["graph_context"]:
        answer = "I don't have enough information in my knowledge graph to answer this question."
        print(f"  No context available — returning fallback.")
        return {"answer": answer}

    context_str = "\n".join(state["graph_context"])

    prompt = f"""You are an assistant with access to a structured knowledge graph about NovaTech company.

The following FACTS have been retrieved from the knowledge graph. Each fact includes the time period it was valid.
Facts marked "→ present" are currently true. Facts with a specific end date are historical.

FACTS:
{context_str}

Based ONLY on the above facts, answer the following question. 
If a fact is historical, clearly say when it was true.
If a fact is current, state it as a current truth.

Question: {state['query']}

Answer:"""

    print(f"  Calling Gemini 2.0 Flash...")
    response = llm.generate_content(prompt)
    answer   = response.text.strip()

    return {"answer": answer}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: SAVE THIS TURN TO GRAPHITI
# ══════════════════════════════════════════════════════════════════════════════

async def save_to_graph_node(state: AgentState) -> dict:
    """
    Node 3: Persist this conversation turn into the Graphiti knowledge graph.

    - Reads  : state["query"], state["answer"], state["session_id"]
    - Writes : nothing (side effect only)

    WHY SAVE CONVERSATIONS?
      Every Q&A turn may contain new information that future queries should know.
      Example: If the user says "Actually, Marcus resigned last week",
               Graphiti extracts that as a new fact and temporally invalidates
               the old "Marcus is CEO" fact.

      This is the PERCEPTION layer — the agent perceives new information
      and updates its world model in real-time.
    """
    print(f"\n[Node 3: SAVE TURN TO GRAPHITI]")

    graphiti = get_graphiti_client()

    # Save as a MESSAGE episode (preserves the Q/A structure)
    from graphiti_core.nodes import EpisodeType
    conversation_text = f"User asked: {state['query']}\nAgent answered: {state['answer']}"

    await graphiti.add_episode(
        name               = f"session_{state['session_id']}_turn_{int(datetime.now().timestamp())}",
        episode_body       = conversation_text,
        source             = EpisodeType.text,
        source_description = f"Conversation session {state['session_id']}",
        reference_time     = datetime.now(timezone.utc),
    )

    await graphiti.close()
    print(f"  Turn saved to Graphiti — agent will remember this in future sessions.")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_agent_graph():
    """
    Build and compile the LangGraph agent with Graphiti memory.

    The graph is async-compatible — LangGraph handles async nodes natively.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate",         generate_node)
    graph.add_node("save_to_graph",    save_to_graph_node)

    graph.add_edge(START,              "retrieve_context")
    graph.add_edge("retrieve_context", "generate")
    graph.add_edge("generate",         "save_to_graph")
    graph.add_edge("save_to_graph",    END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: ZEP CLOUD (optional — needs ZEP_API_KEY)
# ══════════════════════════════════════════════════════════════════════════════

def show_zep_cloud_comparison():
    """
    Explains how Zep Cloud differs from self-hosted Graphiti.
    Runs even without a Zep API key — educational only.
    """
    print("\n" + "=" * 60)
    print("  ZEP CLOUD vs SELF-HOSTED GRAPHITI")
    print("=" * 60)
    print("""
  Self-hosted Graphiti (what we built today):
    + Free, open source
    + Full control over data
    + Neo4j runs on your machine (or your server)
    - You manage infrastructure
    - You handle scaling

  Zep Cloud (managed platform):
    + Sub-200ms latency at global scale
    + No infrastructure to manage
    + Enterprise SLA + data governance
    + BYOM (Bring Your Own LLM model)
    + Built-in user/session management
    - Requires API key + paid plan (free tier available)

  Code difference — almost nothing:

    # Self-hosted Graphiti
    from graphiti_core import Graphiti
    client = Graphiti("bolt://localhost:7687", "neo4j", "password")

    # Zep Cloud
    from zep_cloud.client import AsyncZep
    client = AsyncZep(api_key=ZEP_API_KEY)
    # Same search(), add_episode() API — just different backend

  When to use which:
    Learning / dev / small scale  →  Self-hosted Graphiti
    Production / enterprise       →  Zep Cloud
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    query      = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Who is the current CTO of NovaTech?"
    session_id = "session_001"

    print("=" * 60)
    print("  LangGraph Agent with Graphiti Memory")
    print("  START → retrieve_context → generate → save_to_graph → END")
    print("=" * 60)
    print(f"\n  Query      : {query}")
    print(f"  Session    : {session_id}")
    print(f"  Memory     : Graphiti (Neo4j @ bolt://localhost:7687)")
    print(f"  LLM        : Gemini 2.0 Flash")

    app = build_agent_graph()

    initial_state: AgentState = {
        "query"         : query,
        "graph_context" : [],
        "answer"        : "",
        "session_id"    : session_id,
    }

    final_state = await app.ainvoke(initial_state)

    print("\n" + "=" * 60)
    print("  FINAL ANSWER")
    print("=" * 60)
    print(f"\n  Query : {final_state['query']}")
    print(f"\n  Answer:\n")
    for line in final_state["answer"].split("\n"):
        print(f"    {line}")

    print("\n" + "=" * 60)
    print(f"  Graph context used : {len(final_state['graph_context'])} facts")
    print(f"  Turn saved to graph: yes — agent will remember this")
    print("=" * 60)

    # Show Zep Cloud comparison
    show_zep_cloud_comparison()


if __name__ == "__main__":
    asyncio.run(main())
