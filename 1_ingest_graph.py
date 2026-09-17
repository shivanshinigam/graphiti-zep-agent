"""
1_ingest_graph.py  —  TASK 1: Build a Temporal Knowledge Graph with Graphiti

WHAT IS AN EPISODE?
  An episode is a raw piece of information you feed into Graphiti.
  It can be:
    - EpisodeType.TEXT     : plain text (conversation, document, note)
    - EpisodeType.JSON     : structured data (API response, record)
    - EpisodeType.MESSAGE  : a chat message with role (user/assistant)

  Graphiti calls the LLM internally to:
    1. Extract entities   → "Alice", "Acme Corp", "CTO"
    2. Extract relations  → Alice --[JOINED_AS]--> CTO --[AT]--> Acme Corp
    3. Store everything   → Nodes + edges in Neo4j WITH timestamps

WHY THIS IS DIFFERENT FROM RAG:
  RAG stores text chunks → finds similar chunks by vector distance
  Graphiti stores FACTS  → finds by entity, relationship, or TIME

  RAG: "Find text similar to 'Alice's role'"
  Graphiti: "What is Alice's CURRENT role?" (knows it changed in June)

SCENARIO:
  We're building a knowledge graph about a fictional company "NovaTech".
  We'll feed in several episodes over "time" to show temporal updates.

RUN:
  python 1_ingest_graph.py
"""

import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv

from graphiti_core.nodes import EpisodeType
from utils.graphiti_client import get_graphiti_client

load_dotenv()

# ── Sample episodes — a company's story told over time ───────────────────────
#
# Each episode has:
#   name       : unique identifier for this episode
#   episode    : the raw text (what the agent "perceives")
#   source_description : context for the LLM ("this is a news article / slack message / etc")
#   reference_time : the point in time this information was TRUE
#
EPISODES = [
    {
        "name"               : "ep_001_founding",
        "episode"            : "NovaTech was founded in 2018 by Priya Sharma and Leon Müller. "
                               "Priya became CEO and Leon became CTO. "
                               "The company focuses on AI-powered logistics software.",
        "source_description" : "Company founding document",
        "reference_time"     : datetime(2018, 3, 15, tzinfo=timezone.utc),
        "type"               : EpisodeType.text,
    },
    {
        "name"               : "ep_003_cto_change",
        "episode"            : "Leon Müller stepped down as CTO in March 2022 to start his own company. "
                               "Aisha Okonkwo was appointed as the new CTO of NovaTech.",
        "source_description" : "Internal announcement",
        "reference_time"     : datetime(2022, 3, 1, tzinfo=timezone.utc),
        "type"               : EpisodeType.text,
    },
]


async def ingest_episodes():
    print("Starting Task 1: Building Temporal Knowledge Graph")
    print("=" * 60)
    print(f"Episodes to ingest : {len(EPISODES)}")
    print(f"Scenario           : NovaTech company history (2018–2024)")
    print(f"LLM                : phi3:mini via Ollama (entity + relation extraction)")
    print(f"Graph DB           : Neo4j (bolt://localhost:7687)")
    print("=" * 60)

    # Connect to Graphiti
    graphiti = get_graphiti_client()

    # Build indices on first run (idempotent — safe to call again)
    print("\nInitialising graph schema and indices...")
    await graphiti.build_indices_and_constraints()
    print("Schema ready.")

    # Ingest each episode
    for i, ep in enumerate(EPISODES, 1):
        print(f"\n[{i}/{len(EPISODES)}] Ingesting: {ep['name']}")
        print(f"  Source : {ep['source_description']}")
        print(f"  Date   : {ep['reference_time'].strftime('%B %Y')}")
        print(f"  Text   : {ep['episode'][:80]}...")

        await graphiti.add_episode(
            name             = ep["name"],
            episode_body     = ep["episode"],
            source           = ep["type"],
            source_description = ep["source_description"],
            reference_time   = ep["reference_time"],
        )
        print(f"  Graphiti extracted entities + relations and stored in Neo4j.")

    await graphiti.close()

    print("\n" + "=" * 60)
    print("Task 1 Complete!")
    print(f"  Episodes ingested : {len(EPISODES)}")
    print(f"  Time span covered : 2018 – 2024")
    print(f"  Explore the graph : http://localhost:7474")
    print(f"  Login             : neo4j / graphiti123")
    print("\n  Run MATCH (n) RETURN n LIMIT 50 in Neo4j Browser to see the graph.")
    print("\n  Next step: python 2_query_graph.py")


if __name__ == "__main__":
    asyncio.run(ingest_episodes())
