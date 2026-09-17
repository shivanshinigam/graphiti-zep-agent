"""
2_query_graph.py  —  TASK 2: Query the Temporal Knowledge Graph

TYPES OF SEARCH IN GRAPHITI:
  1. Semantic search  : "Who is the CTO?" → finds semantically related facts
  2. Entity search    : look up a specific node by name
  3. Temporal search  : "Who was CTO in 2021?" → time-bounded query

THE KEY DIFFERENCE FROM VECTOR SEARCH:
  Vector DB: returns the TEXT CHUNKS most similar to your query
  Graphiti : returns FACTS (entity + relation + time) about your query

  Vector result: "Leon Müller stepped down as CTO..."
  Graphiti result: Entity=Leon Müller, Role=CTO, ValidFrom=2018, ValidUntil=2022-03-01

RUN:
  python 2_query_graph.py
"""

import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator
from utils.graphiti_client import get_graphiti_client

load_dotenv()


def print_results(results, label: str):
    print(f"\n{'─' * 60}")
    print(f"  Query : {label}")
    print(f"  Found : {len(results)} result(s)")
    print(f"{'─' * 60}")

    if not results:
        print("  No results found.")
        return

    for i, edge in enumerate(results, 1):
        # Each result is an Edge in the knowledge graph
        # An Edge represents a FACT: subject --[relation]--> object [valid_at ... invalid_at]
        fact       = edge.fact if hasattr(edge, "fact") else str(edge)
        valid_at   = edge.valid_at.strftime("%Y-%m-%d") if hasattr(edge, "valid_at") and edge.valid_at else "unknown"
        invalid_at = edge.invalid_at.strftime("%Y-%m-%d") if hasattr(edge, "invalid_at") and edge.invalid_at else "present"

        print(f"\n  Result #{i}")
        print(f"  Fact       : {fact}")
        print(f"  Valid      : {valid_at}  →  {invalid_at}")
        if hasattr(edge, "score"):
            print(f"  Score      : {round(edge.score, 4)}")


async def run_queries():
    print("Starting Task 2: Temporal Knowledge Graph Queries")
    print("=" * 60)

    graphiti = get_graphiti_client()

    # ── Query 1: Current CTO ──────────────────────────────────────────────────
    # Graphiti automatically prioritises CURRENT (non-invalidated) facts.
    # Leon was CTO → but that fact was invalidated when Aisha took over.
    # Graphiti should return Aisha as the current CTO.
    print("\n[Query 1] Who is the current CTO of NovaTech?")
    results = await graphiti.search("Who is the CTO of NovaTech?")
    print_results(results, "Who is the current CTO of NovaTech?")

    # ── Query 2: Historical — who was CEO in 2020? ────────────────────────────
    # This demonstrates temporal reasoning: Priya was CEO from 2018.
    # Marcus became CEO in July 2024. In 2020, Priya was still CEO.
    print("\n[Query 2] Who was CEO of NovaTech in 2020?")
    results = await graphiti.search(
        "Who was the CEO of NovaTech?",
        search_filter=SearchFilters(
            valid_at=[[DateFilter(date=datetime(2020, 6, 1, tzinfo=timezone.utc), comparison_operator=ComparisonOperator.less_than_equal)]]
        ),
    )
    print_results(results, "Who was CEO of NovaTech in 2020?")

    # ── Query 3: Funding and investors ───────────────────────────────────────
    print("\n[Query 3] What funding has NovaTech received?")
    results = await graphiti.search("NovaTech funding investors")
    print_results(results, "NovaTech funding and investors")

    # ── Query 4: Company acquisitions ─────────────────────────────────────────
    print("\n[Query 4] What companies did NovaTech acquire?")
    results = await graphiti.search("NovaTech acquisition")
    print_results(results, "NovaTech acquisitions")

    # ── Query 5: Aisha's contributions ────────────────────────────────────────
    print("\n[Query 5] What has Aisha Okonkwo done at NovaTech?")
    results = await graphiti.search("Aisha Okonkwo NovaTech")
    print_results(results, "Aisha Okonkwo at NovaTech")

    await graphiti.close()

    print("\n" + "=" * 60)
    print("Task 2 Complete!")
    print("\n  Key insight: Graphiti returned FACTS with timestamps,")
    print("  not just text chunks. It knows WHEN things changed.")
    print("\n  Next step: python 3_langgraph_agent.py")


if __name__ == "__main__":
    asyncio.run(run_queries())
