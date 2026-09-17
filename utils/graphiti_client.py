"""
utils/graphiti_client.py

Shared Graphiti client setup — used by all three task scripts.

WHAT IS GRAPHITI?
  Graphiti is an open-source framework for building TEMPORAL KNOWLEDGE GRAPHS.
  Unlike a vector DB (which stores chunks of text), Graphiti stores:
    - Entities  : people, places, organisations, concepts
    - Relations : edges between entities ("Alice WORKS_AT Company X")
    - Episodes  : the raw events/messages that generated those facts
    - Time      : every fact knows WHEN it became true and WHEN it changed

  This allows an AI agent to answer questions like:
    "What did Alice's role change to in June?"
    "Which company was Bob working at last year?"
    "What is the CURRENT status of project X?"
"""

import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

load_dotenv()


def get_graphiti_client() -> Graphiti:
    """
    Returns a fully configured Graphiti client connected to:
      - Neo4j  (graph database backend, running via Docker)
      - Gemini (LLM for entity + relationship extraction)
      - Gemini (embedder for semantic search)
      - Gemini (reranker for result quality)

    HOW GRAPHITI USES THE LLM:
      When you call graphiti.add_episode("Alice joined Acme Corp as CTO today"),
      Graphiti internally calls the LLM to extract:
        - Entities : Alice (Person), Acme Corp (Organisation)
        - Relation : Alice --[JOINED_AS CTO]--> Acme Corp
        - Time     : valid_at = today's date
      These are then stored as nodes + edges in Neo4j.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your-gemini-api-key-here":
        raise ValueError(
            "GEMINI_API_KEY not set!\n"
            "Get a free key at: https://aistudio.google.com/apikey\n"
            "Then add it to your .env file."
        )

    neo4j_uri  = os.getenv("NEO4J_URI",     "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER",    "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD","graphiti123")

    return Graphiti(
        uri      = neo4j_uri,
        user     = neo4j_user,
        password = neo4j_pass,

        # LLM for entity/relationship extraction from raw text
        llm_client = GeminiClient(
            api_key = api_key,
            config  = LLMConfig(model="gemini-2.0-flash"),
        ),

        # Embedding model for semantic similarity search
        embedder = GeminiEmbedder(
            api_key = api_key,
            config  = GeminiEmbedderConfig(model="text-embedding-004"),
        ),

        # Reranker to improve result quality after retrieval
        reranker = GeminiRerankerClient(api_key=api_key),
    )
