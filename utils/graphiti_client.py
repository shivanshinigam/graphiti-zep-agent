"""
utils/graphiti_client.py

Shared Graphiti client — using Ollama as the local LLM + embedder backend.

WHAT IS OLLAMA?
  Ollama is a tool that runs open-source LLMs (Mistral, Llama, etc.) locally.
  It exposes an OpenAI-compatible API at http://localhost:11434/v1
  So any code written for OpenAI works with Ollama — just change the base_url.

HOW GRAPHITI USES THE LLM:
  When you call graphiti.add_episode("Alice joined Acme Corp as CTO today"),
  Graphiti internally calls the LLM to extract:
    - Entities : Alice (Person), Acme Corp (Organisation)
    - Relation : Alice --[JOINED_AS CTO]--> Acme Corp
    - Time     : valid_at = today's date
  These are stored as nodes + edges in Neo4j.

MODELS USED:
  LLM      : mistral (via Ollama) — entity + relationship extraction
  Embedder : nomic-embed-text (via Ollama) — semantic similarity search
  Both run 100% locally. Zero API keys required.
"""

import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_client import OpenAIClient, LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

load_dotenv()


def get_graphiti_client() -> Graphiti:
    """
    Returns a fully configured Graphiti client connected to:
      - Neo4j        (graph database, running via Docker)
      - Ollama/Mistral  (LLM for entity + relationship extraction, local)
      - Ollama/nomic-embed-text (embedder for semantic search, local)

    WHY OPENAI-COMPATIBLE CLIENT FOR OLLAMA?
      Ollama's API is 100% compatible with OpenAI's API format.
      Graphiti's OpenAIClient works with any OpenAI-compatible endpoint —
      just point base_url to Ollama instead of api.openai.com.
      No code change needed in the rest of the pipeline.
    """
    ollama_url  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434/v1")
    llm_model   = os.getenv("LLM_MODEL",        "mistral")
    embed_model = os.getenv("EMBED_MODEL",       "nomic-embed-text")
    neo4j_uri   = os.getenv("NEO4J_URI",         "bolt://localhost:7687")
    neo4j_user  = os.getenv("NEO4J_USER",        "neo4j")
    neo4j_pass  = os.getenv("NEO4J_PASSWORD",    "graphiti123")

    return Graphiti(
        uri      = neo4j_uri,
        user     = neo4j_user,
        password = neo4j_pass,

        # LLM: Mistral via Ollama (OpenAI-compatible endpoint)
        llm_client = OpenAIClient(
            config = LLMConfig(
                api_key  = "ollama",      # any non-empty string works with Ollama
                model    = llm_model,
                base_url = ollama_url,
            )
        ),

        # Embedder: nomic-embed-text via Ollama
        # nomic-embed-text produces 768-dim vectors (vs 1536 for OpenAI ada-002)
        embedder = OpenAIEmbedder(
            config = OpenAIEmbedderConfig(
                api_key       = "ollama",
                model         = embed_model,
                base_url      = ollama_url,
                embedding_dim = 768,      # nomic-embed-text output dimension
            )
        ),
    )
