#!/bin/bash
# setup_models.sh
# Run this ONCE after `docker compose up -d` to pull both models into Ollama.
# Models are cached in a Docker volume — you only download them once.

set -e

echo "Pulling phi3:mini (LLM for entity extraction)..."
echo "  Size: ~2.3GB — Microsoft's Phi-3 Mini, fast and efficient."
docker exec graphiti-ollama ollama pull phi3:mini

echo ""
echo "Pulling nomic-embed-text (embedding model)..."
echo "  Size: ~274MB"
docker exec graphiti-ollama ollama pull nomic-embed-text

echo ""
echo "All models ready. Verifying..."
docker exec graphiti-ollama ollama list

echo ""
echo "Setup complete. You can now run:"
echo "  python 1_ingest_graph.py"
