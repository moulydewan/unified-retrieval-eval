#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment
source .venv/bin/activate

echo "Running small smoke test"
echo "Dataset: TREC DL 2020 (5 queries)"
echo "Model: Claude 4.5 Haiku (AWS Bedrock)"
echo "Strategy 1: RAG"
echo "------------------------------------"

# Strategy 1: Human Proxy (RAG Synthesis) 
python -m src.run_experiment \
  --datasets trecdl \
  --year 2020 \
  --limit 10 \
  --strategies rag \
  --models claude-4-5-haiku \
  --top_k 20

echo
echo "Startegy 1: Smoke test completed"

# Strategy 2: Agent Only (Dense Retrieval)
python -m src.run_experiment \
  --datasets trecdl \
  --year 2020 \
  --limit 10 \
  --strategies denseretrieval \
  --top_k 10

echo
echo "Startegy 2: Smoke test completed"

# Strategy 3: Hybrid Baseline (Standard RAG)
python -m src.run_experiment \
  --datasets trecdl \
  --year 2020 \
  --limit 10 \
  --strategies standardrag \
  --top_k 10

echo
echo "Startegy 3: Smoke test completed"
