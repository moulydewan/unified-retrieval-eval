#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

echo "Evaluating all experimental results..."
echo

RESULTS_DIR="outputs/results"
EVAL_DIR="outputs/eval"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "Error: $RESULTS_DIR directory not found. Run experiments first."
  exit 1
fi

jsonl_count=$(find "$RESULTS_DIR" -name "*.jsonl" | wc -l | tr -d ' ')
echo "Found $jsonl_count result files to evaluate"

if [ "$jsonl_count" -eq 0 ]; then
  echo "No result files found. Run experiments first."
  exit 1
fi

mkdir -p "$EVAL_DIR"

#RAG


python -m src.evaluation.eval_rag \
  --inputs outputs/results \
  --out outputs/eval/rag_summary.csv \
  --detailed-out outputs/eval/rag_detailed.csv \
  --by-complexity-out outputs/eval/rag_by_complexity.csv

# Denseretrieval

python -m src.evaluation.eval_denseretrieval \
  --inputs "$RESULTS_DIR" \
  --out "$EVAL_DIR/denseretrieval_summary.csv" \
  --detailed-out "$EVAL_DIR/denseretrieval_detailed.csv" \
  --by-complexity-out "$EVAL_DIR/denseretrieval_by_complexity.csv"

# Human Proxy rag

python -m src.evaluation.eval_human_proxy_rag \
  --inputs "$RESULTS_DIR" \
  --out "$EVAL_DIR/human_proxy_rag_summary.csv" \
  --detailed-out "$EVAL_DIR/human_proxy_rag_detailed.csv" \
  --by-complexity-out "$EVAL_DIR/human_proxy_rag_by_complexity.csv"

# Hierarchy

python -m src.evaluation.eval_hierarchy \
  --inputs "$RESULTS_DIR" \
  --out "$EVAL_DIR/hierarchy_summary.csv" \
  --detailed-out "$EVAL_DIR/hierarchy_detailed.csv" \
  --by-complexity-out "$EVAL_DIR/hierarchy_by_complexity.csv"

# peer o peer

python -m src.evaluation.eval_peer \
  --inputs "$RESULTS_DIR" \
  --out "$EVAL_DIR/peer_summary.csv" \
  --detailed-out "$EVAL_DIR/peer_detailed.csv" \
  --by-complexity-out "$EVAL_DIR/peer_by_complexity.csv"

# functional

python -m src.evaluation.eval_functional \
  --inputs "$RESULTS_DIR" \
  --out "$EVAL_DIR/functional_summary.csv" \
  --detailed-out "$EVAL_DIR/functional_detailed.csv" \
  --by-complexity-out "$EVAL_DIR/functional_by_complexity.csv"

echo
echo "Done. Saved evaluation outputs under $EVAL_DIR"
