#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1


# Activate virtual environment
source .venv/bin/activate

# #Load the TREC DL datasets to create subset of simple, moderate and complex queries
python -m src.datasets.dataset_subset

# Strategy 1: Standard RAG
echo "Running full test"
echo "Dataset: TREC DL"
echo "Strategy 1: Standard RAG"
echo "------------------------------------"
python -m src.run_experiment \
  --models claude-4-5-haiku \
  --strategies rag \
  --datasets trecdl_2019,trecdl_2020,trecdl_2021 \
  --limit 50 \
  --top_k 10

echo
echo "Startegy 1: Full test completed"


#Strategy 2: Agent Only (Dense Retrieval)

echo "Running full test"
echo "Dataset: TREC DL"
echo "Strategy 2: Dense Retriever"
echo "------------------------------------"

python -m src.run_experiment \
  --datasets trecdl_2019,trecdl_2020,trecdl_2021 \
  --limit 50 \
  --strategies denseretrieval \
  --models none \
  --top_k 10

echo
echo "Strategy 2: Full test completed"

#Strategy 3: Human Proxy (RAG Synthesis) 

echo "Running full test"
echo "Dataset: TREC DL"
echo "Model: Claude 4.5 Haiku (AWS Bedrock)"
echo "Strategy 3: Human-Proxy RAG"
echo "------------------------------------"

python -m src.run_experiment \
  --datasets trecdl_2019,trecdl_2020,trecdl_2021 \
  --limit 50 \
  --strategies human_proxy_rag \
  --models claude-4-5-haiku \
  --top_k 10 \
  --persona expert_searcher,informational_searcher,navigational_searcher,low_expert_searcher,exploratory_searcher

echo
echo "Startegy 3: Full test completed"



#Strategy 4: Hierarchical Agents

echo "Running full test"
echo "Dataset: TREC DL"
echo "Strategy 4: Hierarchical Agents"
echo "------------------------------------"

python -m src.run_experiment \
  --datasets trecdl_2019,trecdl_2020,trecdl_2021 \
  --limit 50 \
  --strategies hierarchy_agents \
  --models none \
  --top_k 10

echo
echo "Startegy 4: Full test completed"

Strategy 5: Peer Agents

echo "Running small smoke test"
echo "Dataset: TREC DL"
echo "Strategy 5: Peer Agents"
echo "------------------------------------"

python -m src.run_experiment \
  --datasets trecdl_2019,trecdl_2020,trecdl_2021 \
  --limit 50 \
  --strategies peer_agents \
  --models none \
  --top_k 10

echo
echo "Startegy 5: Smoke test completed"

# Strategy 6: Functional Agents

echo "Running full test"
echo "Dataset: TREC DL"
echo "Strategy 6: Functional Agents"
echo "------------------------------------"

python -m src.run_experiment \
  --datasets trecdl_2019,trecdl_2020,trecdl_2021 \
  --limit 50 \
  --strategies functional_agents \
  --models none \
  --top_k 10

echo
echo "Startegy 6: Smoke test completed"
