## Repository Structure

```
cot-bias-amplification-llms/
├── data/                         # Local dataset cache
├── outputs/
│   ├── raw_generations/          # JSONL files with model outputs
│   ├── eval/                     # Computed bias metrics (CSV)
│   └── logs/                     # Experiment logs
├── configs/
│   ├── models.yaml               # Model configurations
│   ├── strategies.yaml           # Decoding strategies
│   └── datasets.yaml             # Dataset settings
├── prompts/                      # Prompt templates
├── src/
│   ├── run_experiment.py         # Main experiment runner
│   ├── generation_backends/      # Model API adapters
│   ├── datasets/                 # Dataset loaders
│   ├── strategies.py             # Strategy implementations
│   ├── parsing.py                # Extract final answers
│   ├── utils.py                  # IO, seeds, logging
│   └── evaluation/               # Bias metric calculators
├── scripts/                      # Convenience scripts
├── plots/                        # Generated figures
├── README.md
├── SETUP_GUIDE.md               # Detailed setup instructions
├── .env.example
└── requirements.txt

SIGIR26-PERSPECTIVE/
│
├── .venv/
│
├── configs/
│   ├── datasets.yaml            # Dataset settings
│   ├── models.yaml              # Model configurations
│   └── strategies.yaml          # Strategy configurations (Human Proxy, Agent Only, Traditional RAG etc.)
│
├── outputs/
│   ├── logs/                    #Experiment logs
│   ├── results/                 #Result files
│   ├── bm25_retrieved_results.csv
│   └── experiment_metadata.json
│
├── prompts/                      #Prompt Templates
│   └── rag_prompt.txt
│
├── scripts/                      #Convenience 
│   └── run_small_smoke.sh
│
├── src/
│   ├── __pycache__/
│   │
│   ├── datasets/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── beir.py
│   │   ├── naturalquestion.py
│   │   └── trecdl.py
│   │
│   ├── experiments/
│   │   ├── __pycache__/
│   │   └── rag.py
│   │
│   ├── generation_backend/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── aws_bedrock_backend.py
│   │  
│   │
│   ├── __init__.py
│   ├── main.py
│   ├── run_experiment.py
│   └── utils.py
│
└── requirements.txt


```

## Setup Instructions

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).
