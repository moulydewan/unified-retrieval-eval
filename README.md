# Human Agent Collaboration - SIGIR Perspective Paper Repository

## Repository Structure

```
SIGIR26-PERSPECTIVE/
│
├── .venv/
│
├── configs/
│   ├── datasets.yaml                  # Dataset settings
│   ├── models.yaml                    # Model configurations
│   └── strategies.yaml                # Strategy configurations (Human Proxy, Agent Only, Traditional RAG etc.)
├── outputs/
│   ├── logs/                          #Experiment logs
│   ├── results/                       #Result files
├── prompts/                           #Prompt Templates
├── scripts/                           #Convenience scripts
├── src/
│   ├── main.py                        #experiment checker
│   ├── run_experiment.py              #Main experiment runner
│   |── utils.py                       #IO, seeds, logging
│   ├── datasets/                      #Dataset manager
│   ├── generation_backend/            #LLM Model API Adapters
└── requirements.txt


```

## Setup Instructions

For detailed setup instructions, see [setup.md](setup.md).
