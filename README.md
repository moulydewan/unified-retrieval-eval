# Human Agent Collaboration - SIGIR Resource Paper Repository

## Title: A Reproducible Framework for Evaluating Retrieval Strategies Across IR Paradigms

## Abstract

Information retrieval has evolved from single-agent, query--document systems toward increasingly collaborative and agentic paradigms that integrate traditional retrieval, retrieval-augmented generation (RAG), and multi-agent reasoning frameworks. While these approaches differ substantially in design, their evaluation remains fragmented and often strategy-specific, making fair and reproducible comparison across paradigms challenging. In this work, we introduce a unified and extensible evaluation resource for systematically comparing diverse retrieval strategies under shared experimental setup. Our resource framework standardizes inputs, outputs, and evaluation protocols across retrieval settings, enabling reproducible comparison of behaviors and outcomes. We implement six retrieval strategies spanning dense retrieval, RAG pipelines, and multi-agent frameworks, all operating within a shared interface and evaluation pipeline. The design is plug-and-play, allowing new strategies to be integrated without modifying downstream components. We further introduce a unified evaluation approach that considers query characteristics and coordination structure, supporting both standard IR measures and strategy-agnostic metrics applicable across paradigms. This work provides a reusable and reproducible resource for benchmarking IR systems and studying how different forms of collaboration influence retrieval performance across benchmarks.

## Problem Statement

**How to evaluate various retrieval strategies ranging from single agent to multi-agent in one umified evaluation frameowrk?**

6 Retrieval Strategies 

We test 6 strategies:
1. **Strategy 1. Dense Retriever**: Pure dense retrieval followed by a reranking pipeline.
2. **Strategy 2. Standard RAG**: Standard Retrieval-Augmented Generation (RAG) with BM25.
3. **Strategy 3. Human Proxy RAG**: Persona enhanced RAG ersonas with varied search expertise and behavior personas.
4. **Strategy 4. Hierarchical Agent**: Hierarchical agents with multi-agent task decomposition.
5. **Strategy 5. Peer-to-Peer Agent**: Peer-to-peer agents 3 agents with different search type followed by a consensus mechanism.
6. **Strategy 6. Functional Agent.**: Functional agents with each agent specializing in one role.

## Methodology

### Experimental Design

**Models Tested:**
- **Open Source**: Claude 3.5 Sonnet, Claude 4.5 HaikU

**Datasets:**
- **TREC DL 2019**
- **TREC DL 201**
- **TREC DL 201s**
- **BEIR/SCIDOCS**
### Evaluation Metrics

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
