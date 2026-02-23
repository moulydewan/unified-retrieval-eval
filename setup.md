# Setup Guide

## Prerequisites

- Python 3.12+
- macOS (for MPS support) or Linux/Windows
- 8GB+ RAM recommended
- Internet connection for API models

## Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd unified-retrieval-eval

# Create Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip wheel
pip install -r requirements.txt

# Verify PyTorch MPS support (macOS)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Configuration

### Models Configuration

Edit `configs/models.yaml`:

```yaml
models:
  - name: claude-4-5-sonnet
    provider: bedrock
    model_id: "arn:aws:bedrock:us-east-1:033792130535:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
  - name: claude-4-5-haiku
    provider: bedrock
    model_id: "arn:aws:bedrock:us-east-1:033792130535:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
  - name: claude-3-5-sonnet
    provider: bedrock
    model_id: "anthropic.claude-3-5-sonnet-20240620-v1:0"
  - name: claude-3-5-haiku
    provider: bedrock
    model_id: "arn:aws:bedrock:us-east-1:033792130535:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0"
```

### Dataset Configuration

Edit `configs/datasets.yaml`:

```yaml
datasets:
  trecdl_2019:
    enabled: true
    year: 2019
    mode: passage
    limit: 50
    pyserini_prebuilt_index: msmarco-v1-passage

  trecdl_2020:
    enabled: true
    year: 2020
    mode: passage
    limit: 50
    pyserini_prebuilt_index: msmarco-v1-passage

  trecdl_2021:
    enabled: true
    year: 2021
    mode: passage
    limit: 50
    pyserini_prebuilt_index: msmarco-v2-passage
```

## We recommend doing a smoke test with limit 5 first.

### 1. Full Experiment (limit 5)

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run quick test (30 items, local models only)
./scripts/run_small_smoke.sh

# Check results
ls outputs/results/
```

# Evaluate results
./scripts/evaluate_all.sh

# Generate plots
python scripts/plots.py

# Generate stats
python scripts/stats.py
```

### 3. Full Pipeline

```bash
# Run everything: experiments + evaluation + plots
./scripts/run_full_paper_pipeline.sh
```

## Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
# Verify .env file exists and has correct keys
cat .env

# Test API connection
python -c "import os; print('OpenAI key:', bool(os.getenv('OPENAI_API_KEY')))"
```
```

### Performance Tips

**1. Use Local Models for Development**
- Ollama models are faster for iteration
- No API costs
- Offline capability

**2. Start Small**
- Use `limit 5` for testing
- Test with 1-2 models first
- Use `run_small_smoke.sh` for quick validation

**3. Parallel Processing**
- The framework automatically uses available CPU cores
- For large experiments, consider running overnight

## Advanced Usage

### Custom Models

Add new models to `configs/models.yaml`:

```yaml
open:
  - name: custom-model
    provider: ollama
    model_id: your-model:latest
    max_new_tokens: 512
    temperature: 0.7
```

### Custom Datasets

Create new dataset adapters in `src/datasets/`:

```python
class CustomDatasetAdapter:
    def __init__(self):
        self.name = "custom_dataset"
    
    def load_dataset(self):
        # Load your dataset
        pass
    
    def iter_prompts(self, max_items=None):
        # Yield prompts
        pass
```

### Custom Strategies

Add new decoding strategies in `src/strategies.py`:

```python
def custom_strategy(model, prompt, **kwargs):
    # Your custom strategy implementation
    pass
```

## File Structure

```
cot-bias-amplification-llms/
├── configs/           # Configuration files
├── data/             # Dataset cache
├── outputs/          # Results and logs
├── prompts/          # Prompt templates
├── scripts/          # Utility scripts
├── src/              # Source code
├── plots/            # Generated figures
├── .env              # API keys (create this)
├── .env.example      # API keys template
└── requirements.txt  # Python dependencies
```
