# 🏆 Mistral × W&B Hackathon — Fine-Tuning Track

A complete fine-tuning pipeline for the **Mistral Worldwide Hackathon 2026**, built with **Weights & Biases** for maximum points.

## 🏗️ Architecture

```
run.py (CLI orchestrator)
├── setup    → Verify W&B + Mistral connectivity
├── data     → Prepare training data (built-in + synthetic)
├── train    → Fine-tune via W&B Serverless SFT
├── eval     → Compare base vs fine-tuned (Weave-traced)
├── improve  → Self-improvement loop (Mac Mini 🔄)
└── all      → Run everything end-to-end
```

## 🚀 Quick Start

### 1. Setup
```bash
# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   WANDB_API_KEY   → https://wandb.ai/authorize
#   MISTRAL_API_KEY → https://console.mistral.ai/api-keys

# Verify setup
python run.py setup
```

### 2. Prepare Data
```bash
# Use built-in examples only
python run.py data

# Or generate synthetic training data too
python run.py data --synthetic 50
```

### 3. Fine-Tune
```bash
python run.py train
```

### 4. Evaluate
```bash
# Evaluate base model
python run.py eval

# Evaluate fine-tuned model
python run.py eval --model ft:mistral-small:your-job-id
```

### 5. Self-Improvement Loop (Mac Mini Challenge 🔄)
```bash
python run.py improve
```

### 6. Run Everything
```bash
python run.py all --synthetic 50
```

## 📊 W&B Integration Points (E2E Scoring)

| Tool | Purpose | Points |
|------|---------|--------|
| **W&B Training** | Serverless SFT fine-tuning | ⭐⭐⭐⭐ |
| **W&B Models** | Registry for versioned model artifacts | ⭐⭐ |
| **W&B Weave** | Tracing & evaluation of model outputs | ⭐⭐ |
| **W&B MCP Server** | Self-improvement loop | 🔄 Mac Mini |

## 📁 Project Structure

```
├── run.py              # Main CLI entry point
├── src/
│   ├── config.py       # W&B, Weave, Mistral setup
│   ├── data_prep.py    # Training data preparation
│   ├── train.py        # Fine-tuning pipeline
│   ├── evaluate.py     # Weave-traced evaluation
│   └── self_improve.py # Self-improvement loop
├── data/               # Training & eval datasets
├── requirements.txt
└── .env.example
```
