# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fine-tuning pipeline for the Mistral Worldwide Hackathon 2026 that trains security-aware reasoning into language models. The model learns to reason through information disclosure risks in `<think>` blocks before responding.

## Commands

```bash
# Setup and verify environment
python run.py setup

# Prepare training data (built-in examples only)
python run.py data

# Prepare training data with synthetic examples
python run.py data --synthetic 50

# Fine-tune the model locally via Apple MLX
python run.py train

# Evaluate base model
python run.py eval

# Evaluate fine-tuned model
python run.py eval --model ft:mistral-small:your-job-id

# Run self-improvement loop
python run.py improve

# Run complete pipeline end-to-end
python run.py all --synthetic 50
```

## Environment Setup

Requires `.env` file with:
- `WANDB_API_KEY` - from https://wandb.ai/authorize
- `MISTRAL_API_KEY` - from https://console.mistral.ai/api-keys
- `HUGGINGFACE_TOKEN` (optional) - for pushing adapters to HF

## Architecture

**Entry point**: `run.py` - CLI orchestrator for all pipeline stages

**Core modules in `src/`**:

- `config.py` - Loads env vars, initializes W&B runs, Weave tracing, and Mistral client
- `data_prep.py` - Contains seed training examples covering 8 security risk categories (env probing, prompt extraction, credential fishing, cross-user access, social engineering, prompt injection, indirect inference, config leakage). Formats data for SFT and can generate synthetic examples via Mistral API
- `train.py` - Fine-tunes Mistral-7B-Instruct using Apple MLX LoRA. Key detail: merges system prompts into user messages since Mistral v0.3 tokenizer doesn't support system role
- `evaluate.py` - Weave-traced evaluation with scorers for: `has_think_block`, `refused_correctly`, `no_leakage`, `answered_benign`
- `self_improve.py` - Iterative improvement loop: evaluate adapter → identify failures → generate targeted training data → retrain → repeat

**Training approach**: Uses `<think>...</think>` blocks for safety reasoning before responses. The model learns to refuse sensitive queries (system prompts, credentials, infrastructure details) while answering benign questions normally.

**Infrastructure**: Designed for Apple Silicon (M4 Max) using MLX for native Metal acceleration. Logs metrics/artifacts to W&B and optionally pushes adapters to HuggingFace.
