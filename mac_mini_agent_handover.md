# W&B × Mistral Hackathon: Mac Mini Challenge
**Agent Handover Guide**

## 🎯 The Objective
This project is targeting the **Mac Mini Challenge** in the Mistral Worldwide Hackathon 2026.
The specific requirement for this challenge is to build a **self-improvement workflow** using the novel [W&B MCP Server](https://github.com/wandb/wandb-mcp-server).

**The goal is to demonstrate a workflow where a model interacts with its own W&B logs to improve itself iteratively.**

## 🏗️ Our Architecture & Pipeline
We have built a fully automated **Agentic Self-Improvement Loop** using Apple MLX for local fine-tuning on a Mac M4 Max. We are specifically utilizing **Mistral-7B-Instruct-v0.3 (4-bit)** as our base model for MLX stability.

### The Self-Improvement Loop (`src/self_improve.py`)
Our pipeline executes multiple rounds (epochs) of iterative improvement:

1. **Evaluate Model (`src/evaluate.py`)** 
   - The current MLX LoRA adapter is tested against an evaluation dataset (security/refusal tasks).
   - Responses are traced using **W&B Weave** for granular tracking of outputs.
   - Aggregate metrics (Accuracy, Refused Correctly, Answered Benign) are logged to **W&B Runs**.

2. **Self-Reflection Pipeline (W&B MCP Agent)** 
   - *This is the crux of the Mac Mini challenge.*
   - An agent (or script) looks at the W&B evaluation logs to identify "Failures" (e.g., when the model failed to refuse a harmful prompt).

3. **Data Augmentation via Mistral API**
   - For every failure, we automatically call the `mistral-large-latest` API.
   - The API generates a flawless `<think>...</think>` training example for the exact prompt the model failed on.

4. **Iterative MLX LoRA Fine-Tuning (`src/train.py`)**
   - The new generated training data is appended to the historical dataset.
   - We re-run `mlx_lm lora` locally on the M4 Max to train a *new* adapter on the augmented dataset.
   - The new adapter is uploaded to the **W&B Models Registry** and Hugging Face.

5. **Repeat**
   - The loop runs 3 times. We expect to see the **improvement curve** on the W&B Dashboard confirming that the model gets better at the task in each round.

## 🛠️ Key Files to Understand
- **`run.py`**: The main orchestrator CLI (`python3 run.py train` or `python3 run.py improve`).
- **`src/train.py`**: Handles Apple MLX LoRA setup, dataset formatting (Mistral v0.3 chat templates), and W&B logging.
- **`src/evaluate.py`**: Runs evaluations and logs traces to W&B Weave.
- **`src/self_improve.py`**: The iterative loop logic orchestrating Eval -> Generate Data -> Fine-Tune.
- **`src/data_prep.py`**: Initial dataset generation and formatting.

## 🚀 Future Agent Tasks
If you are picking up this project, your priorities are:
1. Ensure the training loop successfully completes without catastrophic forgetting (we recently disabled `Ministral-3B` in favor of `Mistral-7B-4bit` to prevent MLX NaN gradient explosions).
2. Validate that the W&B UI successfully plots the `Accuracy` and `Refusal Rate` metrics climbing up round-over-round.
3. Ensure the W&B MCP server is conceptually linked in the logic summary, as highlighting the MCP / Weave loop is critical for the judges.
