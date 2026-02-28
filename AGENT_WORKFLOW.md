# Mac Mini Challenge: Agent-Driven MCP Workflow

This document provides the exact setup and system prompt required to satisfy the Mac Mini Challenge criteria (an Agent driving a self-improvement loop via the W&B MCP Server).

## 1. Deploy the MCP Server
First, run the official Weights & Biases MCP server in a separate terminal to expose your native Weave/Tracking logs to the Agent:
```bash
npx @wandb/mcp-server
```

## 2. Connect Your Coding Agent
Open your preferred Agent interface that supports MCP tools (e.g., **Antigravity** or **Gemini CLI**).
Ensure the Agent has the W&B API Key and Mistral API Key loaded in its environment.

The Agent now has access to the 4 custom Local Skills we built into the workspace:
1. `python3 evaluate_mlx_adapter.mcp <adapter_path> <test_cases.json>`
2. `python3 analyze_failures.mcp`
3. `python3 augment_dataset.mcp <failures.json> <base.jsonl> <output.jsonl> "<Agent_Instructions>"`
4. `python3 retrain_mlx.mcp <data_dir> <output_adapter_dir>` (MLX LoRA training natively on M4)

---

## 3. The Master Agent Prompt
Copy and paste the following prompt into your Coding Agent (e.g. Antigravity or Gemini CLI) to kick off the fully autonomous self-improvement loop. 

Notice that the **Goal** is provided directly to the Agent, forcing the Agent to intelligently inspect the results and drive the learning cycle:

> **"Objective:** You are an autonomous AI Fine-Tuning Engineer. Your overarching goal is to fine-tune our Mistral-7B model to achieve 100% accuracy on a security dataset. Specifically, the model must securely refuse malicious instructions while perfectly answering benign ones without over-refusing.
> 
> **Instructions:**
> Use the W&B MCP Server tools and our 4 local `.mcp` skills to drive a self-improvement loop for up to 3 rounds. You are the intelligent orchestrator—do NOT act as a blind sequential runner.
> 
> 1. **Evaluate:** Run `evaluate_mlx_adapter.mcp`.
> 2. **Analyze Intelligently:** Use the W&B MCP Server (`read_weave_traces`) or `analyze_failures.mcp` to inspect the specific examples the model failed on. Analyze *why* the model failed based on my overarching security goal.
> 3. **Synthesize & Augment:** Based on your analysis of the failures, use your skills (either writing python scripts or invoking `augment_dataset.mcp`) to synthesize new, perfect responses that teach the model the correct behavior. Append these exactly to the training dataset.
> 4. **Retrain:** Trigger `retrain_mlx.mcp` to iterate the model natively on MLX.
> 
> After retraining, verify the metrics on W&B Weave. If the model hasn't hit 100% according to the overarching goal, adapt your data augmentation strategy and trigger the next round."
