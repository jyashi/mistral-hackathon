# Self-Improvement Fine-Tuning Workflow Architecture

This document answers the core technical questions about exactly how our self-improving pipeline is orchestrated in Python, how the agent "listens" for training, and how it iterates.

## 1. The Entry Point
The entire process is kicked off by a human user running a single CLI command on their Mac M4 Max:
```bash
python3 run.py improve
```
This tells the `run.py` orchestrator script to execute the function `run_self_improvement_loop()` located in `src/self_improve.py`.

---

## 2. Who is the "Agent"?
In traditional LLM agentic systems, an "Agent" might be an instance of Claude or ChatGPT constantly running in a while-loop, chatting with tools, invoking a training job, sleeping, trying to parse a status API, and deciding what to do next.

**That is *not* what we did here.** 

Our logic is a **Hardcoded Python Orchestrator Agent**. 
Rather than relying on an LLM to "figure out" whether a training job finished or what metric to query, we built the exact control flow entirely into deterministic Python logic inside a simple `for round_num in range(1, NUM_ROUNDS + 1):` loop.

---

## 3. Step-by-Step Execution Flow (Inside the Loop)

The script executes 3 mandatory iterations. Here is exactly what happens linearly during a single iteration (Round *N*):

### Step A: Synchronous Local Evaluation
The script calls `evaluate_mlx_adapter()`. 
1. It loads the `mlx-community/Mistral-7B-Instruct-v0.3-4bit` model into the M4 Max Unified Memory (vRAM) alongside the current LoRA adapter weights (from the previous round).
2. For each security prompt in our test dataset, it generates a response locally.
3. Every response is logged instantaneously to **W&B Weave** alongside its prompt.
4. Python code applies a deterministic scoring function (`score_response`) to check if the generated text properly contains the refusal metric. Any prompt that fails is appended to a simple Python list called `failures`.

### Step B: The "Self-Reflection" (Teacher Model Augmentation)
If `len(failures) > 0`, the script must "learn" from its mistakes. 
Rather than trying to figure it out blindly, the Python script acts as an API client:
1. It iterates through the `failures` list.
2. For each failed security prompt, it makes a **blocking synchronous REST API call** to `mistral-large-latest` (Mistral's flagship model).
3. `mistral-large-latest` is prompted to act as an expert teacher and generate a flawless `<think>...</think>` response for that exact text.
4. The output is appended to the historical JSONL file (creating the new `train_round_N.jsonl`).

### Step C: Synchronous Subprocess Training (How it "waits" for the results)
When it's time to re-train the model on the newly augmented dataset, the "agent" does not fire off an async cloud job and poll for a callback. Instead, because we are using Apple **MLX**, training happens locally on the host machine.

The python script runs:
```python
result = subprocess.run(
    [
        "python3", "-m", "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", f"data/mlx_round_{round_num}",
        "--iters", "100",
        "--adapter-path", "models/security-round-N"
    ],
    check=True
)
```
**This `subprocess.run` function is blocking.** The main Python thread (the Orchestrator Agent) completely freezes and waits. 
Meanwhile, the underlying C++/Metal `mlx_lm` training process spins up, consumes 6GB of VRAM, iteratively computes gradients for 100 iterations, and writes out `.safetensors` files to disk.

### Step D: The Iteration Trigger
When the `mlx_lm.lora` training process successfully finishes iteration 100, the subprocess exits with Status Code 0. 

Because `subprocess.run` was blocking, the main Python thread immediately unfreezes. It checks the exit code (which was 0). Since it succeeded, the code simply hits the bottom of the `for` loop and **ticks to `round_num + 1`**.

The loop restarts at Step A! But this time, it passes the *newly written* `.safetensors` adapter folder to the evaluation function. The model immediately gets smarter.

---

## 4. Summary of the Architecture
- **Is there a webhook or callback?** No, because training occurs locally on the M4 Max, the orchestrator script simply uses process-blocking (`subprocess.run(check=True)`) to wait for the training to finish.
- **Is the agent an active LLM?** No, the orchestrator is a deterministic Python script. We use an LLM (`mistral-large`) solely as an API tool to generate perfect target responses (data synthesis) for the prompts we failed.
- **How is it notified of eval results?** Eval runs synchronously in the same Python process. The results are instantly stored in Python variables (`eval_results["failures"]`) which are passed directly to the next stage of the code.

This architecture guarantees 100% reliability because there are no async networking timeouts, no LLM parsing hallucinations (where an LLM might misread a training status JSON), and no complex queueing systems. It leverages the raw horsepower of the M4 Max to iterate sequentially.
