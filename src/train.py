"""
Fine-tuning Ministral-3B using Apple MLX (mlx-lm) — runs natively on M4 Max.

MLX handles 4-bit quantization natively on Apple Silicon — no Triton, no CUDA needed.
W&B logs training metrics via --report-to wandb, artifacts logged manually.
Adapter is pushed to HuggingFace org (mistral-hackaton-2026) after training.
"""

import os
import json
import subprocess
import wandb

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv(override=True)

# Authenticate HuggingFace before any HF operations
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass  # token set via env var, login optional

console = Console()

BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
MODEL_NAME = "security-aware-ministral-3b"
ADAPTER_PATH = "./models/ministral-3b-security-lora"
HF_ORG = "mistral-hackaton-2026"


def prepare_mlx_data(training_file: str, output_dir: str = "data/mlx") -> str:
    """
    MLX-LM natively supports conversational data in `{"messages": [{"role": "...", "content": "..."}]}` format.
     Crucially, when using this format, mlx-lm automatically applies the model's chat template
    AND masks the system/user prompts so loss is ONLY calculated on the assistant's completions.
    Passing raw text formats causes catastrophic overfitting where the model tries to predict the prompts itself.
    """
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"[dim]Copying {training_file} to MLX training directory...[/dim]")

    import shutil
    output_file = f"{output_dir}/train.jsonl"
    shutil.copy(training_file, output_file)
    
    valid_count = 0
    with open(training_file) as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                msgs = data.get("messages", [])
                
                # Mistral v0.3 tokenizer doesn't support 'system' role. 
                # Merge it with the first 'user' msg.
                if len(msgs) > 0 and msgs[0]["role"] == "system":
                    sys_content = msgs[0]["content"]
                    if len(msgs) > 1 and msgs[1]["role"] == "user":
                        msgs[1]["content"] = f"{sys_content}\n\n{msgs[1]['content']}"
                        msgs = msgs[1:] # discard standalone system message
                
                data["messages"] = msgs
                f_out.write(json.dumps(data) + "\n")
                valid_count += 1
            except Exception:
                pass

    console.print(f"[green]✅ Validated {valid_count} conversational examples → {output_file}[/green]")
    return output_dir


def run_local_sft(
    config: dict,
    training_file: str = "data/train.jsonl",
    model_name: str = MODEL_NAME,
    num_iters: int = 100,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    lora_layers: int = 8,
    adapter_path: str = ADAPTER_PATH,
):
    """
    Fine-tune Ministral-3B locally using Apple MLX LoRA.
    Logs metrics to W&B via mlx-lm's built-in wandb reporter,
    and logs dataset + model artifacts via wandb Python API.
    """
    console.print(Panel(
        f"🚀 Starting Local Fine-Tuning\n"
        f"Model: [cyan]{BASE_MODEL}[/cyan]\n"
        f"Method: Apple MLX LoRA (native M4 Max)\n"
        f"Iterations: {num_iters}",
        style="bold blue"
    ))

    from src.config import init_wandb
    run = init_wandb(config, run_name=f"mlx-sft-{model_name}", job_type="training")

    wandb.config.update({
        "base_model": BASE_MODEL,
        "model_name": model_name,
        "num_iters": num_iters,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora_layers": lora_layers,
        "method": "mlx-lora",
        "backend": "apple-mps-mlx",
    }, allow_val_change=True)

    # Log training dataset as W&B Artifact
    dataset_artifact = wandb.Artifact(
        name="security-training-data",
        type="dataset",
        description="Security-aware reasoning SFT dataset for Ministral-3B fine-tuning",
        metadata={"num_examples": sum(1 for _ in open(training_file))}
    )
    dataset_artifact.add_file(training_file)
    run.log_artifact(dataset_artifact)
    console.print("[green]✅ Dataset logged to W&B Artifacts[/green]")

    # Convert data to MLX format
    mlx_data_dir = prepare_mlx_data(training_file)

    # Create output adapter directory
    os.makedirs(adapter_path, exist_ok=True)

    # Build mlx_lm training command
    # mlx-lm's tuner has native --report-to wandb support
    cmd = [
        "python3", "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", mlx_data_dir,
        "--adapter-path", adapter_path,
        "--num-layers", str(lora_layers),
        "--batch-size", str(batch_size),
        "--iters", str(num_iters),
        "--learning-rate", str(learning_rate),
        "--steps-per-report", "10",
        "--steps-per-eval", "25",
        "--max-seq-length", "1024",
        "--grad-checkpoint",
        "--report-to", "wandb",
        "--project-name", config.get("wandb_project", "mistral-hackathon"),
    ]

    console.print(f"[dim]Running: {' '.join(cmd[:8])}...[/dim]")
    console.print("[bold yellow]Training in progress — watch W&B for live metrics...[/bold yellow]")

    # Set wandb env vars so mlx-lm's wandb call picks up the right project
    env = os.environ.copy()
    env["WANDB_PROJECT"] = config.get("wandb_project", "mistral-hackathon")
    env["WANDB_ENTITY"] = config.get("wandb_entity", "")
    env["WANDB_API_KEY"] = config.get("wandb_api_key", os.getenv("WANDB_API_KEY", ""))

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        console.print("[red]❌ Training failed. Check output above for errors.[/red]")
        run.finish(exit_code=1)
        raise RuntimeError("MLX training failed")

    console.print("[green]✅ Training complete![/green]")

    # Log fine-tuned adapter as W&B Model Artifact
    hf_repo_url = None
    if os.path.exists(adapter_path):
        model_artifact = wandb.Artifact(
            name=f"{model_name}-lora-adapter",
            type="model",
            description=(
                f"LoRA adapter for {BASE_MODEL} fine-tuned on security-aware reasoning. "
                f"Trained locally with Apple MLX on M4 Max."
            ),
            metadata={
                "base_model": BASE_MODEL,
                "lora_layers": lora_layers,
                "num_iters": num_iters,
                "learning_rate": learning_rate,
                "task": "security-aware-reasoning",
                "framework": "mlx-lm",
                "hardware": "Apple M4 Max (MPS)",
            },
        )
        model_artifact.add_dir(adapter_path)
        run.log_artifact(model_artifact)
        console.print("[green]✅ LoRA adapter registered in W&B Models Registry[/green]")

        # Push adapter to HuggingFace org
        hf_repo_url = push_adapter_to_hf(adapter_path, model_name)

    run.finish()

    console.print(Panel(
        f"[green bold]Fine-tuning complete![/green bold]\n\n"
        f"Adapter saved: [cyan]{adapter_path}[/cyan]\n"
        f"Base model: [cyan]{BASE_MODEL}[/cyan]\n"
        + (f"HF Repo: [cyan]{hf_repo_url}[/cyan]\n" if hf_repo_url else "")
        + f"\nView in W&B:\n"
        f"  https://wandb.ai/{config.get('wandb_entity')}/{config.get('wandb_project')}",
        title="✅ Success",
        style="green",
    ))

    return adapter_path


def push_adapter_to_hf(adapter_path: str, run_name: str, run_name_suffix: str = "") -> str | None:
    """Push the LoRA adapter directory to HuggingFace (org, then user namespace)."""
    if not HF_TOKEN:
        console.print("[yellow]⚠️ No HUGGINGFACE_TOKEN — skipping HF push[/yellow]")
        return None

    try:
        from huggingface_hub import HfApi, whoami
        api = HfApi(token=HF_TOKEN)
        user_info = whoami(token=HF_TOKEN)
        hf_user = user_info["name"]

        repo_name = f"{run_name}-mlx-lora".replace("_", "-").lower()

        # Try org namespace first, fall back to user namespace
        for namespace in [HF_ORG, hf_user]:
            repo_id = f"{namespace}/{repo_name}"
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=adapter_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {run_name} MLX LoRA adapter (Ministral-3B security fine-tune)",
                )
                url = f"https://huggingface.co/{repo_id}"
                console.print(f"[green]✅ Adapter pushed to HF: {url}[/green]")
                return url
            except Exception as e:
                console.print(f"[dim]HF push to {namespace} failed: {e!s:.100} — trying next...[/dim]")
                continue

        console.print("[yellow]⚠️ HF push failed for all namespaces[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]⚠️ HF push error: {e}[/yellow]")
        return None


if __name__ == "__main__":
    from src.config import load_config
    c = load_config()
    if not os.path.exists("data/train.jsonl"):
        console.print("[red]❌ No training data. Run `python3 run.py data` first.[/red]")
        raise SystemExit(1)
    run_local_sft(c)
