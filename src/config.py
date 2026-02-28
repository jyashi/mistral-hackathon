"""
Configuration and setup module for the hackathon project.
Handles API keys, W&B initialization, and Weave setup.
"""

import os
import wandb
import weave
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_config() -> dict:
    """Load configuration from .env file and environment variables."""
    load_dotenv(override=True)

    config = {
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
        "mistral_api_key": os.getenv("MISTRAL_API_KEY"),
        # Optional: personal key with billing for fine-tuning jobs
        "mistral_finetune_api_key": os.getenv("MISTRAL_FINETUNE_API_KEY") or os.getenv("MISTRAL_API_KEY"),
        "wandb_project": os.getenv("WANDB_PROJECT", "mistral-hackathon"),
        "wandb_entity": os.getenv("WANDB_ENTITY"),
    }

    # Validate required keys
    missing = []
    if not config["wandb_api_key"]:
        missing.append("WANDB_API_KEY")
    if not config["mistral_api_key"]:
        missing.append("MISTRAL_API_KEY")

    if missing:
        console.print(
            Panel(
                f"[red bold]Missing API keys:[/red bold] {', '.join(missing)}\n\n"
                "1. Copy .env.example to .env\n"
                "2. Fill in your keys:\n"
                "   - W&B: https://wandb.ai/authorize\n"
                "   - Mistral: https://console.mistral.ai/api-keys",
                title="⚠️ Setup Required",
                border_style="red",
            )
        )
        raise SystemExit(1)

    return config


def init_wandb(config: dict, run_name: str = None, job_type: str = "training"):
    """Initialize W&B run for experiment tracking."""
    wandb.login(key=config["wandb_api_key"])

    run = wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        name=run_name,
        job_type=job_type,
        config={
            "model": "mistral-small-latest",
            "method": "serverless-sft",
            "framework": "wandb-training",
        },
    )

    console.print(f"[green]✅ W&B run initialized:[/green] {run.url}")
    return run


def init_weave(config: dict, project_name: str = None):
    """Initialize W&B Weave for tracing and evaluation."""
    project = project_name or config["wandb_project"]
    weave.init(project)
    console.print(f"[green]✅ Weave initialized for project:[/green] {project}")


def get_mistral_client(config: dict):
    """Create and return a Mistral client."""
    from mistralai import Mistral

    client = Mistral(api_key=config["mistral_api_key"])
    console.print("[green]✅ Mistral client ready[/green]")
    return client


if __name__ == "__main__":
    console.print(Panel("🔧 Verifying Setup", style="bold blue"))
    config = load_config()

    console.print("\n[bold]Initializing W&B...[/bold]")
    run = init_wandb(config, run_name="setup-test", job_type="setup")

    console.print("\n[bold]Initializing Weave...[/bold]")
    init_weave(config)

    console.print("\n[bold]Connecting to Mistral...[/bold]")
    client = get_mistral_client(config)

    # Quick test
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Say 'Hello Hackathon!' in one line"}],
    )
    console.print(f"[cyan]Mistral says:[/cyan] {response.choices[0].message.content}")

    run.finish()
    console.print(Panel("[green bold]✅ All systems ready!", style="green"))
