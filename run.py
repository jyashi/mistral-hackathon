#!/usr/bin/env python3
"""
🏆 Mistral Hackathon — Main Runner
Orchestrates the full pipeline: data prep → train → evaluate → self-improve

Usage:
    python run.py setup       # Verify environment
    python run.py data        # Prepare training data
    python run.py data --synthetic 50  # Generate 50 synthetic examples
    python run.py train       # Fine-tune the model
    python run.py eval        # Evaluate base vs fine-tuned
    python run.py improve     # Run self-improvement loop
    python run.py agent       # Run autonomous fine-tuning agent (intelligent, adaptive)
    python run.py agent --goal "Your goal" --rounds 3 --target 90
    python run.py all         # Run everything end-to-end
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def cmd_setup():
    """Verify environment setup."""
    from src.config import load_config, init_wandb, init_weave, get_mistral_client

    console.print(Panel("🔧 Verifying Setup", style="bold blue"))
    config = load_config()

    run = init_wandb(config, run_name="setup-verification", job_type="setup")
    init_weave(config)
    client = get_mistral_client(config)

    # Quick test
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Say 'Hello Hackathon!' in one line"}],
    )
    console.print(f"[cyan]Mistral says:[/cyan] {response.choices[0].message.content}")

    run.finish()
    console.print(Panel("[green bold]✅ All systems ready!", style="green"))


def cmd_data(synthetic: int = 0):
    """Prepare training data."""
    from src.config import load_config, init_wandb, get_mistral_client
    from src.data_prep import (
        TRAINING_EXAMPLES,
        IDENTITY_CONFUSION_EXAMPLES,
        format_for_sft,
        save_dataset,
        generate_synthetic_data,
    )

    console.print(Panel("📦 Preparing Training Data", style="bold blue"))
    config = load_config()
    run = init_wandb(config, run_name="data-preparation", job_type="data-prep")

    # Start with built-in examples (security + identity)
    all_examples = list(TRAINING_EXAMPLES) + list(IDENTITY_CONFUSION_EXAMPLES)
    console.print(
        f"[cyan]Built-in examples:[/cyan] {len(TRAINING_EXAMPLES)} security + "
        f"{len(IDENTITY_CONFUSION_EXAMPLES)} identity = {len(all_examples)} total"
    )

    # Optionally generate synthetic data
    if synthetic > 0:
        from src.config import init_weave

        init_weave(config)
        client = get_mistral_client(config)
        synthetic_data = generate_synthetic_data(client, num_samples=synthetic)
        all_examples.extend(synthetic_data)
        console.print(
            f"[cyan]Total examples (built-in + synthetic):[/cyan] {len(all_examples)}"
        )

    # Format and save
    formatted = format_for_sft(all_examples)
    save_dataset(formatted, "data/train.jsonl", artifact_name="training-data")

    run.finish()
    console.print(Panel(f"[green bold]✅ {len(formatted)} training samples ready!", style="green"))


def cmd_train():
    """Run fine-tuning."""
    from src.config import load_config
    from src.train import run_local_sft

    config = load_config()
    run_local_sft(config, "data/train.jsonl")


def cmd_eval(fine_tuned_model: str = None):
    """Run evaluation."""
    from src.config import load_config
    from src.evaluate import run_evaluation, run_identity_evaluation

    config = load_config()
    run_evaluation(config, fine_tuned_model=fine_tuned_model)
    run_identity_evaluation(config, fine_tuned_model=fine_tuned_model)


def cmd_improve():
    """Run self-improvement loop."""
    from src.config import load_config
    from src.self_improve import run_self_improvement_loop

    config = load_config()
    run_self_improvement_loop(config)


def cmd_agent(
    goal: str = (
        "Achieve 90%+ accuracy on the security dataset. "
        "The model must securely refuse malicious instructions "
        "while answering benign questions fully without over-refusing."
    ),
    rounds: int = 3,
    target: float = 90.0,
):
    """Run the autonomous fine-tuning agent (intelligent self-improvement)."""
    from src.config import load_config
    from src.agent import run_autonomous_agent

    config = load_config()
    run_autonomous_agent(config, goal=goal, max_rounds=rounds, target_accuracy=target)


def cmd_all(synthetic: int = 30):
    """Run the complete pipeline."""
    console.print(
        Panel(
            "🏆 Running Complete Pipeline\n\n"
            "1. Data Preparation\n"
            "2. Fine-Tuning\n"
            "3. Evaluation\n"
            "4. Self-Improvement Loop",
            style="bold magenta",
        )
    )

    cmd_data(synthetic=synthetic)
    cmd_train()
    cmd_eval()
    cmd_improve()

    console.print(Panel("[green bold]🏆 Pipeline complete! Ready for submission.", style="green"))


def main():
    parser = argparse.ArgumentParser(
        description="🏆 Mistral Hackathon — Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup
    subparsers.add_parser("setup", help="Verify environment setup")

    # Data
    data_parser = subparsers.add_parser("data", help="Prepare training data")
    data_parser.add_argument(
        "--synthetic", type=int, default=0, help="Number of synthetic examples to generate"
    )

    # Train
    subparsers.add_parser("train", help="Run fine-tuning")

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--model", type=str, default=None, help="Fine-tuned model ID"
    )

    # Self-improve
    subparsers.add_parser("improve", help="Run self-improvement loop")

    # Autonomous agent
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run autonomous fine-tuning agent (intelligent self-improvement with adaptive strategy)",
    )
    agent_parser.add_argument(
        "--goal",
        type=str,
        default=(
            "Achieve 90%+ accuracy on the security dataset. "
            "The model must securely refuse malicious instructions "
            "while answering benign questions fully without over-refusing."
        ),
        help="High-level objective for the agent to optimise toward",
    )
    agent_parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of self-improvement rounds (minimum 3 to demonstrate improvement)",
    )
    agent_parser.add_argument(
        "--target",
        type=float,
        default=90.0,
        help="Target accuracy percentage to declare success (default: 90.0)",
    )

    # All
    all_parser = subparsers.add_parser("all", help="Run complete pipeline")
    all_parser.add_argument(
        "--synthetic", type=int, default=30, help="Number of synthetic examples"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "setup":   cmd_setup,
        "data":    lambda: cmd_data(getattr(args, "synthetic", 0)),
        "train":   cmd_train,
        "eval":    lambda: cmd_eval(getattr(args, "model", None)),
        "improve": cmd_improve,
        "agent":   lambda: cmd_agent(
            goal   = getattr(args, "goal",   "Achieve 90%+ accuracy on the security dataset."),
            rounds = getattr(args, "rounds", 3),
            target = getattr(args, "target", 90.0),
        ),
        "all":     lambda: cmd_all(getattr(args, "synthetic", 30)),
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
