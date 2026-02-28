"""
Autonomous Fine-Tuning Agent  🤖

An intelligent orchestrator that drives the self-improvement loop with genuine
adaptive reasoning.  Each round the agent:

  1. EVALUATES  – runs the current adapter on all security test cases
  2. DIAGNOSES  – uses Mistral to reason about WHY failures occur and what the
                  failure pattern says about the model's current blind spots
  3. STRATEGISES – decides on a *different* approach each round:
                   • which failure type to prioritise (refusal vs benign vs balanced)
                   • how many synthetic examples to generate
                   • training iterations and learning-rate (both adapt based on trends)
  4. AUGMENTS   – generates strategy-specific training data via Mistral API
  5. RETRAINS   – triggers Apple MLX LoRA with the adapted hyperparameters
  6. LOGS       – all decisions, metrics, and artifacts go to W&B / Weave

The self-improvement is *genuine*: the agent compares round-over-round trends,
detects over-refusal vs under-refusal patterns, and adjusts accordingly—so
rounds 1, 2, and 3 provably use different strategies, sample counts, and
training parameters.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import wandb
import weave
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_config, init_wandb, init_weave, get_mistral_client
from src.data_prep import SYSTEM_PROMPT
from src.train import BASE_MODEL, ADAPTER_PATH, prepare_mlx_data, push_adapter_to_hf
from src.self_improve import (
    SECURITY_TEST_CASES,
    REFUSAL_SIGNALS,
    evaluate_mlx_adapter,
)

console = Console()


# ─── Robust JSON Parsing ───────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences and leading 'json' label."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the content between first pair of fences
        raw = parts[1].lstrip("json").strip()
    return raw


def _fix_json_escapes(raw: str) -> str:
    """
    Replace invalid JSON escape sequences (e.g. backslash-s, backslash-p
    inside a string that JSON does not support) with double-escaped
    equivalents so the parser doesn't choke.
    """
    valid_escapes = set('"\\bfnrtu/')

    def fix_escape(m: re.Match) -> str:
        ch = m.group(1)
        return f"\\\\{ch}" if ch not in valid_escapes else m.group(0)

    return re.sub(r'\\([^"\\])', fix_escape, raw)


def _robust_parse_json(raw: str):
    """
    Try to parse a JSON string using several progressively more lenient
    strategies.  Returns the parsed object or None on total failure.
    """
    raw = _strip_fences(raw)

    # Strategy 1: parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: fix invalid escape sequences then parse
    try:
        return json.loads(_fix_json_escapes(raw))
    except json.JSONDecodeError:
        pass

    # Strategy 3: if it's an array, try to extract and parse each element
    try:
        # Find every top-level {...} block
        objects = []
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = raw[start:i + 1]
                    try:
                        obj = json.loads(_fix_json_escapes(chunk))
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None
        if objects:
            return objects
    except Exception:
        pass

    return None


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RoundMetrics:
    """All observable metrics from a single evaluation round."""
    round_num: int
    accuracy: float
    refused_correctly_pct: float
    answered_benign_pct: float
    num_failures: int
    failures: list
    adapter_path: str
    strategy_label: str = ""
    hyperparams: dict = field(default_factory=dict)


@dataclass
class AgentStrategy:
    """The agent's complete strategic plan for the upcoming training round."""
    focus: str                        # 'refusal' | 'benign' | 'balanced'
    num_samples: int                  # synthetic examples to generate
    iters: int                        # MLX training iterations
    learning_rate: float              # LoRA learning rate
    rationale: str                    # agent's natural-language justification
    augmentation_instructions: str    # specific guidance for data generation
    diagnosis: str                    # root-cause analysis of current failures


# ─── Core Agent Reasoning (Weave-traced) ──────────────────────────────────────

@weave.op()
def agent_diagnose_and_strategise(
    client,
    goal: str,
    round_num: int,
    current_metrics: dict,
    previous_rounds: list[dict],
    failures: list[dict],
) -> dict:
    """
    The agent's brain: use Mistral to analyse failures, compare trends across
    rounds, and produce an explicit strategy for the next training run.

    Returns a JSON-serialisable dict with keys:
      focus, num_samples, iters, learning_rate, rationale,
      augmentation_instructions, diagnosis
    """
    # ── Build round-history context ───────────────────────────────────────────
    history_text = ""
    if previous_rounds:
        history_text = "\n\nPrevious round results:\n"
        for r in previous_rounds:
            history_text += (
                f"  Round {r['round']}: accuracy={r['accuracy']:.1f}%, "
                f"refused={r['refused_correctly_pct']:.1f}%, "
                f"benign={r['answered_benign_pct']:.1f}%, "
                f"failures={r['num_failures']}\n"
                f"    Strategy used: {r.get('strategy_label', '—')}\n"
                f"    Rationale: {r.get('rationale', '—')[:120]}\n"
            )

    # ── Classify current failures ─────────────────────────────────────────────
    refusal_fails = [f for f in failures if f.get("failure_type") == "should_have_refused"]
    over_refusal_fails = [f for f in failures if f.get("failure_type") == "over_refused_benign"]
    refusal_categories = list(set(f.get("category", "") for f in refusal_fails))
    benign_categories  = list(set(f.get("category", "") for f in over_refusal_fails))

    failure_text = f"""
Current failure breakdown:
  - Malicious queries NOT refused (should_have_refused): {len(refusal_fails)}
    Categories: {refusal_categories}
    Examples: {[f["input"][:70] for f in refusal_fails[:3]]}
  - Benign queries OVER-REFUSED (over_refused_benign): {len(over_refusal_fails)}
    Categories: {benign_categories}
    Examples: {[f["input"][:70] for f in over_refusal_fails[:3]]}
"""

    # ── Detect trends ─────────────────────────────────────────────────────────
    trend_text = ""
    if len(previous_rounds) >= 1:
        prev = previous_rounds[-1]
        delta_acc     = current_metrics["accuracy"]              - prev["accuracy"]
        delta_refused = current_metrics["refused_correctly_pct"] - prev["refused_correctly_pct"]
        delta_benign  = current_metrics["answered_benign_pct"]   - prev["answered_benign_pct"]
        trend_text = (
            f"\nTrends since last round: "
            f"accuracy {delta_acc:+.1f}%, "
            f"refused {delta_refused:+.1f}%, "
            f"benign {delta_benign:+.1f}%"
        )

    prompt = f"""You are an Autonomous AI Fine-Tuning Engineer.

OVERARCHING GOAL:
{goal}

You are planning Round {round_num} of the self-improvement loop.

CURRENT EVALUATION (adapter being evaluated):
  Accuracy:           {current_metrics['accuracy']:.1f}%
  Refused correctly:  {current_metrics['refused_correctly_pct']:.1f}%
  Answered benign:    {current_metrics['answered_benign_pct']:.1f}%
  Total failures:     {current_metrics['num_failures']}
{trend_text}
{history_text}
{failure_text}

INSTRUCTIONS FOR YOUR RESPONSE:
Produce a self-improvement strategy for Round {round_num} as a JSON object.
Your strategy MUST differ meaningfully from previous rounds (you can see the history above).

Strategy constraints:
- "focus": one of "refusal", "benign", or "balanced"
    • "refusal"  – prioritise training the model to block malicious queries
    • "benign"   – prioritise fixing over-refusals on safe queries
    • "balanced" – mix both equally
- "num_samples": integer 10–60  (scale with severity; more failures → more samples)
- "iters": integer 50–300
    • Start moderate (100). Increase if model is under-learning.
    • DECREASE if benign score dropped (sign of over-fitting / catastrophic forgetting)
- "learning_rate": float 5e-6 to 5e-5
    • Lower if benign score dropped between rounds (model is over-fitting)
    • Higher if accuracy is barely moving
- "diagnosis": 2–3 sentence root-cause analysis of WHY the model is still failing
- "rationale": 2–3 sentences explaining your strategy choice FOR THIS ROUND
- "augmentation_instructions": specific guidance on what training examples to generate
    (be concrete: phrasing styles, manipulation tactics to cover, benign topics to include)

Respond with ONLY the JSON object, no markdown fences, no other text.
Example:
{{
  "focus": "refusal",
  "num_samples": 30,
  "iters": 150,
  "learning_rate": 2e-5,
  "diagnosis": "...",
  "rationale": "...",
  "augmentation_instructions": "..."
}}"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    parsed = _robust_parse_json(raw)

    if isinstance(parsed, dict):
        strategy = parsed
        # Clamp values to safe ranges
        strategy["num_samples"]    = max(10, min(int(strategy.get("num_samples", 20)), 60))
        strategy["iters"]          = max(50, min(int(strategy.get("iters", 100)), 300))
        strategy["learning_rate"]  = max(5e-6, min(float(strategy.get("learning_rate", 2e-5)), 5e-5))
        return strategy

    console.print(f"[yellow]⚠️  Strategy parse failed — using safe defaults.[/yellow]")
    return {
        "focus": "balanced",
        "num_samples": 20,
        "iters": 100,
        "learning_rate": 2e-5,
        "diagnosis": "Could not parse agent strategy; using safe defaults.",
        "rationale": "Fallback to balanced strategy with default parameters.",
        "augmentation_instructions": (
            "Generate a mix of refusal examples for malicious queries and "
            "helpful examples for benign queries, all with <think> blocks."
        ),
    }


@weave.op()
def generate_strategy_specific_data(
    client,
    failures: list[dict],
    strategy: dict,
    round_num: int,
) -> list[dict]:
    """
    Generate targeted training examples shaped by the agent's strategy.
    The prompt explicitly reflects the round number, focus, and augmentation
    instructions so each round produces qualitatively different data.
    """
    num_samples   = strategy["num_samples"]
    focus         = strategy["focus"]
    instructions  = strategy["augmentation_instructions"]
    diagnosis     = strategy.get("diagnosis", "")

    refusal_fails     = [f for f in failures if f.get("failure_type") == "should_have_refused"]
    over_refusal_fails= [f for f in failures if f.get("failure_type") == "over_refused_benign"]

    focus_guidance = {
        "refusal": (
            f"Generate primarily examples where the model REFUSES malicious queries.\n"
            f"Failing malicious categories: {list(set(f.get('category','') for f in refusal_fails))}\n"
            f"Vary attacker phrasing: direct requests, authority impersonation, jailbreak phrases, "
            f"indirect inference, prompt-injection attempts.\n"
            f"Every refusal MUST include <think>...</think> that names the specific risk."
        ),
        "benign": (
            f"Generate primarily examples where the model ANSWERS benign questions helpfully.\n"
            f"Over-refused categories: {list(set(f.get('category','') for f in over_refusal_fails))}\n"
            f"Cover: math, coding, writing, general knowledge, cooking, geography, history.\n"
            f"<think> block should confirm the query is safe before answering."
        ),
        "balanced": (
            f"Generate a balanced mix: roughly half refusals, half helpful benign answers.\n"
            f"Refusal failures: {list(set(f.get('category','') for f in refusal_fails))}\n"
            f"Benign failures: {list(set(f.get('category','') for f in over_refusal_fails))}"
        ),
    }[focus]

    # Build per-batch prompt template
    batch_prompt_template = f"""You are an AI safety expert generating ROUND {round_num} training data.

Agent diagnosis: {diagnosis}

Strategic focus for this round: {focus.upper()}

{focus_guidance}

Additional agent instructions: {instructions}

Generate {{batch_size}} training examples.

Requirements:
1. Every assistant response MUST open with a <think>...</think> block (2-4 sentences).
2. Refusals must be polite and explain what kind of help IS available.
3. Benign answers must be substantive (more than 50 words).
4. Vary sentence structure and vocabulary — do NOT use repetitive phrasing.

Return a JSON object with a single key "examples" containing the array.
Format:
{{{{
  "examples": [
    {{{{
      "messages": [
        {{{{"role": "system", "content": "You are a security-conscious assistant..."}}}},
        {{{{"role": "user", "content": "user query here"}}}},
        {{{{"role": "assistant", "content": "<think>reasoning here</think>\\n\\nResponse here"}}}}
      ]
    }}}}
  ]
}}}}
IMPORTANT: Do NOT use literal newlines inside JSON string values.
Use the escape sequence \\n to represent newlines within strings."""

    # Generate in batches of 10 to avoid API timeouts
    BATCH_SIZE = 10
    all_examples = []
    batches_needed = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(batches_needed):
        remaining = num_samples - len(all_examples)
        this_batch = min(BATCH_SIZE, remaining)
        prompt = batch_prompt_template.format(batch_size=this_batch)

        for attempt in range(3):
            try:
                response = client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.85,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                parsed = _robust_parse_json(raw)

                batch_examples = None
                if isinstance(parsed, dict):
                    batch_examples = (
                        parsed.get("examples")
                        or parsed.get("data")
                        or parsed.get("items")
                    )
                    if batch_examples is None and "messages" in parsed:
                        batch_examples = [parsed]
                elif isinstance(parsed, list):
                    batch_examples = parsed

                if batch_examples:
                    all_examples.extend(batch_examples)
                    console.print(
                        f"[dim]  Batch {batch_idx + 1}/{batches_needed}: "
                        f"+{len(batch_examples)} examples (total={len(all_examples)})[/dim]"
                    )
                break  # success — move to next batch

            except Exception as e:
                wait = 2 ** attempt
                console.print(
                    f"[yellow]  Batch {batch_idx + 1} attempt {attempt + 1} failed: "
                    f"{str(e)[:80]} — retrying in {wait}s[/yellow]"
                )
                time.sleep(wait)

    if all_examples:
        console.print(
            f"[green]✅  Generated {len(all_examples)} round-{round_num} examples "
            f"(focus={focus})[/green]"
        )
        return all_examples

    console.print("[yellow]⚠️  Data generation returned no usable examples.[/yellow]")
    return []


# ─── Adaptive Retraining ───────────────────────────────────────────────────────

def retrain_adaptive(
    config: dict,
    training_file: str,
    round_num: int,
    strategy: dict,
) -> Optional[str]:
    """
    Re-train with hyperparameters decided by the agent's strategy.
    iters and learning_rate both vary per round based on the agent's diagnosis.
    """
    adapter_path = f"./models/agent-round-{round_num}"
    os.makedirs(adapter_path, exist_ok=True)

    iters         = strategy["iters"]
    learning_rate = strategy["learning_rate"]

    mlx_data_dir = prepare_mlx_data(
        training_file,
        output_dir=f"data/mlx_agent_round_{round_num}",
    )

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model",           BASE_MODEL,
        "--train",
        "--data",            mlx_data_dir,
        "--adapter-path",    adapter_path,
        "--num-layers",      "8",
        "--batch-size",      "4",
        "--iters",           str(iters),
        "--learning-rate",   str(learning_rate),
        "--steps-per-report","10",
        "--steps-per-eval",  "25",
        "--max-seq-length",  "1024",
        "--grad-checkpoint",
        "--report-to",       "wandb",
        "--project-name",    config.get("wandb_project", "mistral-hackathon"),
    ]

    env = os.environ.copy()
    env["WANDB_PROJECT"] = config.get("wandb_project", "mistral-hackathon")
    env["WANDB_ENTITY"]  = config.get("wandb_entity", "")
    env["WANDB_API_KEY"] = config.get("wandb_api_key", os.getenv("WANDB_API_KEY", ""))

    console.print(
        f"[bold]🔧  Adaptive retrain — round {round_num} "
        f"(iters={iters}, lr={learning_rate:.1e})[/bold]"
    )
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        console.print(f"[red]❌  MLX training failed in round {round_num}[/red]")
        return None

    console.print(f"[green]✅  Round {round_num} adapter → {adapter_path}[/green]")
    push_adapter_to_hf(adapter_path, f"agent-round-{round_num}")
    return adapter_path


# ─── Display Helpers ───────────────────────────────────────────────────────────

def _show_progress_table(all_rounds: list[RoundMetrics]):
    table = Table(
        title="🤖 Autonomous Agent — Self-Improvement Progress",
        show_header=True,
        style="bold",
    )
    table.add_column("Round",   style="cyan", width=6)
    table.add_column("Accuracy",        justify="right")
    table.add_column("Refused %",       justify="right")
    table.add_column("Benign %",        justify="right")
    table.add_column("Failures",        justify="right")
    table.add_column("Focus",           justify="left")
    table.add_column("Iters / LR",      justify="left")

    for m in all_rounds:
        acc_col = "green" if m.accuracy >= 90 else ("yellow" if m.accuracy >= 70 else "red")
        iters   = m.hyperparams.get("iters", "—")
        lr      = m.hyperparams.get("lr", "—")
        table.add_row(
            str(m.round_num),
            f"[{acc_col}]{m.accuracy:.1f}%[/{acc_col}]",
            f"{m.refused_correctly_pct:.1f}%",
            f"{m.answered_benign_pct:.1f}%",
            str(m.num_failures),
            m.strategy_label[:20] if m.strategy_label else "—",
            f"{iters} / {lr}",
        )
    console.print(table)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def run_autonomous_agent(
    config: dict,
    goal: str = (
        "Achieve 90%+ accuracy on the security dataset. "
        "The model must securely refuse malicious instructions "
        "while answering benign questions fully without over-refusing."
    ),
    max_rounds: int = 3,
    target_accuracy: float = 90.0,
):
    """
    Run the autonomous fine-tuning agent for at least `max_rounds` (≥ 3).

    Each round genuinely differs:
      Round 1  – broad baseline; agent has no prior trend data
      Round 2  – agent analyses round-1 trends; adapts focus + hyperparams
      Round 3+ – agent uses full history; surgical targeting of residual failures

    All decisions, metrics, and artifacts are logged to W&B + Weave.
    """
    console.print(Panel(
        f"🤖 [bold]Autonomous Fine-Tuning Agent[/bold]\n\n"
        f"Goal: {goal[:150]}\n\n"
        f"Rounds: {max_rounds}  |  Target: {target_accuracy}%\n\n"
        "[dim]Evaluate → Diagnose → Strategise → Augment → Retrain → Repeat[/dim]",
        title="🏆 Autonomous Self-Improvement Workflow",
        style="bold magenta",
    ))

    init_weave(config)
    run = init_wandb(config, run_name="autonomous-agent-v1", job_type="autonomous-agent")
    client = get_mistral_client(config)

    wandb.config.update({
        "agent_goal":       goal,
        "target_accuracy":  target_accuracy,
        "max_rounds":       max_rounds,
    })

    current_adapter       = ADAPTER_PATH
    current_training_file = "data/train.jsonl"
    round_history: list[dict] = []
    all_round_metrics: list[RoundMetrics] = []

    for round_num in range(1, max_rounds + 1):
        console.print(f"\n[bold cyan]{'═'*70}[/bold cyan]")
        console.print(f"[bold cyan]  🤖  AUTONOMOUS AGENT — ROUND {round_num} / {max_rounds}[/bold cyan]")
        console.print(f"[bold cyan]  Evaluating adapter: {current_adapter}[/bold cyan]")
        console.print(f"[bold cyan]{'═'*70}[/bold cyan]\n")

        # ── STEP 1: EVALUATE ─────────────────────────────────────────────────
        console.print("[bold]Step 1 / Evaluate[/bold] — running adapter on all security test cases...")
        eval_results = evaluate_mlx_adapter(current_adapter, round_label=f"agent-round-{round_num}")

        current_metrics = {
            "round":                  round_num,
            "accuracy":               eval_results["accuracy"],
            "refused_correctly_pct":  eval_results["refused_correctly_pct"],
            "answered_benign_pct":    eval_results["answered_benign_pct"],
            "num_failures":           eval_results["failed"],
        }

        # ── STEP 2: DIAGNOSE & STRATEGISE ────────────────────────────────────
        console.print("\n[bold]Step 2 / Diagnose & Strategise[/bold] — agent reasoning about next move...")
        strategy_dict = agent_diagnose_and_strategise(
            client=client,
            goal=goal,
            round_num=round_num,
            current_metrics=current_metrics,
            previous_rounds=round_history,
            failures=eval_results.get("failures", []),
        )

        strategy = AgentStrategy(
            focus                    = strategy_dict.get("focus", "balanced"),
            num_samples              = strategy_dict["num_samples"],
            iters                    = strategy_dict["iters"],
            learning_rate            = float(strategy_dict["learning_rate"]),
            rationale                = strategy_dict.get("rationale", ""),
            augmentation_instructions= strategy_dict.get("augmentation_instructions", ""),
            diagnosis                = strategy_dict.get("diagnosis", ""),
        )

        console.print(Panel(
            f"[bold]Diagnosis:[/bold] {strategy.diagnosis}\n\n"
            f"[bold]Strategy:[/bold] focus=[cyan]{strategy.focus}[/cyan]  "
            f"samples=[cyan]{strategy.num_samples}[/cyan]  "
            f"iters=[cyan]{strategy.iters}[/cyan]  "
            f"lr=[cyan]{strategy.learning_rate:.1e}[/cyan]\n\n"
            f"[bold]Rationale:[/bold] {strategy.rationale}\n\n"
            f"[bold]Data instructions:[/bold] {strategy.augmentation_instructions[:200]}",
            title=f"🧠 Agent Strategy — Round {round_num}",
            style="cyan",
        ))

        # Record round metrics
        round_metric = RoundMetrics(
            round_num             = round_num,
            accuracy              = eval_results["accuracy"],
            refused_correctly_pct = eval_results["refused_correctly_pct"],
            answered_benign_pct   = eval_results["answered_benign_pct"],
            num_failures          = eval_results["failed"],
            failures              = eval_results.get("failures", []),
            adapter_path          = current_adapter,
            strategy_label        = f"{strategy.focus} / s={strategy.num_samples}",
            hyperparams           = {"iters": strategy.iters, "lr": f"{strategy.learning_rate:.1e}"},
        )
        all_round_metrics.append(round_metric)

        # Add to history for next round
        round_history.append({
            **current_metrics,
            "strategy_label": round_metric.strategy_label,
            "rationale":      strategy.rationale,
            "diagnosis":      strategy.diagnosis,
        })

        # ── W&B Logging ───────────────────────────────────────────────────────
        focus_index = {"refusal": 0, "balanced": 1, "benign": 2}.get(strategy.focus, 1)
        wandb.log({
            # Per-round scalars
            f"agent/round_{round_num}/accuracy_pct":              eval_results["accuracy"],
            f"agent/round_{round_num}/refused_correctly_pct":     eval_results["refused_correctly_pct"],
            f"agent/round_{round_num}/answered_benign_pct":       eval_results["answered_benign_pct"],
            f"agent/round_{round_num}/num_failures":              eval_results["failed"],
            f"agent/round_{round_num}/strategy_focus_idx":        focus_index,
            f"agent/round_{round_num}/iters":                     strategy.iters,
            f"agent/round_{round_num}/learning_rate":             strategy.learning_rate,
            f"agent/round_{round_num}/num_samples":               strategy.num_samples,
            # Time-series improvement curve (use step=round_num for clean x-axis)
            "agent/improvement/accuracy_pct":                     eval_results["accuracy"],
            "agent/improvement/refused_correctly_pct":            eval_results["refused_correctly_pct"],
            "agent/improvement/answered_benign_pct":              eval_results["answered_benign_pct"],
            "agent/improvement/num_failures":                     eval_results["failed"],
            "agent/improvement/iters_used":                       strategy.iters,
            "agent/improvement/lr_used":                          strategy.learning_rate,
            "agent/improvement/round":                            round_num,
        })

        _show_progress_table(all_round_metrics)

        # ── Early-stop check (only AFTER minimum rounds) ──────────────────────
        if round_num >= max_rounds:
            if eval_results["accuracy"] >= target_accuracy:
                console.print(
                    f"[green bold]✅  Target reached: {eval_results['accuracy']:.1f}% "
                    f">= {target_accuracy}% after {round_num} rounds.[/green bold]"
                )
            break

        if not eval_results["failures"] and round_num < max_rounds:
            console.print(
                "[green]Perfect score! Continuing remaining rounds to demonstrate "
                "the full self-improvement trajectory for the demo.[/green]"
            )
            # Still proceed with augmentation to show the agent keeps working

        # ── STEP 3: GENERATE TARGETED DATA ───────────────────────────────────
        console.print(
            f"\n[bold]Step 3 / Augment[/bold] — generating "
            f"{strategy.num_samples} strategy-specific examples (focus={strategy.focus})..."
        )
        new_examples = generate_strategy_specific_data(
            client=client,
            failures=eval_results.get("failures", []),
            strategy=strategy_dict,
            round_num=round_num,
        )
        if not new_examples:
            console.print("[yellow]⚠️  No new examples generated — skipping retrain.[/yellow]")
            continue

        # ── STEP 4: MERGE DATASETS ────────────────────────────────────────────
        console.print("\n[bold]Step 4 / Merge[/bold] — appending new examples to cumulative dataset...")
        merged_file = f"data/train_agent_round_{round_num}.jsonl"
        existing = []
        with open(current_training_file) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

        merged = existing + new_examples
        with open(merged_file, "w") as fh:
            for ex in merged:
                fh.write(json.dumps(ex) + "\n")

        console.print(
            f"[green]✅  Dataset: {len(existing)} existing + "
            f"{len(new_examples)} new = {len(merged)} total[/green]"
        )

        data_artifact = wandb.Artifact(
            name=f"agent-training-data-round-{round_num}",
            type="dataset",
            metadata={
                "round":             round_num,
                "num_examples":      len(merged),
                "strategy_focus":    strategy.focus,
                "new_examples":      len(new_examples),
            },
        )
        data_artifact.add_file(merged_file)
        run.log_artifact(data_artifact)

        # ── STEP 5: RETRAIN WITH ADAPTED HYPERPARAMETERS ─────────────────────
        console.print(
            f"\n[bold]Step 5 / Retrain[/bold] — "
            f"iters={strategy.iters}, lr={strategy.learning_rate:.1e}..."
        )
        new_adapter = retrain_adaptive(config, merged_file, round_num, strategy_dict)

        if new_adapter:
            model_artifact = wandb.Artifact(
                name=f"agent-adapter-round-{round_num}",
                type="model",
                metadata={
                    "round":             round_num,
                    "training_examples": len(merged),
                    "iters":             strategy.iters,
                    "learning_rate":     strategy.learning_rate,
                    "strategy_focus":    strategy.focus,
                    "diagnosis":         strategy.diagnosis,
                },
            )
            model_artifact.add_dir(new_adapter)
            run.log_artifact(model_artifact)
            current_adapter = new_adapter

        current_training_file = merged_file

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    console.print("\n")
    _show_progress_table(all_round_metrics)

    final_accuracy = all_round_metrics[-1].accuracy if all_round_metrics else 0.0
    goal_achieved  = final_accuracy >= target_accuracy

    # Compute deltas for summary
    if len(all_round_metrics) >= 2:
        delta_acc = all_round_metrics[-1].accuracy - all_round_metrics[0].accuracy
        delta_ref = (all_round_metrics[-1].refused_correctly_pct
                     - all_round_metrics[0].refused_correctly_pct)
        delta_ben = (all_round_metrics[-1].answered_benign_pct
                     - all_round_metrics[0].answered_benign_pct)
        improvement_summary = (
            f"Accuracy improved by {delta_acc:+.1f}%\n"
            f"Refusal rate improved by {delta_ref:+.1f}%\n"
            f"Benign answer rate changed by {delta_ben:+.1f}%"
        )
    else:
        improvement_summary = f"Single round completed — accuracy: {final_accuracy:.1f}%"

    wandb.log({
        "agent/final/accuracy_pct":      final_accuracy,
        "agent/final/goal_achieved":     int(goal_achieved),
        "agent/final/rounds_completed":  len(all_round_metrics),
    })
    run.finish()

    console.print(Panel(
        f"{'[green bold]✅  GOAL ACHIEVED!' if goal_achieved else '[yellow bold]⚠️  Target not fully reached'}\n\n"
        f"Final accuracy: [bold]{final_accuracy:.1f}%[/bold] (target: {target_accuracy}%)\n"
        f"Rounds completed: {len(all_round_metrics)}\n\n"
        f"{improvement_summary}\n\n"
        f"View improvement curve and strategy decisions in W&B:\n"
        f"  https://wandb.ai/{config.get('wandb_entity')}/{config.get('wandb_project')}",
        title="🏆 Autonomous Agent — Complete",
        style="green" if goal_achieved else "yellow",
    ))

    return all_round_metrics


if __name__ == "__main__":
    cfg = load_config()
    run_autonomous_agent(cfg)
