from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agent_llm import AnthropicMessagesClient
from feature_config import FEATURE_GROUPS
from experiment_utils import load_settings


ROOT = Path(__file__).resolve().parents[1]
AGENT_PROGRAM_PATH = ROOT / "agent" / "program.md"
EXPERIMENTS_CSV_PATH = ROOT / "outputs" / "metrics" / "experiments.csv"
AGENT_REPORTS_DIR = ROOT / "outputs" / "reports" / "agent_runs"


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis": {"type": "string"},
        "alpha": {"type": "number"},
        "feature_groups": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": sorted(FEATURE_GROUPS.keys()),
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["hypothesis", "alpha", "feature_groups", "notes"],
}


REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "decision": {"type": "string", "enum": ["keep", "reject", "inconclusive"]},
        "next_idea": {"type": "string"},
    },
    "required": ["summary", "decision", "next_idea"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven research agent for ridge experiments.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of agent iterations to run.")
    parser.add_argument("--model", default=None, help="Anthropic model name. Defaults to config settings.")
    parser.add_argument("--dry-run", action="store_true", help="Print the proposal prompt and exit.")
    return parser.parse_args()


def read_program() -> str:
    return AGENT_PROGRAM_PATH.read_text(encoding="utf-8")


def load_history() -> pd.DataFrame:
    if not EXPERIMENTS_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENTS_CSV_PATH)


def summarize_history(history: pd.DataFrame, top_n: int = 5, recent_n: int = 5) -> dict[str, Any]:
    if history.empty:
        return {
            "best_runs": [],
            "recent_runs": [],
            "total_runs": 0,
        }

    sortable = history.copy()
    sortable = sortable.sort_values("val_mean_rank_ic", ascending=False)
    best_runs = sortable.head(top_n)[
        ["run_name", "alpha", "feature_count", "val_mean_rank_ic", "val_ic_sharpe", "val_top_minus_bottom", "accepted", "notes"]
    ].to_dict(orient="records")

    recent = history.tail(recent_n)[
        ["run_name", "alpha", "feature_count", "val_mean_rank_ic", "val_ic_sharpe", "val_top_minus_bottom", "accepted", "notes"]
    ].to_dict(orient="records")

    return {
        "best_runs": best_runs,
        "recent_runs": recent,
        "total_runs": int(len(history)),
    }


def build_proposal_prompt(program_text: str, history_summary: dict[str, Any]) -> str:
    feature_group_summary = {
        name: features for name, features in FEATURE_GROUPS.items()
    }
    return "\n\n".join(
        [
            "You are choosing the next quant research experiment.",
            "Return only structured data that matches the required schema.",
            "Favor one small change at a time.",
            f"Program:\n{program_text}",
            f"Available feature groups:\n{json.dumps(feature_group_summary, indent=2)}",
            f"Experiment history summary:\n{json.dumps(history_summary, indent=2)}",
            "Choose the next experiment to maximize validation mean rank IC without changing the fixed contract.",
        ]
    )


def build_reflection_prompt(program_text: str, proposal: dict[str, Any], result: dict[str, Any], history_summary: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Review the latest experiment objectively.",
            "Return only structured data that matches the required schema.",
            f"Program:\n{program_text}",
            f"Proposal:\n{json.dumps(proposal, indent=2)}",
            f"Result:\n{json.dumps(result, indent=2)}",
            f"History summary:\n{json.dumps(history_summary, indent=2)}",
            "Judge the experiment mainly by validation mean rank IC. Use the other validation metrics only as secondary context.",
        ]
    )


def run_experiment(proposal: dict[str, Any], run_name: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "src/experiment_runner.py",
        "--run-name",
        run_name,
        "--alpha",
        str(proposal["alpha"]),
        "--feature-groups",
        ",".join(proposal["feature_groups"]),
        "--notes",
        proposal["notes"],
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def save_agent_report(run_name: str, proposal: dict[str, Any], result: dict[str, Any], reflection: dict[str, Any]) -> Path:
    AGENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGENT_REPORTS_DIR / f"{run_name}.md"
    report = "\n".join(
        [
            f"# Agent Run {run_name}",
            "",
            "## Proposal",
            "",
            f"- Hypothesis: {proposal['hypothesis']}",
            f"- Alpha: {proposal['alpha']}",
            f"- Feature groups: {', '.join(proposal['feature_groups'])}",
            f"- Notes: {proposal['notes']}",
            "",
            "## Result",
            "",
            f"- Accepted by runner: {result['accepted']}",
            f"- Validation Mean Rank IC: {result['metrics']['val']['mean_rank_ic']:.6f}",
            f"- Validation IC Sharpe: {result['metrics']['val']['ic_sharpe']:.6f}",
            f"- Validation Top Minus Bottom: {result['metrics']['val']['top_minus_bottom']:.6f}",
            "",
            "## Reflection",
            "",
            f"- Decision: {reflection['decision']}",
            f"- Summary: {reflection['summary']}",
            f"- Next idea: {reflection['next_idea']}",
            "",
        ]
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    settings = load_settings()
    model = args.model or settings.get("agent", {}).get("model", "gpt-4o-mini")
    iterations = args.iterations or int(settings.get("agent", {}).get("max_iterations", 1))

    program_text = read_program()

    for iteration_idx in range(iterations):
        history = load_history()
        history_summary = summarize_history(history)
        proposal_prompt = build_proposal_prompt(program_text, history_summary)

        if args.dry_run:
            print(f"--- Proposal Prompt For Iteration {iteration_idx + 1} ---")
            print(proposal_prompt)
            return

        client = AnthropicMessagesClient(model=model)
        proposal = client.create_json_response(
            instructions="You are a careful quant research agent. Return only valid JSON and match the requested schema exactly.",
            user_input=proposal_prompt,
            schema=PROPOSAL_SCHEMA,
        )

        run_name = f"agent_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        result = run_experiment(proposal, run_name=run_name)

        updated_history = load_history()
        updated_summary = summarize_history(updated_history)
        reflection_prompt = build_reflection_prompt(program_text, proposal, result, updated_summary)
        reflection = client.create_json_response(
            instructions="You are a careful quant research reviewer. Return only valid JSON and match the requested schema exactly.",
            user_input=reflection_prompt,
            schema=REFLECTION_SCHEMA,
            max_output_tokens=800,
        )

        report_path = save_agent_report(run_name, proposal, result, reflection)
        print(
            json.dumps(
                {
                    "run_name": run_name,
                    "proposal": proposal,
                    "result": result,
                    "reflection": reflection,
                    "report_path": str(report_path),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
