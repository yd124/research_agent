from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any

import pandas as pd

from agent_llm import create_json_client
from feature_config import FEATURE_GROUPS
from experiment_utils import load_settings


ROOT = Path(__file__).resolve().parents[1]
AGENT_PROGRAM_PATH = ROOT / "agent" / "program.md"
EXPERIMENTS_CSV_PATH = ROOT / "outputs" / "metrics" / "experiments.csv"
AGENT_REPORTS_DIR = ROOT / "outputs" / "reports" / "agent_runs"
FEATURES_PY_PATH = ROOT / "src" / "features.py"
SUPPORTED_EXTRA_FEATURE_KINDS = [
    "pct_change",
    "ma_gap",
    "rolling_std",
    "rolling_mean",
    "relative_to_rolling_mean",
    "distance_from_rolling_max",
    "distance_from_rolling_min",
    "multiply",
]


def _build_feature_spec_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": SUPPORTED_EXTRA_FEATURE_KINDS,
            },
            "source": {"type": ["string", "null"]},
            "window": {"type": ["integer", "null"], "minimum": 2},
            "reference": {"type": ["string", "null"]},
            "left": {"type": ["string", "null"]},
            "right": {"type": ["string", "null"]},
        },
        "required": ["name", "kind", "source", "window", "reference", "left", "right"],
    }


def _build_feature_engineering_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": ["none", "replace_specs"]},
            "summary": {"type": "string"},
            "specs": {"type": "array", "items": _build_feature_spec_schema()},
        },
        "required": ["mode", "summary", "specs"],
    }


ANALYST_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": ["explore", "exploit", "prune", "stop"],
        },
        "run_intent": {
            "type": "string",
            "enum": ["diagnostic", "exploratory", "confirmatory", "pruning", "stop"],
        },
        "diagnosis": {"type": "string"},
        "reason": {"type": "string"},
        "focus": {"type": "string"},
        "hypothesis": {"type": "string"},
        "research_mode": {
            "type": "string",
            "enum": ["explore", "exploit", "prune", "stop"],
        },
        "alpha": {"type": "number"},
        "feature_groups": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": sorted(FEATURE_GROUPS.keys()),
            },
        },
        "feature_names": {
            "type": "array",
            "items": {"type": "string"},
        },
        "feature_engineering": _build_feature_engineering_schema(),
        "notes": {"type": "string"},
    },
    "required": [
        "action",
        "run_intent",
        "diagnosis",
        "reason",
        "focus",
        "hypothesis",
        "research_mode",
        "alpha",
        "feature_groups",
        "feature_names",
        "feature_engineering",
        "notes",
    ],
}

CRITIC_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["approve", "revise", "reject", "stop"],
        },
        "summary": {"type": "string"},
        "revised_research_mode": {
            "type": "string",
            "enum": ["explore", "exploit", "prune", "stop"],
        },
        "revised_run_intent": {
            "type": "string",
            "enum": ["diagnostic", "exploratory", "confirmatory", "pruning", "stop"],
        },
        "revised_alpha": {"type": "number"},
        "revised_feature_groups": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": sorted(FEATURE_GROUPS.keys()),
            },
        },
        "revised_feature_names": {
            "type": "array",
            "items": {"type": "string"},
        },
        "revised_feature_engineering": _build_feature_engineering_schema(),
        "revised_notes": {"type": "string"},
    },
    "required": [
        "verdict",
        "summary",
        "revised_research_mode",
        "revised_run_intent",
        "revised_alpha",
        "revised_feature_groups",
        "revised_feature_names",
        "revised_feature_engineering",
        "revised_notes",
    ],
}


REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "decision": {"type": "string", "enum": ["keep", "reject", "inconclusive"]},
        "interpretation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommendation": {"type": "string"},
        "next_idea": {"type": "string"},
    },
    "required": ["summary", "decision", "interpretation", "confidence", "recommendation", "next_idea"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven research agent for ridge experiments.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of agent iterations to run.")
    parser.add_argument("--model", default=None, help="LLM model name. Defaults to config settings.")
    parser.add_argument("--dry-run", action="store_true", help="Print the proposal prompt and exit.")
    return parser.parse_args()


def read_program() -> str:
    return AGENT_PROGRAM_PATH.read_text(encoding="utf-8")


def load_feature_state() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("features_snapshot", FEATURES_PY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load feature state from {FEATURES_PY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "feature_columns": list(module.FEATURE_COLUMNS),
        "base_feature_columns": list(getattr(module, "BASE_FEATURE_COLUMNS", module.FEATURE_COLUMNS)),
        "extra_feature_specs": list(getattr(module, "EXTRA_FEATURE_SPECS", [])),
        "supported_extra_feature_kinds": list(getattr(module, "SUPPORTED_EXTRA_FEATURE_KINDS", [])),
    }


def load_history() -> pd.DataFrame:
    if not EXPERIMENTS_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENTS_CSV_PATH)


def summarize_history(history: pd.DataFrame, top_n: int = 5, recent_n: int = 5) -> dict[str, Any]:
    if history.empty:
        return {
            "best_runs": [],
            "recent_runs": [],
            "feature_group_counts": {},
            "alpha_counts": {},
            "recent_duplicate_configs": [],
            "recent_rejected_streak": 0,
            "recent_weak_improvement_streak": 0,
            "total_runs": 0,
        }

    base = history.copy()
    for column, default_value in {
        "research_mode": "legacy",
        "notes": "",
        "accepted": False,
        "feature_count": 0,
        "val_ic_sharpe": 0.0,
        "val_top_minus_bottom": 0.0,
        "delta_vs_best_val_ic": 0.0,
        "run_intent": "legacy",
    }.items():
        if column not in base.columns:
            base[column] = default_value

    recent_frame = base.copy()
    sortable = base.copy()
    sortable["config_key"] = sortable.apply(
        lambda row: f"alpha={row['alpha']}|features={row.get('features_json', '[]')}",
        axis=1,
    )
    sortable = sortable.sort_values("val_mean_rank_ic", ascending=False)
    best_runs = sortable.head(top_n)[
        [
            "run_name",
            "alpha",
            "feature_count",
            "val_mean_rank_ic",
            "val_ic_sharpe",
            "val_top_minus_bottom",
            "accepted",
            "research_mode",
            "run_intent",
            "notes",
        ]
    ].to_dict(orient="records")

    recent = recent_frame.tail(recent_n)[
        [
            "run_name",
            "alpha",
            "feature_count",
            "val_mean_rank_ic",
            "val_ic_sharpe",
            "val_top_minus_bottom",
            "accepted",
            "research_mode",
            "run_intent",
            "notes",
        ]
    ].to_dict(orient="records")

    def parse_features(raw: Any) -> list[str]:
        if pd.isna(raw):
            return []
        try:
            return list(json.loads(str(raw)))
        except json.JSONDecodeError:
            return []

    parsed_features = history.get("features_json", pd.Series(dtype=object)).apply(parse_features)
    feature_group_counts = {
        group_name: int(
            parsed_features.apply(
                lambda feature_list: any(feature in FEATURE_GROUPS[group_name] for feature in feature_list)
            ).sum()
        )
        for group_name in FEATURE_GROUPS
    }
    alpha_counts = history["alpha"].value_counts().sort_index().astype(int).to_dict()
    duplicate_counts = sortable["config_key"].value_counts()
    recent_duplicate_configs = duplicate_counts[duplicate_counts > 1].head(5).to_dict()
    accepted_rate = float(base["accepted"].astype(str).str.lower().eq("true").mean())
    recent_rejected_streak = 0
    for accepted_value in reversed(recent_frame["accepted"].astype(str).str.lower().tolist()):
        if accepted_value == "true":
            break
        recent_rejected_streak += 1

    recent_weak_improvement_streak = 0
    weak_mask = (
        recent_frame["delta_vs_best_val_ic"].fillna(0.0) <= 0
    ) & (~recent_frame["accepted"].astype(str).str.lower().eq("true"))
    for is_weak in reversed(weak_mask.tolist()):
        if not is_weak:
            break
        recent_weak_improvement_streak += 1

    return {
        "best_runs": best_runs,
        "recent_runs": recent,
        "feature_group_counts": feature_group_counts,
        "alpha_counts": alpha_counts,
        "recent_duplicate_configs": recent_duplicate_configs,
        "accepted_rate": accepted_rate,
        "recent_rejected_streak": recent_rejected_streak,
        "recent_weak_improvement_streak": recent_weak_improvement_streak,
        "total_runs": int(len(history)),
    }


def build_analyst_prompt(
    program_text: str,
    history_summary: dict[str, Any],
    feature_state: dict[str, Any],
    plateau_guidance: str | None = None,
) -> str:
    sections = [
        "You are the Analyst agent in a two-agent quant research process.",
        "Return only structured data that matches the required schema.",
        "Choose exactly one action: explore, exploit, prune, or stop.",
        "Start from diagnosis: identify the current weakness, instability, or opportunity before proposing a change.",
        "Treat this like a research notebook entry: diagnosis, hypothesis, small test, then notes.",
        "If you propose feature engineering, stay within the supported feature kinds shown in the current feature state and provide semantically complete specs.",
        "Do not invent unsupported feature families such as rank transforms unless they can be expressed through the supported feature kinds.",
        "If you choose stop, still fill the remaining fields with minimal valid placeholders and explain the stop decision in notes.",
    ]
    if plateau_guidance:
        sections.append(f"Plateau guidance:\n{plateau_guidance}")
    sections.extend(
        [
            f"Program:\n{program_text}",
            f"Current feature state:\n{json.dumps(feature_state, indent=2)}",
            f"Experiment history summary:\n{json.dumps(history_summary, indent=2)}",
        ]
    )
    return "\n\n".join(sections)


def build_critic_prompt(
    program_text: str,
    history_summary: dict[str, Any],
    analyst_plan: dict[str, Any],
    feature_state: dict[str, Any],
) -> str:
    feature_group_summary = {
        name: features for name, features in FEATURE_GROUPS.items()
    }
    return "\n\n".join(
        [
            "You are the Critic agent in a two-agent quant research process.",
            "Return only structured data that matches the required schema.",
            "Behave like a skeptical quant reviewer, not a deterministic grid-search script.",
            "Approve a plan only if it is controlled, novel enough, and aligned with the research contract.",
            "Revise a plan if the idea is directionally good but too large, too repetitive, or poorly justified.",
            "Reject or stop if the proposal is too noisy, redundant, or contract-violating.",
            "Pay close attention to diagnosis quality, novelty, and whether the proposal is actually testing a falsifiable idea.",
            f"Program:\n{program_text}",
            f"Current feature state:\n{json.dumps(feature_state, indent=2)}",
            f"Available feature groups:\n{json.dumps(feature_group_summary, indent=2)}",
            f"Experiment history summary:\n{json.dumps(history_summary, indent=2)}",
            f"Analyst plan:\n{json.dumps(analyst_plan, indent=2)}",
        ]
    )


def apply_critic_review(analyst_plan: dict[str, Any], critic_review: dict[str, Any]) -> dict[str, Any]:
    final_plan = dict(analyst_plan)
    verdict = critic_review["verdict"]

    if verdict in {"reject", "stop"}:
        final_plan["action"] = "stop"
        final_plan["research_mode"] = "stop"
        final_plan["run_intent"] = "stop"
        final_plan["reason"] = critic_review["summary"]
        final_plan["focus"] = "pause_and_summarize"
        final_plan["notes"] = critic_review["summary"]
        return final_plan

    if verdict == "revise":
        final_plan["action"] = critic_review["revised_research_mode"]
        final_plan["research_mode"] = critic_review["revised_research_mode"]
        final_plan["run_intent"] = critic_review["revised_run_intent"]
        final_plan["alpha"] = critic_review["revised_alpha"]
        final_plan["feature_groups"] = critic_review["revised_feature_groups"]
        final_plan["feature_names"] = critic_review["revised_feature_names"]
        final_plan["feature_engineering"] = critic_review["revised_feature_engineering"]
        final_plan["notes"] = critic_review["revised_notes"]
        final_plan["reason"] = critic_review["summary"]

    return final_plan


def build_reflection_prompt(program_text: str, proposal: dict[str, Any], result: dict[str, Any], history_summary: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Review the latest experiment objectively.",
            "Return only structured data that matches the required schema.",
            f"Program:\n{program_text}",
            f"Proposal:\n{json.dumps(proposal, indent=2)}",
            f"Result:\n{json.dumps(result, indent=2)}",
            f"History summary:\n{json.dumps(history_summary, indent=2)}",
            "Judge the experiment mainly by validation mean rank IC, and use validation IC Sharpe, novelty, and simplicity as secondary evidence. Do not use test results for research-loop decisions.",
            "Write the reflection like a short research memo: what was tested, what happened, how confident we should be, and what should happen next.",
        ]
    )


def _format_feature_specs_for_python(specs: list[dict[str, Any]]) -> str:
    return pformat(specs, width=100, sort_dicts=True)


def update_extra_feature_specs(specs: list[dict[str, Any]]) -> None:
    features_text = FEATURES_PY_PATH.read_text(encoding="utf-8")
    replacement = (
        "# BEGIN_AGENT_MANAGED_EXTRA_FEATURE_SPECS\n"
        f"EXTRA_FEATURE_SPECS: list[dict[str, object]] = {_format_feature_specs_for_python(specs)}\n"
        "# END_AGENT_MANAGED_EXTRA_FEATURE_SPECS"
    )
    new_text, count = re.subn(
        r"# BEGIN_AGENT_MANAGED_EXTRA_FEATURE_SPECS\n.*?\n# END_AGENT_MANAGED_EXTRA_FEATURE_SPECS",
        replacement,
        features_text,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Could not locate the managed extra feature specs block in src/features.py")
    FEATURES_PY_PATH.write_text(new_text, encoding="utf-8")


def normalize_feature_engineering_plan(plan: dict[str, Any], feature_state: dict[str, Any]) -> dict[str, Any]:
    feature_engineering = plan["feature_engineering"]
    if feature_engineering["mode"] == "none":
        return plan

    known_columns = set(feature_state["feature_columns"]) | {"close", "open", "high", "low", "volume", "ret_1d", "range_1d"}
    normalized_specs: list[dict[str, Any]] = []

    for raw_spec in feature_engineering["specs"]:
        spec = dict(raw_spec)
        kind = spec["kind"]
        name = spec["name"]

        if kind in {"pct_change", "ma_gap"}:
            spec["source"] = spec.get("source") or "close"
            spec["window"] = spec.get("window") or 10
        elif kind == "rolling_std":
            spec["source"] = spec.get("source") or "ret_1d"
            spec["window"] = spec.get("window") or 10
        elif kind in {"rolling_mean", "relative_to_rolling_mean"}:
            inferred_source = spec.get("source")
            if not inferred_source:
                if kind == "rolling_mean" and "_rolling_mean_" in name:
                    inferred_source = name.split("_rolling_mean_")[0]
                elif kind == "relative_to_rolling_mean" and "_relative_to_rolling_mean_" in name:
                    inferred_source = name.split("_relative_to_rolling_mean_")[0]
            spec["source"] = inferred_source
            spec["window"] = spec.get("window") or 10
        elif kind == "distance_from_rolling_max":
            spec["source"] = spec.get("source") or "high"
            spec["reference"] = spec.get("reference") or "close"
            spec["window"] = spec.get("window") or 10
        elif kind == "distance_from_rolling_min":
            spec["source"] = spec.get("source") or "low"
            spec["reference"] = spec.get("reference") or "close"
            spec["window"] = spec.get("window") or 10
        elif kind == "multiply":
            left = spec.get("left")
            right = spec.get("right")
            if (not left or not right) and "_x_" in name:
                maybe_left, maybe_right = name.split("_x_", 1)
                if not left and maybe_left in known_columns:
                    left = maybe_left
                if not right and maybe_right in known_columns:
                    right = maybe_right
            spec["left"] = left
            spec["right"] = right

        normalized_specs.append(spec)

    plan["feature_engineering"]["specs"] = normalized_specs
    return plan


def validate_feature_engineering_plan(plan: dict[str, Any], feature_state: dict[str, Any]) -> None:
    feature_engineering = plan["feature_engineering"]
    mode = feature_engineering["mode"]
    if mode == "none":
        return

    supported = set(feature_state["supported_extra_feature_kinds"])
    known_columns = set(feature_state["feature_columns"]) | {"close", "open", "high", "low", "volume", "ret_1d", "range_1d"}
    seen_names: set[str] = set()
    for spec in feature_engineering["specs"]:
        name = spec["name"].strip()
        kind = spec["kind"]
        if not name:
            raise ValueError("Feature engineering spec has an empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate proposed feature name: {name}")
        if kind not in supported:
            raise ValueError(f"Unsupported feature kind proposed: {kind}")
        seen_names.add(name)

        if kind in {"pct_change", "ma_gap", "rolling_std", "rolling_mean", "relative_to_rolling_mean"}:
            default_source = "close" if kind in {"pct_change", "ma_gap"} else "ret_1d"
            if kind in {"rolling_mean", "relative_to_rolling_mean"}:
                default_source = ""
            source = spec.get("source") or default_source
            if not source or source not in known_columns:
                raise ValueError(f"Unknown source column for feature {name}: {source}")
            if kind == "pct_change":
                safe_pct_change_sources = {
                    "close",
                    "open",
                    "high",
                    "low",
                    "volume",
                    "dollar_volume",
                    "log_dollar_volume",
                }
                if source not in safe_pct_change_sources:
                    raise ValueError(
                        f"Unsupported pct_change source for feature {name}: {source}. "
                        "pct_change is only allowed on raw price/volume-like columns to avoid unstable divisions."
                    )
            if int(spec.get("window", 10) or 10) < 2:
                raise ValueError(f"Invalid rolling window for feature {name}")
        elif kind in {"distance_from_rolling_max", "distance_from_rolling_min"}:
            default_source = "high" if kind == "distance_from_rolling_max" else "low"
            source = spec.get("source") or default_source
            reference = spec.get("reference") or "close"
            if source not in known_columns or reference not in known_columns:
                raise ValueError(f"Unknown source/reference for feature {name}")
            if int(spec.get("window", 10) or 10) < 2:
                raise ValueError(f"Invalid rolling window for feature {name}")
        elif kind == "multiply":
            left = spec.get("left") or ""
            right = spec.get("right") or ""
            if left not in known_columns or right not in known_columns:
                raise ValueError(f"Unknown multiply inputs for feature {name}")

    explicit_features = plan.get("feature_names", [])
    if explicit_features:
        future_columns = known_columns | seen_names | {spec["name"] for spec in feature_engineering["specs"]}
        unknown_features = sorted(set(explicit_features) - future_columns)
        if unknown_features:
            raise ValueError(f"Unknown explicit features requested: {unknown_features}")


def make_proposal_runnable_after_feature_error(
    proposal: dict[str, Any],
    feature_state: dict[str, Any],
    error_message: str,
) -> dict[str, Any]:
    adjusted = json.loads(json.dumps(proposal))
    proposed_spec_names = {
        str(spec.get("name", "")).strip()
        for spec in adjusted.get("feature_engineering", {}).get("specs", [])
        if str(spec.get("name", "")).strip()
    }
    valid_existing_features = set(feature_state["feature_columns"])

    adjusted["feature_engineering"] = {
        "mode": "none",
        "summary": f"Dropped invalid feature-engineering proposal: {error_message}",
        "specs": [],
    }
    adjusted["feature_names"] = [
        feature
        for feature in adjusted.get("feature_names", [])
        if feature in valid_existing_features and feature not in proposed_spec_names
    ]
    adjusted["notes"] = (
        f"{adjusted.get('notes', '')} [feature proposal dropped: {error_message}]"
    ).strip()
    return adjusted


def make_proposal_runnable_after_runner_error(
    proposal: dict[str, Any],
    error_message: str,
) -> dict[str, Any]:
    adjusted = json.loads(json.dumps(proposal))
    adjusted["feature_engineering"] = {
        "mode": "none",
        "summary": f"Dropped feature-engineering after runner failure: {error_message}",
        "specs": [],
    }
    adjusted["feature_names"] = []
    adjusted["notes"] = (
        f"{adjusted.get('notes', '')} [runner fallback: {error_message}]"
    ).strip()
    return adjusted


def refresh_processed_dataset() -> None:
    cmd = [sys.executable, "src/prepare.py"]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if completed.stdout:
        print(completed.stdout.strip(), file=sys.stderr)


def restore_previous_feature_state(previous_extra_specs: list[dict[str, Any]]) -> None:
    update_extra_feature_specs(previous_extra_specs)
    refresh_processed_dataset()


def run_experiment(proposal: dict[str, Any], run_name: str) -> dict[str, Any]:
    feature_state = load_feature_state()
    previous_extra_specs = list(feature_state.get("extra_feature_specs", []))
    applied_new_specs = False
    proposal = normalize_feature_engineering_plan(proposal, feature_state)
    try:
        validate_feature_engineering_plan(proposal, feature_state)
    except ValueError as exc:
        proposal = make_proposal_runnable_after_feature_error(proposal, feature_state, str(exc))
        validate_feature_engineering_plan(proposal, feature_state)

    if proposal["feature_engineering"]["mode"] == "replace_specs":
        update_extra_feature_specs(proposal["feature_engineering"]["specs"])
        refresh_processed_dataset()
        applied_new_specs = True

    try:
        cmd = [
            sys.executable,
            "src/experiment_runner.py",
            "--run-name",
            run_name,
            "--alpha",
            str(proposal["alpha"]),
            "--research-mode",
            proposal["research_mode"],
            "--notes",
            proposal["notes"],
        ]
        if proposal["feature_names"]:
            cmd.extend(["--features", ",".join(proposal["feature_names"])])
        else:
            cmd.extend(["--feature-groups", ",".join(proposal["feature_groups"])])

        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(completed.stdout)
        except subprocess.CalledProcessError as exc:
            error_text = (exc.stderr or exc.stdout or "").strip()
            if applied_new_specs:
                restore_previous_feature_state(previous_extra_specs)
                applied_new_specs = False

            fallback_proposal = make_proposal_runnable_after_runner_error(proposal, error_text)
            fallback_cmd = [
                sys.executable,
                "src/experiment_runner.py",
                "--run-name",
                run_name,
                "--alpha",
                str(fallback_proposal["alpha"]),
                "--research-mode",
                fallback_proposal["research_mode"],
                "--notes",
                fallback_proposal["notes"],
                "--feature-groups",
                ",".join(fallback_proposal["feature_groups"]),
            ]
            try:
                completed = subprocess.run(
                    fallback_cmd,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return json.loads(completed.stdout)
            except subprocess.CalledProcessError as fallback_exc:
                fallback_error = (fallback_exc.stderr or fallback_exc.stdout or "").strip()
                raise RuntimeError(
                    "Experiment runner failed for both the original proposal and the fallback proposal.\n"
                    f"Original error:\n{error_text}\n\nFallback error:\n{fallback_error}"
                ) from fallback_exc
    finally:
        if applied_new_specs:
            restore_previous_feature_state(previous_extra_specs)


def save_agent_report(
    run_name: str,
    analyst_plan: dict[str, Any],
    critic_review: dict[str, Any],
    final_plan: dict[str, Any],
    result: dict[str, Any],
    reflection: dict[str, Any],
) -> Path:
    AGENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGENT_REPORTS_DIR / f"{run_name}.md"
    report = "\n".join(
        [
            f"# Agent Run {run_name}",
            "",
            "## Analyst Plan",
            "",
            f"- Action: {analyst_plan['action']}",
            f"- Run intent: {analyst_plan['run_intent']}",
            f"- Diagnosis: {analyst_plan['diagnosis']}",
            f"- Reason: {analyst_plan['reason']}",
            f"- Focus: {analyst_plan['focus']}",
            "",
            f"- Hypothesis: {analyst_plan['hypothesis']}",
            f"- Research mode: {analyst_plan['research_mode']}",
            f"- Alpha: {analyst_plan['alpha']}",
            f"- Feature groups: {', '.join(analyst_plan['feature_groups'])}",
            f"- Explicit features: {', '.join(analyst_plan['feature_names']) if analyst_plan['feature_names'] else '(none)'}",
            f"- Feature engineering mode: {analyst_plan['feature_engineering']['mode']}",
            f"- Feature engineering summary: {analyst_plan['feature_engineering']['summary']}",
            f"- Feature specs: {json.dumps(analyst_plan['feature_engineering']['specs'])}",
            f"- Notes: {analyst_plan['notes']}",
            "",
            "## Critic Review",
            "",
            f"- Verdict: {critic_review['verdict']}",
            f"- Summary: {critic_review['summary']}",
            f"- Revised mode: {critic_review['revised_research_mode']}",
            f"- Revised run intent: {critic_review['revised_run_intent']}",
            f"- Revised alpha: {critic_review['revised_alpha']}",
            f"- Revised feature groups: {', '.join(critic_review['revised_feature_groups'])}",
            f"- Revised explicit features: {', '.join(critic_review['revised_feature_names']) if critic_review['revised_feature_names'] else '(none)'}",
            f"- Revised feature engineering mode: {critic_review['revised_feature_engineering']['mode']}",
            f"- Revised feature engineering summary: {critic_review['revised_feature_engineering']['summary']}",
            f"- Revised feature specs: {json.dumps(critic_review['revised_feature_engineering']['specs'])}",
            f"- Revised notes: {critic_review['revised_notes']}",
            "",
            "## Final Approved Plan",
            "",
            f"- Action: {final_plan['action']}",
            f"- Run intent: {final_plan['run_intent']}",
            f"- Diagnosis: {final_plan['diagnosis']}",
            f"- Reason: {final_plan['reason']}",
            f"- Focus: {final_plan['focus']}",
            f"- Hypothesis: {final_plan['hypothesis']}",
            f"- Research mode: {final_plan['research_mode']}",
            f"- Alpha: {final_plan['alpha']}",
            f"- Feature groups: {', '.join(final_plan['feature_groups'])}",
            f"- Explicit features: {', '.join(final_plan['feature_names']) if final_plan['feature_names'] else '(none)'}",
            f"- Feature engineering mode: {final_plan['feature_engineering']['mode']}",
            f"- Feature engineering summary: {final_plan['feature_engineering']['summary']}",
            f"- Feature specs: {json.dumps(final_plan['feature_engineering']['specs'])}",
            f"- Notes: {final_plan['notes']}",
            "",
            "## Result",
            "",
            f"- Accepted by runner: {result['accepted']}",
            f"- Novel configuration: {result.get('is_novel_configuration', 'N/A')}",
            f"- Validation Mean Rank IC: {result['metrics']['val']['mean_rank_ic']:.6f}",
            f"- Validation IC Sharpe: {result['metrics']['val']['ic_sharpe']:.6f}",
            f"- Validation Top Minus Bottom: {result['metrics']['val']['top_minus_bottom']:.6f}",
            "",
            "## Research Memo",
            "",
            f"- Decision: {reflection['decision']}",
            f"- Summary: {reflection['summary']}",
            f"- Interpretation: {reflection['interpretation']}",
            f"- Confidence: {reflection['confidence']}",
            f"- Recommendation: {reflection['recommendation']}",
            f"- Next idea: {reflection['next_idea']}",
            "",
        ]
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path


def save_stop_report(decision: dict[str, Any]) -> Path:
    AGENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"agent_stop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_path = AGENT_REPORTS_DIR / f"{run_name}.md"
    report = "\n".join(
        [
            f"# Agent Stop Decision {run_name}",
            "",
            f"- Action: {decision['action']}",
            f"- Run intent: {decision.get('run_intent', 'stop')}",
            f"- Diagnosis: {decision.get('diagnosis', decision['reason'])}",
            f"- Reason: {decision['reason']}",
            f"- Focus: {decision['focus']}",
            "",
        ]
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path


def should_force_stop(history_summary: dict[str, Any], agent_settings: dict[str, Any]) -> str | None:
    hard_stop_after_experiments = int(agent_settings.get("hard_stop_after_experiments", 35))
    hard_stop_weak_improvement_streak = int(agent_settings.get("hard_stop_weak_improvement_streak", 10))
    hard_stop_rejected_streak = int(agent_settings.get("hard_stop_rejected_streak", 10))

    total_runs = int(history_summary.get("total_runs", 0))
    if total_runs < hard_stop_after_experiments:
        return None

    if history_summary.get("recent_weak_improvement_streak", 0) >= hard_stop_weak_improvement_streak:
        return (
            f"Recent runs show {hard_stop_weak_improvement_streak} consecutive weak or non-accepted outcomes "
            f"after {hard_stop_after_experiments} experiments. Stop and summarize before spending more budget."
        )
    if history_summary.get("recent_rejected_streak", 0) >= hard_stop_rejected_streak:
        return (
            f"Recent runs were repeatedly rejected or unaccepted for {hard_stop_rejected_streak} consecutive runs "
            f"after {hard_stop_after_experiments} experiments. Stop and reassess the research direction before continuing."
        )
    return None


def get_plateau_guidance(history_summary: dict[str, Any], agent_settings: dict[str, Any]) -> str | None:
    warmup_experiments = int(agent_settings.get("warmup_experiments", 20))
    plateau_escalation_streak = int(agent_settings.get("plateau_escalation_streak", 5))
    total_runs = int(history_summary.get("total_runs", 0))
    if total_runs < warmup_experiments:
        return None

    weak_streak = int(history_summary.get("recent_weak_improvement_streak", 0))
    rejected_streak = int(history_summary.get("recent_rejected_streak", 0))
    if weak_streak >= plateau_escalation_streak or rejected_streak >= plateau_escalation_streak:
        return (
            "The search appears plateaued after the warmup stage. Do not stop yet. "
            "Instead, consider a broader but still controlled move: replace a feature family, reset the active subset, "
            "or make a more meaningful regularization shift rather than another tiny local tweak."
        )
    return None


def apply_warmup_override(
    final_plan: dict[str, Any],
    history_summary: dict[str, Any],
    agent_settings: dict[str, Any],
) -> dict[str, Any]:
    warmup_experiments = int(agent_settings.get("warmup_experiments", 12))
    allow_agent_stop_after_experiments = int(
        agent_settings.get("allow_agent_stop_after_experiments", warmup_experiments)
    )
    total_runs = int(history_summary.get("total_runs", 0))
    if total_runs >= allow_agent_stop_after_experiments:
        return final_plan
    if final_plan.get("action") != "stop":
        return final_plan

    overridden = dict(final_plan)
    default_alpha = float(load_settings().get("model", {}).get("alpha", 1.0))
    has_existing_plan = bool(overridden.get("feature_groups") or overridden.get("feature_names"))

    overridden["action"] = "explore"
    overridden["run_intent"] = "exploratory"
    overridden["research_mode"] = "explore"
    overridden["diagnosis"] = (
        "Warmup override applied: stop decisions are suppressed until at least "
        f"{allow_agent_stop_after_experiments} completed experiments."
    )
    overridden["reason"] = (
        "Warmup override applied before experiment "
        f"{allow_agent_stop_after_experiments}; continue exploring rather than stopping early."
    )
    overridden["focus"] = overridden.get("focus") or "warmup_research"
    overridden["hypothesis"] = (
        overridden.get("hypothesis")
        or "Additional warmup experiments are needed before the loop is allowed to stop."
    )
    if not has_existing_plan:
        overridden["alpha"] = default_alpha
        overridden["feature_groups"] = ["returns", "trend", "volatility"]
        overridden["feature_names"] = []
        overridden["feature_engineering"] = {
            "mode": "none",
            "summary": "Warmup override fallback to a safe baseline experiment.",
            "specs": [],
        }
    overridden["notes"] = (
        f"{overridden.get('notes', '')} [warmup override: stop suppressed before {allow_agent_stop_after_experiments} completed experiments]"
    ).strip()
    return overridden


def refresh_plots() -> None:
    cmd = [sys.executable, "src/plot_experiments.py"]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(
            f"Warning: plot refresh failed after agent run.\n{completed.stderr or completed.stdout}",
            file=sys.stderr,
        )


def refresh_report() -> None:
    cmd = [sys.executable, "src/report.py"]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(
            f"Warning: report refresh failed after agent run.\n{completed.stderr or completed.stdout}",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    settings = load_settings()
    agent_settings = settings.get("agent", {})
    provider = agent_settings.get("provider", "anthropic")
    default_model = "gpt-4.1-mini" if provider == "openai" else "claude-3-7-sonnet-20250219"
    model = args.model or agent_settings.get("model", default_model)
    iterations = args.iterations or int(settings.get("agent", {}).get("max_iterations", 1))

    program_text = read_program()
    client = None

    for iteration_idx in range(iterations):
        history = load_history()
        history_summary = summarize_history(history)
        feature_state = load_feature_state()
        plateau_guidance = get_plateau_guidance(history_summary, agent_settings)
        analyst_prompt = build_analyst_prompt(program_text, history_summary, feature_state, plateau_guidance)

        if args.dry_run:
            print(f"--- Analyst Prompt For Iteration {iteration_idx + 1} ---")
            print(analyst_prompt)
            return

        forced_stop_reason = should_force_stop(history_summary, agent_settings)
        if forced_stop_reason is not None:
            final_plan = {
                "action": "stop",
                "run_intent": "stop",
                "diagnosis": forced_stop_reason,
                "reason": forced_stop_reason,
                "focus": "pause_and_summarize",
                "hypothesis": "No new experiment should be run until the current line of research is summarized.",
                "research_mode": "stop",
                "alpha": 0.0,
                "feature_groups": [],
                "feature_names": [],
                "feature_engineering": {
                    "mode": "none",
                    "summary": "No feature change because the loop is stopping for review.",
                    "specs": [],
                },
                "notes": forced_stop_reason,
            }
            report_path = save_stop_report(final_plan)
            refresh_report()
            print(json.dumps({"final_plan": final_plan, "report_path": str(report_path)}, indent=2))
            break

        if client is None:
            client = create_json_client(provider=provider, model=model)

        analyst_plan = client.create_json_response(
            instructions="You are the Analyst agent. Return only valid JSON and match the requested schema exactly.",
            user_input=analyst_prompt,
            schema=ANALYST_PLAN_SCHEMA,
            max_output_tokens=900,
        )

        critic_prompt = build_critic_prompt(program_text, history_summary, analyst_plan, feature_state)
        critic_review = client.create_json_response(
            instructions="You are the Critic agent. Return only valid JSON and match the requested schema exactly.",
            user_input=critic_prompt,
            schema=CRITIC_REVIEW_SCHEMA,
            max_output_tokens=700,
        )
        final_plan = apply_critic_review(analyst_plan, critic_review)
        final_plan = apply_warmup_override(final_plan, history_summary, agent_settings)

        if final_plan["action"] == "stop":
            report_path = save_stop_report(final_plan)
            refresh_report()
            print(
                json.dumps(
                    {
                        "analyst_plan": analyst_plan,
                        "critic_review": critic_review,
                        "final_plan": final_plan,
                        "report_path": str(report_path),
                    },
                    indent=2,
                )
            )
            break

        run_name = f"agent_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        result = run_experiment(final_plan, run_name=run_name)

        updated_history = load_history()
        updated_summary = summarize_history(updated_history)
        reflection_prompt = build_reflection_prompt(program_text, final_plan, result, updated_summary)
        reflection = client.create_json_response(
            instructions="You are the Critic agent reviewing the executed experiment. Return only valid JSON and match the requested schema exactly.",
            user_input=reflection_prompt,
            schema=REFLECTION_SCHEMA,
            max_output_tokens=800,
        )

        report_path = save_agent_report(run_name, analyst_plan, critic_review, final_plan, result, reflection)
        refresh_plots()
        refresh_report()
        print(
            json.dumps(
                {
                    "run_name": run_name,
                    "analyst_plan": analyst_plan,
                    "critic_review": critic_review,
                    "final_plan": final_plan,
                    "result": result,
                    "reflection": reflection,
                    "report_path": str(report_path),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
