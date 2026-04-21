from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CSV_PATH = ROOT / "outputs" / "metrics" / "experiments.csv"
OUTPUT_REPORTS_DIR = ROOT / "outputs" / "reports"


def format_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def main() -> None:
    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not EXPERIMENTS_CSV_PATH.exists():
        raise FileNotFoundError("No experiments.csv found. Run experiment_runner.py or research_agent.py first.")

    df = pd.read_csv(EXPERIMENTS_CSV_PATH)
    if df.empty:
        raise RuntimeError("experiments.csv exists but contains no rows.")

    if "created_at_utc" in df.columns:
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], format="mixed", utc=True, errors="coerce")

    best_idx = df["val_mean_rank_ic"].idxmax()
    best_row = df.loc[best_idx]
    latest_row = df.sort_values("created_at_utc", ascending=False).iloc[0] if "created_at_utc" in df.columns else df.iloc[-1]
    accepted_runs = int(df["accepted"].astype(str).str.lower().eq("true").sum()) if "accepted" in df.columns else 0

    report = f"""# Latest Research Report

## Objective

Run a controlled AI-agent research loop for improving 5-day forward return prediction on S&P 500 daily data.

## Experiment Summary

- Total experiments: {len(df)}
- Accepted runs: {accepted_runs}
- Latest run: {latest_row.get("run_name", "N/A")}
- Best run: {best_row.get("run_name", "N/A")}

## Best Validation Result

- Alpha: {format_metric(best_row.get("alpha"), 2)}
- Feature count: {int(best_row.get("feature_count")) if pd.notna(best_row.get("feature_count")) else "N/A"}
- Validation Mean Rank IC: {format_metric(best_row.get("val_mean_rank_ic"), 6)}
- Validation IC Sharpe: {format_metric(best_row.get("val_ic_sharpe"), 6)}
- Validation Top Minus Bottom Quintile Spread: {format_metric(best_row.get("val_top_minus_bottom"), 6)}
- Test Mean Rank IC: {format_metric(best_row.get("test_mean_rank_ic"), 6)}

## Latest Experiment

- Run: {latest_row.get("run_name", "N/A")}
- Alpha: {format_metric(latest_row.get("alpha"), 2)}
- Validation Mean Rank IC: {format_metric(latest_row.get("val_mean_rank_ic"), 6)}
- Validation IC Sharpe: {format_metric(latest_row.get("val_ic_sharpe"), 6)}
- Validation Top Minus Bottom Quintile Spread: {format_metric(latest_row.get("val_top_minus_bottom"), 6)}
- Test Mean Rank IC: {format_metric(latest_row.get("test_mean_rank_ic"), 6)}
- Notes: {latest_row.get("notes", "")}

## Interpretation

This report summarizes the current experiment log rather than a single baseline-only run.
Use the best validation result as the current research benchmark, and use the latest run to understand the most recent agent decision.
"""

    output_path = OUTPUT_REPORTS_DIR / "latest_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
