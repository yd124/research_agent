from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_METRICS_DIR = ROOT / "outputs" / "metrics"
OUTPUT_REPORTS_DIR = ROOT / "outputs" / "reports"


def format_metric(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def main() -> None:
    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_METRICS_DIR / "evaluation_summary.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    val_metrics = metrics["val"]
    test_metrics = metrics["test"]

    report = f"""# Latest Research Report

## Objective

Train a simple baseline model to predict 5-day forward returns for Nasdaq-100 stocks using free daily data.

## Validation Summary

- Mean Rank IC: {format_metric(val_metrics["mean_rank_ic"])}
- IC Sharpe: {format_metric(val_metrics["ic_sharpe"])}
- Hit Rate: {format_metric(val_metrics["hit_rate"])}
- Top Minus Bottom Quintile Spread: {format_metric(val_metrics["top_minus_bottom"])}

## Test Summary

- Mean Rank IC: {format_metric(test_metrics["mean_rank_ic"])}
- IC Sharpe: {format_metric(test_metrics["ic_sharpe"])}
- Hit Rate: {format_metric(test_metrics["hit_rate"])}
- Top Minus Bottom Quintile Spread: {format_metric(test_metrics["top_minus_bottom"])}

## Interpretation

This baseline is intended as a reproducible starting point, not as evidence of production alpha.
If validation metrics are weak or unstable, the next step is to add one carefully chosen feature or simplify the universe and rerun the workflow.
"""

    output_path = OUTPUT_REPORTS_DIR / "latest_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
