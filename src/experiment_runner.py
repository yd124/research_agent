from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluate import evaluate_predictions_df
from experiment_utils import (
    OUTPUT_METRICS_DIR,
    fit_and_predict,
    load_dataset,
    load_settings,
    save_prediction_outputs,
    target_col_from_settings,
)
from feature_config import (
    ALL_FEATURE_COLUMNS,
    DEFAULT_FEATURE_CONFIG_PATH,
    FEATURE_GROUPS,
    load_active_features,
    validate_feature_list,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PLOTS_DIR = ROOT / "outputs" / "plots"
EXPERIMENTS_CSV_PATH = OUTPUT_METRICS_DIR / "experiments.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ridge experiment with selected features and alpha.")
    parser.add_argument("--run-name", default=None, help="Optional run name. Defaults to a timestamped experiment id.")
    parser.add_argument("--alpha", type=float, default=None, help="Ridge alpha. Defaults to settings.yaml.")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature list. If omitted, reads from config/feature_selection.json.",
    )
    parser.add_argument(
        "--feature-groups",
        default=None,
        help="Comma-separated feature groups to include. Available groups are returns, trend, volatility, volume_liquidity, interaction.",
    )
    parser.add_argument("--notes", default="", help="Short experiment hypothesis or notes.")
    return parser.parse_args()


def resolve_feature_cols(args: argparse.Namespace) -> list[str]:
    if args.features:
        return validate_feature_list([item.strip() for item in args.features.split(",") if item.strip()])

    if args.feature_groups:
        groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]
        feature_cols: list[str] = []
        for group in groups:
            if group not in FEATURE_GROUPS:
                raise ValueError(f"Unknown feature group: {group}")
            for feature in FEATURE_GROUPS[group]:
                if feature not in feature_cols:
                    feature_cols.append(feature)
        return validate_feature_list(feature_cols)

    return load_active_features(DEFAULT_FEATURE_CONFIG_PATH)


def load_experiment_history() -> pd.DataFrame:
    if not EXPERIMENTS_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENTS_CSV_PATH)


def append_experiment_row(row: dict) -> None:
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = EXPERIMENTS_CSV_PATH.exists()
    with EXPERIMENTS_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    settings = load_settings()
    dataset = load_dataset()

    feature_cols = resolve_feature_cols(args)
    alpha = args.alpha if args.alpha is not None else float(settings["model"]["alpha"])
    run_name = args.run_name or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    prediction_frames, summary = fit_and_predict(
        dataset=dataset,
        feature_cols=feature_cols,
        alpha=alpha,
        settings=settings,
    )
    save_prediction_outputs(prediction_frames, summary, run_name)

    target_col = target_col_from_settings(settings)
    metrics = {
        split_name: evaluate_predictions_df(frame, target_col=target_col)
        for split_name, frame in prediction_frames.items()
    }

    history = load_experiment_history()
    previous_best = history["val_mean_rank_ic"].max() if not history.empty else None
    accepted = bool(previous_best is None or metrics["val"]["mean_rank_ic"] > previous_best)

    run_created_at_utc = datetime.now(timezone.utc)
    row = {
        "run_name": run_name,
        "created_at_utc": run_created_at_utc.isoformat(),
        "alpha": alpha,
        "feature_count": len(feature_cols),
        "features_json": json.dumps(feature_cols),
        "notes": args.notes,
        "accepted": accepted,
        "train_mean_rank_ic": metrics["train"]["mean_rank_ic"],
        "train_ic_sharpe": metrics["train"]["ic_sharpe"],
        "val_mean_rank_ic": metrics["val"]["mean_rank_ic"],
        "val_ic_sharpe": metrics["val"]["ic_sharpe"],
        "val_top_minus_bottom": metrics["val"]["top_minus_bottom"],
        "test_mean_rank_ic": metrics["test"]["mean_rank_ic"],
        "test_ic_sharpe": metrics["test"]["ic_sharpe"],
        "test_top_minus_bottom": metrics["test"]["top_minus_bottom"],
        "universe": settings["universe"],
        "forward_days": settings["forward_days"],
        "train_end": settings["train_end"],
        "val_end": settings["val_end"],
    }
    append_experiment_row(row)

    with (OUTPUT_METRICS_DIR / f"evaluation_summary_{run_name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": str(run_name),
        "accepted": bool(accepted),
        "metrics": {
            split: {key: (value.item() if hasattr(value, "item") else value) for key, value in split_metrics.items()}
            for split, split_metrics in metrics.items()
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
