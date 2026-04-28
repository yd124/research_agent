from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_METRICS_DIR = ROOT / "outputs" / "metrics"


def load_settings() -> dict:
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "dataset.parquet")


def target_col_from_settings(settings: dict) -> str:
    return f"target_fwd_{settings['forward_days']}d"


def time_split(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = df["date"] <= pd.Timestamp(train_end)
    val_mask = (df["date"] > pd.Timestamp(train_end)) & (df["date"] <= pd.Timestamp(val_end))
    test_mask = df["date"] > pd.Timestamp(val_end)
    return df.loc[train_mask].copy(), df.loc[val_mask].copy(), df.loc[test_mask].copy()


def build_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def fit_and_predict(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    alpha: float,
    settings: dict,
) -> tuple[dict[str, pd.DataFrame], dict]:
    target_col = target_col_from_settings(settings)
    train_df, val_df, test_df = time_split(
        dataset,
        train_end=settings["train_end"],
        val_end=settings["val_end"],
    )

    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    cleaned_frames: dict[str, pd.DataFrame] = {}
    for split_name, frame in split_frames.items():
        cleaned = frame.copy()
        cleaned.loc[:, feature_cols] = cleaned[feature_cols].replace([np.inf, -np.inf], np.nan)
        cleaned_frames[split_name] = cleaned

    train_df = cleaned_frames["train"]
    val_df = cleaned_frames["val"]
    test_df = cleaned_frames["test"]

    model = build_model(alpha=alpha)
    model.fit(train_df[feature_cols], train_df[target_col])

    prediction_frames: dict[str, pd.DataFrame] = {}
    for split_name, frame in cleaned_frames.items():
        pred_frame = frame[["date", "ticker", target_col]].copy()
        pred_frame["prediction"] = model.predict(frame[feature_cols])
        prediction_frames[split_name] = pred_frame

    summary = {
        "rows": {split: int(len(frame)) for split, frame in cleaned_frames.items()},
        "features": feature_cols,
        "target": target_col,
        "model": {
            "type": "ridge",
            "alpha": alpha,
        },
    }
    return prediction_frames, summary


def save_prediction_outputs(prediction_frames: dict[str, pd.DataFrame], summary: dict, run_name: str) -> None:
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, frame in prediction_frames.items():
        frame.to_parquet(OUTPUT_METRICS_DIR / f"predictions_{split_name}_{run_name}.parquet", index=False)

    with (OUTPUT_METRICS_DIR / f"train_summary_{run_name}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
