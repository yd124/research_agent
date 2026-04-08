from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
OUTPUT_METRICS_DIR = ROOT / "outputs" / "metrics"


def load_settings() -> dict:
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def daily_rank_ic(df: pd.DataFrame, target_col: str) -> pd.Series:
    values = []
    index = []
    for date, group in df[["date", "prediction", target_col]].groupby("date", sort=True):
        valid = group.dropna()
        if len(valid) < 5:
            values.append(np.nan)
        else:
            corr = spearmanr(valid["prediction"], valid[target_col]).statistic
            values.append(float(corr) if corr == corr else np.nan)
        index.append(date)
    return pd.Series(values, index=index)


def quantile_spread(df: pd.DataFrame, target_col: str) -> dict[str, float]:
    top_returns = []
    bottom_returns = []

    for _, group in df.groupby("date", sort=True):
        valid = group[["prediction", target_col]].dropna().copy()
        if len(valid) < 10:
            continue
        valid["bucket"] = pd.qcut(valid["prediction"], q=5, labels=False, duplicates="drop")
        if valid["bucket"].nunique() < 5:
            continue
        top_returns.append(valid.loc[valid["bucket"] == valid["bucket"].max(), target_col].mean())
        bottom_returns.append(valid.loc[valid["bucket"] == valid["bucket"].min(), target_col].mean())

    top_mean = float(np.mean(top_returns)) if top_returns else np.nan
    bottom_mean = float(np.mean(bottom_returns)) if bottom_returns else np.nan
    return {
        "top_quintile_return": top_mean,
        "bottom_quintile_return": bottom_mean,
        "top_minus_bottom": top_mean - bottom_mean if top_returns and bottom_returns else np.nan,
    }


def evaluate_predictions_df(df: pd.DataFrame, target_col: str) -> dict:
    ic_series = daily_rank_ic(df, target_col=target_col).dropna()
    mean_ic = float(ic_series.mean()) if not ic_series.empty else np.nan
    std_ic = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan
    ic_sharpe = float(mean_ic / std_ic) if std_ic and std_ic == std_ic and std_ic != 0 else np.nan

    hit_rate = float((np.sign(df["prediction"]) == np.sign(df[target_col])).mean())
    spread_stats = quantile_spread(df, target_col=target_col)

    return {
        "num_rows": int(len(df)),
        "num_dates": int(df["date"].nunique()),
        "mean_rank_ic": mean_ic,
        "std_rank_ic": std_ic,
        "ic_sharpe": ic_sharpe,
        "hit_rate": hit_rate,
        **spread_stats,
    }


def evaluate_split(split_name: str, settings: dict) -> dict:
    target_col = f"target_fwd_{settings['forward_days']}d"
    df = pd.read_parquet(OUTPUT_METRICS_DIR / f"predictions_{split_name}.parquet")
    metrics = evaluate_predictions_df(df, target_col=target_col)
    return {"split": split_name, **metrics}


def main() -> None:
    settings = load_settings()
    results = {
        split: evaluate_split(split, settings)
        for split in ["train", "val", "test"]
    }

    with (OUTPUT_METRICS_DIR / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
