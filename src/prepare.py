from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from features import FEATURE_COLUMNS, add_features
from universe import load_universe


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the fixed raw dataset and processed feature dataset.")
    parser.add_argument(
        "--refresh-raw",
        action="store_true",
        help="Redownload raw Yahoo data instead of reusing data/raw/prices.parquet.",
    )
    return parser.parse_args()


def load_settings() -> dict:
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_prices(tickers: list[str], start_date: str, end_date: str | None) -> pd.DataFrame:
    history = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames = []
    for ticker in tickers:
        if ticker not in history.columns.get_level_values(0):
            continue
        frame = history[ticker].copy()
        if frame.empty:
            continue
        frame.columns = [c.lower() for c in frame.columns]
        frame = frame.reset_index().rename(columns={"Date": "date", "date": "date"})
        frame["ticker"] = ticker
        frames.append(frame[["date", "ticker", "open", "high", "low", "close", "adj close", "volume"]])

    if not frames:
        raise RuntimeError("No price history was downloaded.")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"adj close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_target(df: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    grouped = df.groupby("ticker", group_keys=False)
    df[f"target_fwd_{forward_days}d"] = grouped["close"].shift(-forward_days) / df["close"] - 1.0
    return df


def add_benchmark_feature(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    benchmark_history = yf.download(
        tickers=[benchmark],
        start=df["date"].min().strftime("%Y-%m-%d"),
        end=None,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if hasattr(benchmark_history.columns, "nlevels") and benchmark_history.columns.nlevels > 1:
        benchmark_df = benchmark_history[benchmark].copy()
    else:
        benchmark_df = benchmark_history.copy()
    benchmark_df.columns = [c.lower() for c in benchmark_df.columns]
    benchmark_df = benchmark_df.reset_index().rename(columns={"Date": "date", "date": "date"})
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"]).dt.tz_localize(None)
    benchmark_df["spy_ret_5d"] = benchmark_df["close"].pct_change(5)
    return df.merge(benchmark_df[["date", "spy_ret_5d"]], on="date", how="left")


def main() -> None:
    args = parse_args()
    settings = load_settings()
    tickers = load_universe(settings["universe"])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_prices_path = RAW_DIR / "prices.parquet"
    if raw_prices_path.exists() and not args.refresh_raw:
        prices = pd.read_parquet(raw_prices_path)
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    else:
        prices = download_prices(
            tickers=tickers,
            start_date=settings["start_date"],
            end_date=settings["end_date"],
        )
        prices.to_parquet(raw_prices_path, index=False)

    dataset = add_features(prices)
    dataset = add_benchmark_feature(dataset, benchmark=settings["benchmark"])
    dataset = add_target(dataset, forward_days=settings["forward_days"])

    all_features = FEATURE_COLUMNS + ["spy_ret_5d"]
    dataset = dataset.dropna(subset=all_features + [f"target_fwd_{settings['forward_days']}d"])
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    dataset.to_parquet(PROCESSED_DIR / "dataset.parquet", index=False)

    print(f"Saved processed dataset with {len(dataset):,} rows to {PROCESSED_DIR / 'dataset.parquet'}")


if __name__ == "__main__":
    main()
