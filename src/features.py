from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", group_keys=False)
    prev_close = grouped["close"].shift(1)

    df["ret_1d"] = grouped["close"].pct_change(1)
    df["ret_5d"] = grouped["close"].pct_change(5)
    df["ret_10d"] = grouped["close"].pct_change(10)
    df["ret_20d"] = grouped["close"].pct_change(20)
    df["ret_60d"] = grouped["close"].pct_change(60)
    df["ret_120d"] = grouped["close"].pct_change(120)

    df["overnight_ret_1d"] = df["open"] / prev_close - 1.0
    df["intraday_ret_1d"] = df["close"] / df["open"] - 1.0

    ma_10 = grouped["close"].transform(lambda s: s.rolling(10).mean())
    ma_20 = grouped["close"].transform(lambda s: s.rolling(20).mean())
    ma_60 = grouped["close"].transform(lambda s: s.rolling(60).mean())

    df["ma_gap_10"] = df["close"] / ma_10 - 1.0
    df["ma_gap_20"] = df["close"] / ma_20 - 1.0
    df["ma_gap_60"] = df["close"] / ma_60 - 1.0

    df["vol_5d"] = grouped["ret_1d"].transform(lambda s: s.rolling(5).std())
    df["vol_20d"] = grouped["ret_1d"].transform(lambda s: s.rolling(20).std())
    df["vol_ratio_5_20"] = df["vol_5d"] / df["vol_20d"]

    vol_mean_20 = grouped["volume"].transform(lambda s: s.rolling(20).mean())
    vol_std_20 = grouped["volume"].transform(lambda s: s.rolling(20).std())
    df["rel_volume_20"] = df["volume"] / vol_mean_20
    df["volume_zscore_20"] = (df["volume"] - vol_mean_20) / vol_std_20

    df["dollar_volume"] = df["close"] * df["volume"]
    df["log_dollar_volume"] = np.log(df["dollar_volume"].clip(lower=1.0))

    df["range_1d"] = (df["high"] - df["low"]) / df["close"]
    df["range_5d"] = grouped.apply(
        lambda g: ((g["high"] - g["low"]) / g["close"]).rolling(5).mean()
    ).reset_index(level=0, drop=True)
    df["range_20d"] = grouped["range_1d"].transform(lambda s: s.rolling(20).mean())

    rolling_high_20 = grouped["high"].transform(lambda s: s.rolling(20).max())
    rolling_low_20 = grouped["low"].transform(lambda s: s.rolling(20).min())
    rolling_high_60 = grouped["high"].transform(lambda s: s.rolling(60).max())
    df["dist_from_20d_high"] = df["close"] / rolling_high_20 - 1.0
    df["dist_from_20d_low"] = df["close"] / rolling_low_20 - 1.0
    df["dist_from_60d_high"] = df["close"] / rolling_high_60 - 1.0

    df["amihud_illiquidity_20"] = grouped.apply(
        lambda g: (g["ret_1d"].abs() / (g["close"] * g["volume"]).replace(0, np.nan)).rolling(20).mean()
    ).reset_index(level=0, drop=True)

    df["return_volume_pressure_5d"] = df["ret_5d"] * df["rel_volume_20"]
    df["return_volume_pressure_20d"] = df["ret_20d"] * df["rel_volume_20"]

    return df


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "overnight_ret_1d",
    "intraday_ret_1d",
    "ma_gap_10",
    "ma_gap_20",
    "ma_gap_60",
    "vol_5d",
    "vol_20d",
    "vol_ratio_5_20",
    "rel_volume_20",
    "volume_zscore_20",
    "log_dollar_volume",
    "range_1d",
    "range_5d",
    "range_20d",
    "dist_from_20d_high",
    "dist_from_20d_low",
    "dist_from_60d_high",
    "amihud_illiquidity_20",
    "return_volume_pressure_5d",
    "return_volume_pressure_20d",
]
