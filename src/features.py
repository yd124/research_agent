from __future__ import annotations

import numpy as np
import pandas as pd


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

# BEGIN_AGENT_MANAGED_EXTRA_FEATURE_SPECS
EXTRA_FEATURE_SPECS: list[dict[str, object]] = []
# END_AGENT_MANAGED_EXTRA_FEATURE_SPECS


def _validate_extra_feature_specs(specs: list[dict[str, object]]) -> None:
    seen_names: set[str] = set()
    for spec in specs:
        name = str(spec.get("name", "")).strip()
        kind = str(spec.get("kind", "")).strip()
        if not name:
            raise ValueError("Extra feature spec is missing a non-empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate extra feature name: {name}")
        if kind not in SUPPORTED_EXTRA_FEATURE_KINDS:
            raise ValueError(f"Unsupported extra feature kind: {kind}")
        seen_names.add(name)


def _apply_extra_feature_specs(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    _validate_extra_feature_specs(EXTRA_FEATURE_SPECS)

    for spec in EXTRA_FEATURE_SPECS:
        name = str(spec["name"])
        kind = str(spec["kind"])

        if kind == "pct_change":
            source = str(spec.get("source", "close"))
            window = int(spec.get("window", 5))
            df[name] = grouped[source].pct_change(window, fill_method=None)
        elif kind == "ma_gap":
            source = str(spec.get("source", "close"))
            window = int(spec.get("window", 10))
            ma = grouped[source].transform(lambda s: s.rolling(window).mean())
            df[name] = df[source] / ma - 1.0
        elif kind == "rolling_std":
            source = str(spec.get("source", "ret_1d"))
            window = int(spec.get("window", 10))
            df[name] = grouped[source].transform(lambda s: s.rolling(window).std())
        elif kind == "rolling_mean":
            source = str(spec["source"])
            window = int(spec.get("window", 10))
            df[name] = grouped[source].transform(lambda s: s.rolling(window).mean())
        elif kind == "relative_to_rolling_mean":
            source = str(spec["source"])
            window = int(spec.get("window", 10))
            mean = grouped[source].transform(lambda s: s.rolling(window).mean())
            df[name] = df[source] / mean
        elif kind == "distance_from_rolling_max":
            source = str(spec.get("source", "high"))
            reference = str(spec.get("reference", "close"))
            window = int(spec.get("window", 10))
            rolling_max = grouped[source].transform(lambda s: s.rolling(window).max())
            df[name] = df[reference] / rolling_max - 1.0
        elif kind == "distance_from_rolling_min":
            source = str(spec.get("source", "low"))
            reference = str(spec.get("reference", "close"))
            window = int(spec.get("window", 10))
            rolling_min = grouped[source].transform(lambda s: s.rolling(window).min())
            df[name] = df[reference] / rolling_min - 1.0
        elif kind == "multiply":
            left = str(spec["left"])
            right = str(spec["right"])
            df[name] = df[left] * df[right]

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", group_keys=False)
    prev_close = grouped["close"].shift(1)

    df["ret_1d"] = grouped["close"].pct_change(1, fill_method=None)
    df["ret_5d"] = grouped["close"].pct_change(5, fill_method=None)
    df["ret_10d"] = grouped["close"].pct_change(10, fill_method=None)
    df["ret_20d"] = grouped["close"].pct_change(20, fill_method=None)
    df["ret_60d"] = grouped["close"].pct_change(60, fill_method=None)
    df["ret_120d"] = grouped["close"].pct_change(120, fill_method=None)

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
    , include_groups=False).reset_index(level=0, drop=True)
    df["range_20d"] = grouped["range_1d"].transform(lambda s: s.rolling(20).mean())

    rolling_high_20 = grouped["high"].transform(lambda s: s.rolling(20).max())
    rolling_low_20 = grouped["low"].transform(lambda s: s.rolling(20).min())
    rolling_high_60 = grouped["high"].transform(lambda s: s.rolling(60).max())
    df["dist_from_20d_high"] = df["close"] / rolling_high_20 - 1.0
    df["dist_from_20d_low"] = df["close"] / rolling_low_20 - 1.0
    df["dist_from_60d_high"] = df["close"] / rolling_high_60 - 1.0

    df["amihud_illiquidity_20"] = grouped.apply(
        lambda g: (g["ret_1d"].abs() / (g["close"] * g["volume"]).replace(0, np.nan)).rolling(20).mean()
    , include_groups=False).reset_index(level=0, drop=True)

    df["return_volume_pressure_5d"] = df["ret_5d"] * df["rel_volume_20"]
    df["return_volume_pressure_20d"] = df["ret_20d"] * df["rel_volume_20"]
    df = _apply_extra_feature_specs(df, grouped)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


BASE_FEATURE_COLUMNS = [
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

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [str(spec["name"]) for spec in EXTRA_FEATURE_SPECS]
