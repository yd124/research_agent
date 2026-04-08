from __future__ import annotations

import json
from pathlib import Path

from features import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEFAULT_FEATURE_CONFIG_PATH = CONFIG_DIR / "feature_selection.json"
BENCHMARK_FEATURE = "spy_ret_5d"
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + [BENCHMARK_FEATURE]

FEATURE_GROUPS = {
    "returns": [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "overnight_ret_1d",
        "intraday_ret_1d",
    ],
    "trend": [
        "ma_gap_10",
        "ma_gap_20",
        "ma_gap_60",
        "dist_from_20d_high",
        "dist_from_20d_low",
        "dist_from_60d_high",
    ],
    "volatility": [
        "vol_5d",
        "vol_20d",
        "vol_ratio_5_20",
        "range_1d",
        "range_5d",
        "range_20d",
    ],
    "volume_liquidity": [
        "rel_volume_20",
        "volume_zscore_20",
        "log_dollar_volume",
        "amihud_illiquidity_20",
    ],
    "interaction": [
        "return_volume_pressure_5d",
        "return_volume_pressure_20d",
        "spy_ret_5d",
    ],
}


def validate_feature_list(feature_cols: list[str]) -> list[str]:
    unknown = sorted(set(feature_cols) - set(ALL_FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown features requested: {unknown}")
    return feature_cols


def load_active_features(config_path: Path | None = None) -> list[str]:
    path = config_path or DEFAULT_FEATURE_CONFIG_PATH
    if not path.exists():
        return ALL_FEATURE_COLUMNS.copy()

    payload = json.loads(path.read_text(encoding="utf-8"))
    active_features = payload.get("active_features", ALL_FEATURE_COLUMNS)
    return validate_feature_list(list(active_features))


def save_active_features(feature_cols: list[str], config_path: Path | None = None) -> None:
    path = config_path or DEFAULT_FEATURE_CONFIG_PATH
    payload = {"active_features": validate_feature_list(feature_cols)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
