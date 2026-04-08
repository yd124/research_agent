from __future__ import annotations

import json
from pathlib import Path

from feature_config import load_active_features
from experiment_utils import fit_and_predict, load_dataset, load_settings, save_prediction_outputs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_METRICS_DIR = ROOT / "outputs" / "metrics"


def main() -> None:
    settings = load_settings()
    dataset = load_dataset()
    feature_cols = load_active_features()
    alpha = float(settings["model"]["alpha"])

    prediction_frames, summary = fit_and_predict(
        dataset=dataset,
        feature_cols=feature_cols,
        alpha=alpha,
        settings=settings,
    )

    save_prediction_outputs(prediction_frames, summary, run_name="baseline")
    for split_name in ["train", "val", "test"]:
        prediction_frames[split_name].to_parquet(
            OUTPUT_METRICS_DIR / f"predictions_{split_name}.parquet",
            index=False,
        )
    with (OUTPUT_METRICS_DIR / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved baseline train/val/test prediction files to outputs/metrics/")


if __name__ == "__main__":
    main()
