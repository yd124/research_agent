from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for plotting. Install it with `pip install -r requirements.txt`."
    ) from exc

from feature_config import FEATURE_GROUPS


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CSV_PATH = ROOT / "outputs" / "metrics" / "experiments.csv"
OUTPUT_PLOTS_DIR = ROOT / "outputs" / "plots"


def load_experiments() -> pd.DataFrame:
    if not EXPERIMENTS_CSV_PATH.exists():
        raise FileNotFoundError("No experiments.csv found. Run experiment_runner.py first.")
    df = pd.read_csv(EXPERIMENTS_CSV_PATH)
    if "accepted" in df.columns:
        df["accepted"] = df["accepted"].astype(str).str.lower() == "true"
    return df


def parse_feature_json(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return []


def plot_best_so_far(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["experiment_number"] = np.arange(1, len(plot_df) + 1)
    plot_df["best_so_far_val_ic"] = plot_df["val_mean_rank_ic"].cummax()

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["experiment_number"], plot_df["val_mean_rank_ic"], marker="o", label="Validation Mean IC")
    plt.plot(plot_df["experiment_number"], plot_df["best_so_far_val_ic"], linewidth=2, label="Best So Far")
    plt.xlabel("Experiment Number")
    plt.ylabel("Validation Mean Rank IC")
    plt.title("Validation Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "best_so_far_validation_ic.png", dpi=160)
    plt.close()


def plot_alpha_scatter(df: pd.DataFrame) -> None:
    plot_df = df[df["alpha"] > 0].copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(
        np.log10(plot_df["alpha"]),
        plot_df["val_mean_rank_ic"],
        c=plot_df["feature_count"],
        cmap="viridis",
        alpha=0.8,
    )
    plt.xlabel("log10(alpha)")
    plt.ylabel("Validation Mean Rank IC")
    plt.title("Validation IC by Ridge Alpha")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Feature Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "alpha_vs_validation_ic.png", dpi=160)
    plt.close()


def plot_accepted_runs(df: pd.DataFrame) -> None:
    accepted_df = df[df["accepted"] == True].copy()
    if accepted_df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(accepted_df["run_name"], accepted_df["val_mean_rank_ic"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation Mean Rank IC")
    plt.title("Accepted Experiments")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "accepted_runs_validation_ic.png", dpi=160)
    plt.close()


def plot_val_vs_test_scatter(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["val_mean_rank_ic", "test_mean_rank_ic"]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        plot_df["val_mean_rank_ic"],
        plot_df["test_mean_rank_ic"],
        c=plot_df["feature_count"],
        cmap="plasma",
        alpha=0.8,
    )
    min_val = min(plot_df["val_mean_rank_ic"].min(), plot_df["test_mean_rank_ic"].min())
    max_val = max(plot_df["val_mean_rank_ic"].max(), plot_df["test_mean_rank_ic"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Validation Mean Rank IC")
    plt.ylabel("Test Mean Rank IC")
    plt.title("Validation vs Test Mean Rank IC")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Feature Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "validation_vs_test_ic.png", dpi=160)
    plt.close()


def plot_feature_group_heatmap(df: pd.DataFrame) -> None:
    if "features_json" not in df.columns or df.empty:
        return

    plot_df = df.copy()
    plot_df["parsed_features"] = plot_df["features_json"].apply(parse_feature_json)

    group_names = list(FEATURE_GROUPS.keys())
    matrix = np.zeros((len(plot_df), len(group_names)))
    group_feature_sets = {group: set(features) for group, features in FEATURE_GROUPS.items()}

    for row_idx, features in enumerate(plot_df["parsed_features"]):
        feature_set = set(features)
        for col_idx, group_name in enumerate(group_names):
            group_set = group_feature_sets[group_name]
            matrix[row_idx, col_idx] = float(len(feature_set.intersection(group_set)) > 0)

    height = max(4, 0.35 * len(plot_df))
    plt.figure(figsize=(10, height))
    plt.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(label="Group Used")
    plt.xticks(range(len(group_names)), group_names, rotation=30, ha="right")
    plt.yticks(range(len(plot_df)), plot_df["run_name"])
    plt.xlabel("Feature Group")
    plt.ylabel("Experiment")
    plt.title("Feature Group Usage Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "feature_group_heatmap.png", dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_experiments()
    if df.empty:
        raise RuntimeError("experiments.csv exists but contains no rows.")

    plot_best_so_far(df)
    plot_alpha_scatter(df)
    plot_accepted_runs(df)
    plot_val_vs_test_scatter(df)
    plot_feature_group_heatmap(df)

    print(f"Saved experiment plots to {OUTPUT_PLOTS_DIR}")


if __name__ == "__main__":
    main()
