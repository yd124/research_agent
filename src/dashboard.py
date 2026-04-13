from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CSV_PATH = ROOT / "outputs" / "metrics" / "experiments.csv"
PLOTS_DIR = ROOT / "outputs" / "plots"
AGENT_REPORTS_DIR = ROOT / "outputs" / "reports" / "agent_runs"


PLOT_FILES = {
    "Best-So-Far Validation IC": "best_so_far_validation_ic.png",
    "Alpha vs Validation IC": "alpha_vs_validation_ic.png",
    "Accepted Runs": "accepted_runs_validation_ic.png",
    "Validation vs Test IC": "validation_vs_test_ic.png",
    "Feature Group Heatmap": "feature_group_heatmap.png",
}


def load_experiments() -> pd.DataFrame:
    if not EXPERIMENTS_CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(EXPERIMENTS_CSV_PATH)
    if "accepted" in df.columns:
        df["accepted"] = df["accepted"].astype(str).str.lower() == "true"
    if "created_at_utc" in df.columns:
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce")
    return df


def parse_features(features_json: str) -> list[str]:
    try:
        parsed = json.loads(features_json)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        return []
    return []


def infer_feature_groups(features: list[str]) -> list[str]:
    from feature_config import FEATURE_GROUPS

    feature_set = set(features)
    groups = []
    for group_name, group_features in FEATURE_GROUPS.items():
        if feature_set.intersection(group_features):
            groups.append(group_name)
    return groups


def load_agent_report(run_name: str) -> str | None:
    path = AGENT_REPORTS_DIR / f"{run_name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def show_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    if df.empty:
        st.warning("No experiments found yet. Run `python src/experiment_runner.py ...` or `python src/research_agent.py --iterations 1` first.")
        return

    best_idx = df["val_mean_rank_ic"].idxmax()
    best_row = df.loc[best_idx]
    accepted_runs = int(df["accepted"].sum()) if "accepted" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Experiments", len(df))
    col2.metric("Accepted Runs", accepted_runs)
    col3.metric("Best Validation IC", f"{best_row['val_mean_rank_ic']:.5f}")
    col4.metric("Best Alpha", f"{best_row['alpha']:.2f}")

    st.markdown(
        f"""
**Current best run:** `{best_row['run_name']}`  
**Feature count:** {int(best_row['feature_count'])}  
**Validation IC Sharpe:** {best_row['val_ic_sharpe']:.5f}  
**Validation Top-Bottom Spread:** {best_row['val_top_minus_bottom']:.5f}
"""
    )


def show_flow() -> None:
    st.subheader("Research Flow")
    st.markdown(
        """
1. `prepare.py` builds the daily S&P 500 dataset and 5-day forward-return labels.
2. `research_agent.py` reads the experiment history and proposes the next experiment.
3. `experiment_runner.py` runs Ridge regression with a chosen `alpha` and feature-group subset.
4. `evaluate.py` computes validation/test ranking metrics.
5. Results are logged to `outputs/metrics/experiments.csv`.
6. Agent reflections are written to `outputs/reports/agent_runs/`.
7. `plot_experiments.py` updates the experiment visualizations.
"""
    )


def show_plots() -> None:
    st.subheader("Visualizations")
    available = {label: path for label, filename in PLOT_FILES.items() if (path := PLOTS_DIR / filename).exists()}
    if not available:
        st.info("No plots found yet. Run `python src/plot_experiments.py` first.")
        return

    plot_label = st.selectbox("Select plot", list(available.keys()))
    st.image(str(available[plot_label]), use_container_width=True)


def show_experiment_table(df: pd.DataFrame) -> None:
    st.subheader("Experiment Log")
    if df.empty:
        return

    display_df = df.copy()
    display_df["feature_groups"] = display_df["features_json"].apply(lambda x: ", ".join(infer_feature_groups(parse_features(x))))
    display_df["notes"] = display_df["notes"].fillna("").astype(str).str.slice(0, 120)
    columns = [
        "run_name",
        "created_at_utc",
        "alpha",
        "feature_count",
        "feature_groups",
        "val_mean_rank_ic",
        "val_ic_sharpe",
        "test_mean_rank_ic",
        "accepted",
        "notes",
    ]
    st.dataframe(display_df[columns].sort_values("created_at_utc", ascending=False), use_container_width=True)


def show_run_detail(df: pd.DataFrame) -> None:
    st.subheader("Run Detail")
    if df.empty:
        return

    run_name = st.selectbox("Select run", df.sort_values("created_at_utc", ascending=False)["run_name"].tolist())
    row = df[df["run_name"] == run_name].iloc[0]
    features = parse_features(row["features_json"])
    feature_groups = infer_feature_groups(features)

    st.markdown(
        f"""
**Run:** `{row['run_name']}`  
**Alpha:** {row['alpha']:.2f}  
**Accepted:** {bool(row['accepted'])}  
**Validation IC:** {row['val_mean_rank_ic']:.5f}  
**Validation IC Sharpe:** {row['val_ic_sharpe']:.5f}  
**Test IC:** {row['test_mean_rank_ic']:.5f}  
**Feature Groups:** {", ".join(feature_groups) if feature_groups else "N/A"}
"""
    )

    st.markdown("**Notes**")
    st.write(str(row.get("notes", "")))

    st.markdown("**Features**")
    st.code(", ".join(features) if features else "No features parsed.")

    report_text = load_agent_report(run_name)
    if report_text:
        st.markdown("**Agent Reflection**")
        st.markdown(report_text)


def main() -> None:
    st.set_page_config(page_title="research_agent dashboard", layout="wide")
    st.title("research_agent dashboard")
    st.caption("A lightweight dashboard for showcasing the agentic research workflow, iterations, and results.")

    df = load_experiments()

    show_overview(df)
    show_flow()

    left, right = st.columns([1.15, 1.0])
    with left:
        show_plots()
    with right:
        show_run_detail(df)

    show_experiment_table(df)


if __name__ == "__main__":
    main()
