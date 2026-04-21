# research_agent

`research_agent` is a reproducible, agentic quant research workflow for predicting **5-day forward returns** on **S&P 500 stocks** using **free daily OHLCV data**.

The project is built around a simple research loop:

1. prepare a clean daily dataset
2. train a lightweight baseline model
3. evaluate ranking quality
4. log experiment results
5. let an LLM agent propose the next experiment
6. review everything in a dashboard

## Project Scope

- Universe: current S&P 500 constituents fetched at runtime from Wikipedia
- Data source: `yfinance`
- Frequency: daily
- Label: 5-day forward return
- Baseline model: ridge regression
- Primary metrics:
  - mean rank IC
  - IC Sharpe
- Secondary metrics:
  - top-minus-bottom quintile spread
  - hit rate

## Repository Layout

```text
config/
  settings.yaml
  feature_selection.json
src/
  universe.py
  prepare.py
  features.py
  feature_config.py
  agent_llm.py
  train.py
  evaluate.py
  experiment_runner.py
  plot_experiments.py
  research_agent.py
  dashboard.py
  report.py
agent/
  program.md
data/
  raw/
  processed/
outputs/
  metrics/
  plots/
  reports/
```

## Quick Start

### 1. Create and activate the project environment

```bash
cd /Users/yuqingdai/Documents/research_agent/dualitas_research_agent
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the dataset

```bash
python src/prepare.py
```

This step fetches the current S&P 500 universe, downloads daily price data, computes engineered features, builds the 5-day forward-return target, and writes [`data/processed/dataset.parquet`]

### 4. Train the baseline model

```bash
python src/train.py
```

This writes train, validation, and test predictions into `outputs/metrics/`.

### 5. Evaluate the model

```bash
python src/evaluate.py
```

This computes cross-sectional ranking metrics such as mean rank IC, IC Sharpe, hit rate, and top-minus-bottom spread.

### 6. Generate the text report

```bash
python src/report.py
```

This writes [`outputs/reports/latest_report.md`](https://github.com/yd124/research_agent/blob/main/outputs/reports/latest_report.md).

### 7. Run a logged experiment

```bash
python src/experiment_runner.py --alpha 3.0 --feature-groups returns,trend,volatility --notes "Try a tighter momentum and volatility feature set"
```

This appends a row to `outputs/metrics/experiments.csv` and saves run-specific artifacts.

### 8. Plot experiment history

```bash
python src/plot_experiments.py
```

Expected plot outputs include:

- `best_so_far_validation_ic.png`
- `alpha_vs_validation_ic.png`
- `accepted_runs_validation_ic.png`
- `validation_vs_test_ic.png`
- `feature_group_heatmap.png`

### 9. Run the LLM research agent

Set your API key first:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
export OPENAI_API_KEY="your_api_key_here"
```

Then run one or more iterations:

```bash
python src/research_agent.py --iterations 1
python src/research_agent.py --iterations 3
```

The agent will:

- read `agent/program.md`
- inspect `outputs/metrics/experiments.csv`
- propose the next ridge experiment in structured JSON
- run `src/experiment_runner.py`
- write a memo to `outputs/reports/agent_runs/`

### 10. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

The restored dashboard is designed for demos and review sessions. It includes:

- a top-level overview with best-run and latest-run summaries
- a visual explanation of the research workflow
- a plot viewer for saved experiment figures
- a feature-group usage view
- a run explorer with notes and agent reflections
- a sortable experiment log

## Dashboard Notes

The dashboard reads from these generated files when available:

- `outputs/metrics/experiments.csv`
- `outputs/plots/*.png`
- `outputs/reports/agent_runs/*.md`

If those files do not exist yet, the dashboard will still open and show empty-state guidance instead of failing.

## Target Definition

For each stock `i` on date `t`:

```text
forward_5d_return(i, t) = Close(i, t+5) / Close(i, t) - 1
```

## Evaluation Logic

For each date `t`, the project computes cross-sectional rank correlation across stocks:

```text
IC_t = SpearmanCorr_i(pred(i, t), realized_forward_5d_return(i, t))
```

Then it summarizes the IC time series using:

```text
mean_ic = mean(IC_t)
ic_sharpe = mean(IC_t) / std(IC_t)
```

It also reports:

- hit rate
- average top quintile return
- average bottom quintile return
- top-minus-bottom spread

## Feature Selection And Alpha Tuning

The baseline training commands read active features from [`config/feature_selection.json`](/Users/yuqingdai/Documents/research_agent/dualitas_research_agent/config/feature_selection.json).

You can control experiments in three ways:

- edit `config/feature_selection.json`
- pass `--features` with a comma-separated list
- pass `--feature-groups` using groups defined in [`src/feature_config.py`](/Users/yuqingdai/Documents/research_agent/dualitas_research_agent/src/feature_config.py)

Example:

```bash
python src/experiment_runner.py --alpha 10.0 --features ret_5d,ret_20d,ma_gap_20,vol_ratio_5_20,rel_volume_20,spy_ret_5d
```

## Current Caveats

- `prepare.py` needs live network access to fetch the S&P 500 constituent table and Yahoo Finance data.
- If Wikipedia or Yahoo Finance is unreachable, dataset preparation will fail before training begins.
- The project is a research demo, not investment advice.

## Agent Workflow

The file [`agent/program.md`](/Users/yuqingdai/Documents/research_agent/dualitas_research_agent/agent/program.md) defines how the LLM research agent should behave.

The agent is expected to:

- propose one small experiment at a time
- avoid changing the evaluation contract
- compare new ideas against the current baseline
- write concise research conclusions

## Next Extensions

- switch from absolute return to excess return vs `SPY`
- add sector-neutral or market-relative features
- add rolling retraining
- add a simple long-short backtest
- compare S&P 500 vs Nasdaq-100 universes
