# research_agent

`research_agent` is a minimal, reproducible research workflow for predicting **5-day forward returns** on **S&P 500 stocks** using **free daily OHLCV data**.

The goal is not to build a production trading system. The goal is to demonstrate an **agentic research loop**:

1. define a research question
2. prepare a clean dataset
3. train a simple baseline model
4. evaluate ranking quality
5. generate a short research report

## Project Scope

- Universe: current S&P 500 constituents, fetched from the Wikipedia constituent table at runtime, with Yahoo-style ticker normalization
- Data source: `yfinance`
- Frequency: daily
- Label: 5-day forward return
- Primary metrics:
  - mean rank IC
  - IC Sharpe
- Secondary metrics:
  - top-minus-bottom quintile spread
  - directional hit rate

## Repository Layout

```text
config/
  settings.yaml
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

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

```bash
python src/prepare.py
```

This downloads daily price history, fetches the current S&P 500 universe, computes features, builds the 5-day forward return target, and writes a processed dataset to [`data/processed/dataset.parquet`](/Users/yuqingdai/Documents/research_agent/data/processed/dataset.parquet).

### 4. Train the baseline model

```bash
python src/train.py
```

This trains a simple ridge regression model and writes prediction files and summary metrics.

### 5. Evaluate the predictions

```bash
python src/evaluate.py
```

This computes cross-sectional daily rank IC, IC Sharpe, hit rate, and top-minus-bottom spread.

### 6. Generate a markdown report

```bash
python src/report.py
```

This writes a short report to [`outputs/reports/latest_report.md`](/Users/yuqingdai/Documents/research_agent/outputs/reports/latest_report.md).

### 7. Run a logged experiment

```bash
python src/experiment_runner.py --alpha 3.0 --feature-groups returns,trend,volatility --notes "Try a tighter momentum and volatility feature set"
```

This runs one ridge experiment, logs it to `outputs/metrics/experiments.csv`, and saves experiment-specific metrics.

### 8. Plot experiment history

```bash
python src/plot_experiments.py
```

This generates experiment progress plots in `outputs/plots/`.

Current plot outputs include:

- `best_so_far_validation_ic.png`
- `alpha_vs_validation_ic.png`
- `accepted_runs_validation_ic.png`
- `validation_vs_test_ic.png`
- `feature_group_heatmap.png`

### 9. Run the LLM research agent

Set your API key first:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

Then run one or more agent iterations:

```bash
python src/research_agent.py --iterations 1
python src/research_agent.py --iterations 3
```

The agent will:

- read `agent/program.md`
- inspect `outputs/metrics/experiments.csv`
- propose the next ridge experiment with structured JSON
- run `src/experiment_runner.py`
- write an experiment memo to `outputs/reports/agent_runs/`

## Target Definition

For each stock `i` on date `t`, the target is:

```text
forward_5d_return(i, t) = Close(i, t+5) / Close(i, t) - 1
```

## Evaluation Logic

For each date `t`, we compute cross-sectional rank correlation across all stocks in the universe:

```text
IC_t = SpearmanCorr_i(pred(i, t), realized_forward_5d_return(i, t))
```

Then we summarize the daily IC series using:

```text
mean_ic = mean(IC_t)
ic_sharpe = mean(IC_t) / std(IC_t)
```

We also compute:

- directional hit rate
- average top quintile return
- average bottom quintile return
- top-minus-bottom spread

## Feature Selection And Alpha Tuning

The baseline training command reads active features from [`config/feature_selection.json`](/Users/yuqingdai/Documents/research_agent/config/feature_selection.json).

You can control experiments in three ways:

- edit `config/feature_selection.json`
- pass `--features` with a comma-separated list
- pass `--feature-groups` using groups defined in [`src/feature_config.py`](/Users/yuqingdai/Documents/research_agent/src/feature_config.py)

Example:

```bash
python src/experiment_runner.py --alpha 10.0 --features ret_5d,ret_20d,ma_gap_20,vol_ratio_5_20,rel_volume_20,spy_ret_5d
```

Each experiment appends one row to `outputs/metrics/experiments.csv` with:

- run name
- alpha
- feature count
- selected feature list
- validation and test metrics
- accepted or rejected status

## Notes

- This is a research demo, not investment advice.
- Free datasets are noisy and limited.
- The processed universe is static and does not attempt to solve index membership history.
- The model is intentionally simple so that the workflow is easy to understand and extend.

## Agent Workflow

The file [`agent/program.md`](/Users/yuqingdai/Documents/research_agent/agent/program.md) defines how an LLM-based research agent should operate in this repository.

The agent is expected to:

- propose one small experiment at a time
- avoid changing the evaluation contract
- log results and compare against the current baseline
- write concise research conclusions

The new experiment tooling is designed to support an `autoresearch`-style loop:

1. choose a feature subset
2. choose a ridge alpha
3. run one experiment
4. compare validation mean rank IC against the current best
5. update plots and notes

The LLM agent extends this by making the proposal and reflection steps automatic through the Anthropic Messages API.

## Next Extensions

- switch from absolute return to excess return vs `SPY`
- add sector-neutral or market-relative features
- add rolling retraining
- add a simple long-short backtest
- compare S&P 500 vs Nasdaq-100 universes
