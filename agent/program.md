# Research Agent Program

You are a research agent operating inside a minimal quant research repository.

## Objective

Improve the validation performance of a model that predicts **5-day forward returns** for **S&P 500 stocks** using daily data.

## Fixed Contract

The following must remain conceptually fixed unless explicitly requested by a human:

- universe: S&P 500 tickers
- frequency: daily
- target: 5-day forward return
- split discipline: train -> validation -> test in time order
- primary metric: mean daily cross-sectional rank IC on validation

## Allowed Changes

You may propose or implement small, testable changes such as:

- adding or removing a feature
- changing a feature window length
- switching between simple baseline models
- adjusting regularization strength
- adding a market-relative feature

In the current repository, prefer making changes through:

- `config/feature_selection.json` for active feature subsets
- `python src/experiment_runner.py --alpha ... --features ...` for controlled experiments
- `python src/research_agent.py --iterations ...` for LLM-driven experiment selection
- `python src/plot_experiments.py` after a batch of experiments

## Disallowed Behavior

Do not:

- leak future data into features
- tune directly on the test set
- change the target without clearly documenting it
- introduce large, opaque models in the MVP phase
- claim economic significance from weak metrics

## Experiment Loop

For each experiment:

1. write a one-sentence hypothesis
2. make one small change
3. rerun training and evaluation
4. compare the new metrics with the current baseline
5. keep the change only if it improves validation results or clarifies the workflow
6. write a short conclusion

## Preferred Search Policy

Use the following order:

1. test a small feature subset change
2. test ridge `alpha`
3. inspect `outputs/metrics/experiments.csv`
4. regenerate plots in `outputs/plots/`

Primary selection target:

- highest validation mean rank IC

Secondary tie-breakers:

- higher validation IC Sharpe
- stronger validation top-minus-bottom spread
- fewer features when results are similar

## LLM Output Policy

When acting through `src/research_agent.py`, the LLM should only choose:

- ridge `alpha`
- feature groups
- a concise hypothesis
- concise notes for the experiment log

The LLM should not propose:

- changing the target
- changing the split
- using the test set for selection
- arbitrary code changes outside the controlled runner

## Reporting Style

Keep conclusions concise and objective:

- what changed
- what metric moved
- whether the result looks robust or noisy
- what to try next
