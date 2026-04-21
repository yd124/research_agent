# PPT Instructions

## Goal

Use this deck to explain:

- what an AI agent is
- what this project is for
- how the research workflow works
- what feature sets were tested
- what the best agent run found
- what the results mean and what to do next

Keep the tone practical and demo-friendly. The audience should leave with two ideas:

1. this is an agent-assisted research loop, not just a one-off model
2. the system can propose, run, evaluate, and summarize experiments under a fixed contract

## Suggested Title

`Agentic Quant Research Workflow for 5-Day Return Prediction`

Subtitle:

`Using an AI agent to propose and evaluate controlled ridge-regression experiments on S&P 500 daily data`

## Slide Outline

### 1. Title

- Project name
- Your name
- Date
- One-line summary:
  `An AI agent runs a controlled research loop to improve validation rank-IC on 5-day forward returns.`

### 2. What Is an AI Agent?

Use simple language:

- An AI agent is a model that does more than answer once
- It follows a goal, reads context, chooses the next action, runs tools, and updates its plan
- In this project, the agent reads past experiment history, proposes the next experiment, runs it, reviews the result, and writes a memo

Good visual:

- `Context -> Agent -> Tool call -> Result -> Reflection -> Next action`

### 3. Why This Project Exists

Say what problem it solves:

- Quant research often creates many small experiment decisions
- Humans are slow at repeating the same loop consistently
- This project turns that loop into a reproducible agent workflow

Key point bullets:

- Predict 5-day forward returns for S&P 500 stocks
- Keep the research contract fixed
- Let the agent search small, testable changes
- Track results in a dashboard and experiment log

### 4. Project Scope and Fixed Contract

This is important because it shows the agent is constrained.

- Universe: S&P 500
- Frequency: daily
- Target: 5-day forward return
- Split discipline: train -> validation -> test in time order
- Primary metric: validation mean daily cross-sectional rank IC
- Baseline model: ridge regression

Suggested message:

`The agent is allowed to optimize within the rules, but it is not allowed to change the rules.`

### 5. End-to-End Research Workflow

Recommended slide structure:

1. `prepare.py`
   Build dataset from daily OHLCV and benchmark data
2. `research_agent.py`
   Read experiment history and choose next experiment
3. `experiment_runner.py`
   Train and evaluate the chosen configuration
4. `plot_experiments.py`
   Refresh visual summaries
5. `dashboard.py`
   Review results, best runs, and agent reflections

Suggested visual:

- A left-to-right workflow diagram with the five boxes above

### 6. What the Agent Actually Changes

Keep this concrete:

- Ridge `alpha`
- Feature groups
- Small feature subset choices
- Short notes and hypotheses

And what it does not change:

- target definition
- split dates
- test-set selection policy
- model family beyond the allowed baseline workflow

### 7. Feature Sets

Present the feature groups clearly:

- `returns`
  short, medium, and long-horizon returns
- `trend`
  moving-average gaps and distance from highs/lows
- `volatility`
  realized volatility and trading-range features
- `volume_liquidity`
  relative volume, dollar volume, illiquidity
- `interaction`
  return-volume interaction terms and `spy_ret_5d`

Good presentation tip:

- show one table with `group`, `purpose`, `example features`

### 8. Evaluation Metrics

Explain the metrics briefly:

- Validation mean rank IC
  main selection metric
- IC Sharpe
  stability of the IC time series
- Top-minus-bottom spread
  directional spread between top and bottom predicted buckets
- Test mean rank IC
  out-of-sample reference, not selection target

Suggested message:

`Validation mean rank IC is the main optimization target. Test metrics are used as a check, not as a tuning target.`

### 9. Best Agent Run

Use the best current run from the experiment log:

- Run: `agent_20260406T045926Z`
- Alpha: `5.0`
- Feature count: `17`
- Validation mean rank IC: `0.016202`
- Test mean rank IC: `0.029052`

Interpretation:

- The strongest result came from `returns + volatility + interaction`
- Dropping the `trend` group improved the signal
- This suggests trend features were overlapping with returns and adding noise

### 10. What We Learned from the Search

This should sound like research, not hype.

- `trend` was not helpful in the best configuration
- `volatility` appears important
- `volume_liquidity` usually underperformed in this setup
- The best region for `alpha` was a plateau around `5` to `10`
- The agent was most useful when making small, single-dimension changes

### 11. Dashboard and Experiment Tracking

Show what the dashboard contributes:

- sortable experiment log
- best-run overview
- plot history over time
- agent reflection per run

Suggested demo note:

- open the dashboard and point at the best run, the latest runs, and the validation-vs-test plot

### 12. Risks and Limitations

This slide makes the project feel credible.

- best-effort historical backfill was reconstructed for older runs
- results depend on the current dataset and current codebase
- this is a research demo, not production trading logic
- no claim of economic alpha from weak or early-stage signals
- external data dependencies can change over time

### 13. Why the Agent Matters

This is the punchline.

- It automates the repetitive research loop
- It preserves structured experiment memory
- It keeps changes small and auditable
- It produces both metrics and narrative reflections
- It makes iteration faster without removing human oversight

### 14. Next Steps

Good options:

- stricter feature subset search within the winning 17-feature region
- sector-relative or market-relative features
- rolling retraining
- excess-return target versus SPY
- cleaner auto-reporting and slide export

## Presenter Notes

### Best one-sentence summary

`This project shows how an AI agent can operate as a disciplined research assistant: it proposes the next experiment, runs it under fixed rules, evaluates the result, and keeps the experiment history organized.`

### If someone asks “why not just use ChatGPT manually?”

Answer:

- manual prompting does not guarantee consistent experiment logging
- manual workflows are weaker at repeatability
- this setup makes the research loop structured, auditable, and tool-driven

### If someone asks “what is the real contribution?”

Answer:

- the contribution is the agentic research process and decision loop
- the model itself is intentionally simple
- the value is in faster, more disciplined iteration under a fixed contract

## Optional Demo Flow

If you want to demo live:

1. Show the experiment log in `outputs/metrics/experiments.csv`
2. Show the dashboard
3. Show one agent run memo in `outputs/reports/agent_runs/`
4. Explain the best run and why it won
5. Explain what the agent would test next
