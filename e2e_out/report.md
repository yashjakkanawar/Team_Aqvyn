# Agent session report

Period: 2025-03-26 to 2025-07-29 (90 trading days). Paper trading on synthetic data.

| Metric | Agent | Equal-weight buy-and-hold |
|---|---:|---:|
| Total return | +3.56% | -7.85% |
| Sharpe (daily, annualised) | 1.66 | -1.37 |
| Max drawdown | -2.29% | -13.39% |
| Final equity | $103,557 | $92,146 |

**Excess return vs buy-and-hold: +11.41 percentage points.**

## Trades

- Closed trades: 2, win rate 50.0%, average win 0.04%, average loss -3.04%, profit factor 0.01
- Average exposure: 24.0% of equity
- Signals: 9, vetoed by the agent: 2, blocked by risk rules: 0, agent fallbacks: 0

## Notes

- Paper trading only: the broker is simulated and no live-trading path exists in the code.
- Decisions are made at the close and filled at the next open, with 5 bps slippage and 2 bps fees. The benchmark pays no costs.
- On synthetic data these numbers validate the pipeline; they are not evidence of real-world edge.
