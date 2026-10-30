# Agent session report

Period: 2025-03-26 to 2026-03-10 (250 trading days). Paper trading on synthetic data.

| Metric | Agent | Equal-weight buy-and-hold |
|---|---:|---:|
| Total return | -4.09% | -21.42% |
| Sharpe (daily, annualised) | -0.92 | -1.75 |
| Max drawdown | -8.07% | -23.90% |
| Final equity | $95,910 | $78,579 |

**Excess return vs buy-and-hold: +17.33 percentage points.**

## Trades

- Closed trades: 17, win rate 29.4%, average win 1.57%, average loss -3.06%, profit factor 0.24
- Average exposure: 16.9% of equity
- Signals: 35, vetoed by the agent: 3, blocked by risk rules: 3, agent fallbacks: 0

## Notes

- Paper trading only: the broker is simulated and no live-trading path exists in the code.
- Decisions are made at the close and filled at the next open, with 5 bps slippage and 2 bps fees. The benchmark pays no costs.
- On synthetic data these numbers validate the pipeline; they are not evidence of real-world edge.
