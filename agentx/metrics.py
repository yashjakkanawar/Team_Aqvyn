"""Performance metrics from an equity curve."""
from __future__ import annotations

import numpy as np


def curve_metrics(eq) -> dict:
    eq = np.asarray(eq, dtype=float)
    if len(eq) < 2:
        return {"total_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "final_equity": float(eq[-1]) if len(eq) else 0.0}
    rets = np.diff(eq) / eq[:-1]
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    return {
        "total_return_pct": round(float((eq[-1] / eq[0] - 1) * 100), 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(float(((eq - peak) / peak).min() * 100), 2),
        "final_equity": round(float(eq[-1]), 2),
    }


def trade_stats(fills) -> dict:
    closed = [f for f in fills if f.side == "sell" and f.pnl is not None]
    if not closed:
        return {"closed_trades": 0, "win_rate_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "profit_factor": 0.0}
    wins = [f for f in closed if f.pnl > 0]
    losses = [f for f in closed if f.pnl <= 0]
    gross_win, gross_loss = sum(f.pnl for f in wins), -sum(f.pnl for f in losses)
    return {
        "closed_trades": len(closed),
        "win_rate_pct": round(100 * len(wins) / len(closed), 1),
        "avg_win_pct": round(100 * float(np.mean([f.pnl_pct for f in wins])), 2) if wins else 0.0,
        "avg_loss_pct": round(100 * float(np.mean([f.pnl_pct for f in losses])), 2) if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0,
    }
