"""Tools the agent can call. All of them are read-only: the agent cannot place orders."""
from __future__ import annotations

from .backtest import run_backtest
from .strategies import STRATEGIES, rsi, sma_last


def _r(x, n=2):
    return round(float(x), n)


class ToolBox:
    SPEC = {
        "get_market_overview": "() -> per-symbol price stats (5d/20d return, volatility, RSI, trend) and a regime guess",
        "get_news": "(symbol, days=5) -> recent headlines with sentiment from -1 (bad) to +1 (good)",
        "run_backtest": "(strategy, lookback_days=120) -> trailing-window backtest vs equal-weight buy-and-hold under the same risk rules",
        "get_portfolio": "() -> cash, equity, open positions",
        "get_risk_status": "() -> risk limits, today's P&L, kill-switch state",
        "get_memory": "() -> realised results per strategy from earlier trades (persists across sessions)",
    }

    def __init__(self, session):
        self.s = session

    def describe(self) -> str:
        lines = [f"- {name}{sig}" for name, sig in self.SPEC.items()]
        lines.append("Strategies: " + "; ".join(f"{k} ({v[2]})" for k, v in STRATEGIES.items()))
        return "\n".join(lines)

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None or name not in self.SPEC:
            return {"error": f"unknown tool '{name}'", "available": list(self.SPEC)}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # tools must never crash the agent loop
            return {"error": f"{type(e).__name__}: {e}"}

    # --- tools (all look-ahead safe: they only read up to the session's current day) ---
    def _t_get_market_overview(self) -> dict:
        s, t = self.s, self.s.t
        rows, rets20, vols = {}, [], []
        for sym in s.symbols:
            c = s.market.close[sym][: t + 1]
            r5, r20 = (c[-1] / c[-6] - 1) * 100, (c[-1] / c[-21] - 1) * 100
            vol = float((c[-20:][1:] / c[-20:][:-1] - 1).std() * 100)
            rows[sym] = {"last": _r(c[-1]), "ret_5d_pct": _r(r5), "ret_20d_pct": _r(r20), "vol_20d_pct": _r(vol),
                         "rsi14": _r(rsi(c)), "above_sma50": bool(c[-1] > sma_last(c, 50))}
            rets20.append(r20)
            vols.append(vol)
        avg, vol = sum(rets20) / len(rets20), sum(vols) / len(vols)
        regime = "high-volatility downtrend" if avg < -3 and vol > 1.3 else "uptrend" if avg > 2 else "downtrend" if avg < -2 else "choppy/sideways"
        return {"date": s.date, "regime_guess": regime, "avg_20d_return_pct": _r(avg), "avg_daily_vol_pct": _r(vol), "symbols": rows}

    def _t_get_news(self, symbol: str, days: int = 5) -> dict:
        if symbol not in self.s.symbols:
            return {"error": f"unknown symbol {symbol}"}
        items = self.s.market.news_between(symbol, self.s.t, max(1, min(int(days), 30)))
        return {"symbol": symbol, "items": items, "net_sentiment": _r(sum(i["sentiment"] for i in items))}

    def _t_run_backtest(self, strategy: str, lookback_days: int = 120) -> dict:
        if strategy not in STRATEGIES:
            return {"error": f"unknown strategy '{strategy}'", "available": list(STRATEGIES)}
        s, t = self.s, self.s.t
        look = max(40, min(int(lookback_days), 250))
        start = max(0, t - look - 40)
        window = s.market.window(start, t)
        rep = run_backtest(window, strategy, start_idx=min(40, len(window) // 3), risk_cfg=s.gatekeeper.cfg, cash=s.initial_cash)
        return {"strategy": strategy, "days_tested": rep["days"], "total_return_pct": rep["agent"]["total_return_pct"],
                "buy_and_hold_return_pct": rep["buy_and_hold"]["total_return_pct"], "excess_return_pct": rep["excess_return_pct"],
                "sharpe": rep["agent"]["sharpe"], "max_drawdown_pct": rep["agent"]["max_drawdown_pct"],
                "bh_max_drawdown_pct": rep["buy_and_hold"]["max_drawdown_pct"],
                "closed_trades": rep["trades"]["closed_trades"], "win_rate_pct": rep["trades"]["win_rate_pct"]}

    def _t_get_portfolio(self) -> dict:
        s = self.s
        pos = {sym: {"qty": p.qty, "avg_price": _r(p.avg_price), "stop": _r(p.stop) if p.stop else None,
                     "last": _r(s.last_closes.get(sym, p.avg_price))} for sym, p in s.broker.positions.items()}
        return {"cash": _r(s.broker.cash), "equity": _r(s.current_equity), "positions": pos}

    def _t_get_risk_status(self) -> dict:
        s, c = self.s, self.s.gatekeeper.cfg
        return {"day_pnl_pct": _r(s.day_pnl * 100), "daily_loss_limit_pct": c.daily_loss_limit_pct * 100,
                "risk_per_trade_pct": c.risk_per_trade_pct * 100, "max_position_pct": c.max_position_pct * 100,
                "max_open_positions": c.max_open_positions, "open_positions": len(s.broker.positions),
                "kill_switch_engaged": s.kill.is_engaged(), "current_strategy": s.strategy}

    def _t_get_memory(self) -> dict:
        return {"by_strategy": self.s.memory.summary() if self.s.memory is not None else {}}
