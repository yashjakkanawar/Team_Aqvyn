"""Backtester: replays the SAME session engine and risk rules with a fixed strategy and no LLM."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .data import synthetic_market, load_csv_dir
from .risk import RiskConfig
from .session import TradingSession
from .strategies import STRATEGIES


def run_backtest(market, strategy: str, params: Optional[dict] = None, start_idx: int = 40,
                 risk_cfg: Optional[RiskConfig] = None, cash: float = 100_000.0,
                 include_curves: bool = False) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}'. choose from {sorted(STRATEGIES)}")
    s = TradingSession(market, cash=cash, risk_cfg=risk_cfg, agent=None, strategy=strategy,
                       strategy_params=params, start_idx=start_idx)
    s.run()
    rep = s.report(include_curves=include_curves)
    rep["strategy"] = strategy
    return rep


def compare_all(market, start_idx: int = 40) -> list:
    return [run_backtest(market, name, start_idx=start_idx) for name in STRATEGIES]


def format_table(reports: list) -> str:
    head = f"{'strategy':<20}{'return%':>9}{'B&H%':>9}{'excess%':>9}{'sharpe':>8}{'maxDD%':>8}{'trades':>8}{'win%':>7}"
    rows = [head, "-" * len(head)]
    for r in reports:
        rows.append(f"{r['strategy']:<20}{r['agent']['total_return_pct']:>9.2f}{r['buy_and_hold']['total_return_pct']:>9.2f}"
                    f"{r['excess_return_pct']:>9.2f}{r['agent']['sharpe']:>8.2f}{r['agent']['max_drawdown_pct']:>8.2f}"
                    f"{r['trades']['closed_trades']:>8}{r['trades']['win_rate_pct']:>7.1f}")
    return "\n".join(rows)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Backtest every strategy against equal-weight buy-and-hold")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv-dir", help="folder of SYMBOL.csv files (Yahoo style) instead of synthetic data")
    a = ap.parse_args(argv)
    market = load_csv_dir(a.csv_dir) if a.csv_dir else synthetic_market(days=a.days, seed=a.seed)
    print(f"data source: {market.source}, {len(market)} days, symbols: {', '.join(market.symbols)}")
    print(format_table(compare_all(market)))


if __name__ == "__main__":
    main()
