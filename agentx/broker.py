"""In-memory simulated brokerage. There is deliberately no live-trading path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    stop: Optional[float]
    opened: str
    strategy: str
    entry_fees: float = 0.0
    entry_t: int = 0


@dataclass
class Fill:
    date: str
    symbol: str
    side: str
    qty: int
    price: float
    fees: float
    reason: str
    strategy: str
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    held_days: Optional[int] = None


class PaperBroker:
    PAPER_ONLY = True

    def __init__(self, cash: float = 100_000.0, slippage_bps: float = 5.0, fee_bps: float = 2.0):
        self.initial_cash = cash
        self.cash = cash
        self.slip = slippage_bps / 1e4
        self.fee = fee_bps / 1e4
        self.positions: dict = {}
        self.fills: list = []

    def buy(self, symbol, qty, ref_price, date, stop, strategy, reason, t=0) -> Optional[Fill]:
        if qty <= 0 or symbol in self.positions:
            return None
        price = ref_price * (1 + self.slip)
        notional = qty * price
        fees = notional * self.fee
        if notional + fees > self.cash + 1e-9:
            return None
        self.cash -= notional + fees
        self.positions[symbol] = Position(symbol, int(qty), price, stop, date, strategy, fees, t)
        fill = Fill(date, symbol, "buy", int(qty), price, fees, reason, strategy)
        self.fills.append(fill)
        return fill

    def sell(self, symbol, ref_price, date, reason, t=0) -> Optional[Fill]:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        price = ref_price * (1 - self.slip)
        proceeds = pos.qty * price
        fees = proceeds * self.fee
        cost = pos.qty * pos.avg_price + pos.entry_fees
        pnl = proceeds - fees - cost
        self.cash += proceeds - fees
        del self.positions[symbol]
        fill = Fill(date, symbol, "sell", pos.qty, price, fees, reason, pos.strategy,
                    pnl=pnl, pnl_pct=pnl / cost, held_days=t - pos.entry_t)
        self.fills.append(fill)
        return fill

    def gross(self, prices: dict) -> float:
        return sum(p.qty * prices.get(s, p.avg_price) for s, p in self.positions.items())

    def equity(self, prices: dict) -> float:
        return self.cash + self.gross(prices)
