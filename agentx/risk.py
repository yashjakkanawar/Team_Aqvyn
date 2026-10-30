"""Deterministic risk gatekeeper and human kill switch.

Nothing the LLM outputs can change these rules. The model may only (a) say no to an entry or
(b) shrink the position through `conviction`. It can never enlarge a position or skip a stop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20        # one position <= 20% of equity
    risk_per_trade_pct: float = 0.01      # a stop-out loses ~1% of equity
    max_open_positions: int = 5
    max_gross_exposure_pct: float = 0.80  # always keep >=20% in cash
    daily_loss_limit_pct: float = 0.02    # day down 2% -> no new entries
    min_stop_pct: float = 0.01
    max_stop_pct: float = 0.15
    cash_buffer_pct: float = 0.01         # headroom for slippage and fees


@dataclass
class Proposal:
    symbol: str
    side: str                 # "buy" | "sell"
    price: float              # reference price (latest close)
    stop: Optional[float] = None
    conviction: float = 1.0   # 0..1, can only SHRINK the position
    strategy: str = ""
    rationale: str = ""


@dataclass
class Decision:
    approved: bool
    qty: int = 0
    reasons: list = field(default_factory=list)  # why rejected
    notes: list = field(default_factory=list)    # how it was sized


class KillSwitch:
    """Human kill switch. Engage via the dashboard, code, or by creating the KILL_SWITCH file."""

    def __init__(self, file_path: Optional[str] = None):
        self._engaged = False
        self._reason = ""
        self.flatten = False
        self.file_path = Path(file_path) if file_path else None

    def engage(self, reason: str = "manual", flatten: bool = False) -> None:
        self._engaged, self._reason, self.flatten = True, reason, flatten

    def release(self) -> None:
        self._engaged, self._reason, self.flatten = False, "", False

    def is_engaged(self) -> bool:
        return self._engaged or (self.file_path is not None and self.file_path.exists())

    @property
    def reason(self) -> str:
        if self._engaged:
            return self._reason
        if self.file_path is not None and self.file_path.exists():
            return f"{self.file_path.name} file present"
        return ""


def _m(x: float) -> str:
    return f"${x:,.0f}"


class RiskGatekeeper:
    def __init__(self, cfg: RiskConfig, kill: KillSwitch):
        self.cfg, self.kill = cfg, kill

    def review(self, p: Proposal, equity: float, cash: float, positions: dict, prices: dict,
               day_pnl_pct: float) -> Decision:
        c = self.cfg
        if p.side == "sell":
            if p.symbol not in positions:
                return Decision(False, 0, ["no open position to sell"])
            return Decision(True, positions[p.symbol].qty, [], ["exits are always allowed because they reduce risk"])

        reasons = []
        if self.kill.is_engaged():
            reasons.append(f"human kill switch is engaged ({self.kill.reason}): no new entries")
        if day_pnl_pct <= -c.daily_loss_limit_pct:
            reasons.append(f"daily loss limit reached ({day_pnl_pct * 100:.2f}% today vs -{c.daily_loss_limit_pct * 100:.1f}% limit): no new entries")
        if p.symbol in positions:
            reasons.append("already holding this symbol")
        if len(positions) >= c.max_open_positions:
            reasons.append(f"already at the maximum of {c.max_open_positions} open positions")
        if p.stop is None:
            reasons.append("no stop-loss supplied (every entry must have one)")
        else:
            dist = (p.price - p.stop) / p.price
            if dist <= 0:
                reasons.append("stop-loss must be below the entry price")
            elif dist < c.min_stop_pct:
                reasons.append(f"stop-loss too tight ({dist * 100:.2f}% < {c.min_stop_pct * 100:.1f}%)")
            elif dist > c.max_stop_pct:
                reasons.append(f"stop-loss too wide ({dist * 100:.1f}% > {c.max_stop_pct * 100:.0f}%)")
        if reasons:
            return Decision(False, 0, reasons)

        conviction = min(max(p.conviction, 0.0), 1.0)
        per_share_risk = p.price - p.stop
        risk_budget = equity * c.risk_per_trade_pct * conviction
        caps = {
            "risk budget": risk_budget / per_share_risk,
            f"{c.max_position_pct * 100:.0f}% single-position limit": equity * c.max_position_pct / p.price,
            f"{c.max_gross_exposure_pct * 100:.0f}% total-exposure limit":
                (equity * c.max_gross_exposure_pct - sum(pos.qty * prices.get(s, pos.avg_price) for s, pos in positions.items())) / p.price,
            "available cash": cash / (p.price * (1 + c.cash_buffer_pct)),
        }
        binding = min(caps, key=caps.get)
        qty = int(math.floor(max(caps[binding], 0)))
        if qty < 1:
            return Decision(False, 0, [f"position would be under 1 share after the {binding}"])
        notes = [f"sized by the {binding}: {qty} shares (~{_m(qty * p.price)}, {qty * p.price / equity * 100:.1f}% of equity)",
                 f"if the stop at {p.stop:.2f} is hit the loss is about {_m(qty * per_share_risk)} ({qty * per_share_risk / equity * 100:.2f}% of equity)"]
        if conviction < 1.0 and binding == "risk budget":
            notes.append(f"risk budget scaled down to {conviction * 100:.0f}% because of the agent's conviction")
        return Decision(True, qty, [], notes)
