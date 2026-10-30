"""Rule-based strategies. Each looks only at history up to today and returns a Signal.

The agent can pick between them, but cannot change the maths here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Signal:
    enter: bool = False
    exit: bool = False
    stop: Optional[float] = None
    reason: str = ""


def atr(bars: dict, n: int = 14) -> float:
    h, l, c = bars["high"], bars["low"], bars["close"]
    k = min(n, len(c) - 1)
    if k < 1:
        return float(h[-1] - l[-1])
    h, l, pc = h[-k:], l[-k:], c[-k - 1 : -1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(tr.mean())


def sma_last(c: np.ndarray, n: int, back: int = 0) -> float:
    end = len(c) - back
    return float(c[end - n : end].mean())


def rsi(c: np.ndarray, period: int = 14) -> float:
    d = np.diff(c[-(period + 1):])
    gains, losses = d[d > 0].sum() / period, -d[d < 0].sum() / period
    if losses == 0:
        return 100.0
    return float(100 - 100 / (1 + gains / losses))


def sma_crossover(bars, fast: int = 10, slow: int = 30, atr_mult: float = 2.0, recent: int = 3) -> Signal:
    c = bars["close"]
    if len(c) < slow + recent + 1:
        return Signal(reason="insufficient history")
    f_now, s_now = sma_last(c, fast), sma_last(c, slow)
    crossed = any(sma_last(c, fast, b + 1) <= sma_last(c, slow, b + 1) and sma_last(c, fast, b) > sma_last(c, slow, b)
                  for b in range(recent))
    stop = float(c[-1] - atr_mult * atr(bars))
    if f_now > s_now and crossed:
        return Signal(enter=True, stop=stop, reason=f"{fast}-day average crossed above {slow}-day average (trend turning up)")
    if f_now < s_now:
        return Signal(exit=True, reason=f"{fast}-day average fell below {slow}-day average (trend turning down)")
    return Signal(reason="no crossover")


def rsi_mean_reversion(bars, period: int = 14, low: float = 30.0, high: float = 55.0, atr_mult: float = 2.5) -> Signal:
    c = bars["close"]
    if len(c) < 51:
        return Signal(reason="insufficient history")
    r = rsi(c, period)
    stop = float(c[-1] - atr_mult * atr(bars))
    if r < low and c[-1] > 0.9 * sma_last(c, 50):
        return Signal(enter=True, stop=stop, reason=f"RSI {r:.0f} is oversold (<{low:.0f}) while price is still near its 50-day average")
    if r > high:
        return Signal(exit=True, reason=f"RSI {r:.0f} recovered above {high:.0f}")
    return Signal(reason=f"RSI {r:.0f}, no signal")


def breakout(bars, entry_lookback: int = 20, exit_lookback: int = 10, atr_mult: float = 2.0) -> Signal:
    c, h, l = bars["close"], bars["high"], bars["low"]
    if len(c) < entry_lookback + 2:
        return Signal(reason="insufficient history")
    stop = float(c[-1] - atr_mult * atr(bars))
    if c[-1] > h[-entry_lookback - 1 : -1].max():
        return Signal(enter=True, stop=stop, reason=f"closed above the {entry_lookback}-day high (breakout)")
    if c[-1] < l[-exit_lookback - 1 : -1].min():
        return Signal(exit=True, reason=f"closed below the {exit_lookback}-day low (breakout failed)")
    return Signal(reason="inside range")


def risk_off(bars, **_) -> Signal:
    return Signal(exit=True, reason="risk-off mode: no strategy currently justifies holding risk")


# name -> (function, default params, one-line description)
STRATEGIES = {
    "sma_crossover": (sma_crossover, {}, "trend following: buy when fast average crosses above slow average"),
    "rsi_mean_reversion": (rsi_mean_reversion, {}, "mean reversion: buy oversold dips, sell on recovery"),
    "breakout": (breakout, {}, "momentum: buy new 20-day highs, exit on 10-day lows"),
    "risk_off": (risk_off, {}, "stay in cash: close positions, take no new ones"),
}
