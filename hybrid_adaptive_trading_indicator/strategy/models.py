"""
Strategy data models for trading signals and trade setups.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TradeCandidate:
    symbol: str
    timeframe: str
    candle_timestamp: datetime
    strategy: str  # "SIDEWAYS_REVERSION" or "TRENDING_BREAKOUT"
    regime: str    # "SIDEWAYS", "TRANSITION", "TRENDING"
    direction: str # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_distance: float
    reward_tp1_distance: float
    risk_reward_ratio: float
    setup_reason: str
    retest_confirmed: bool = False
    volume_confirmed: bool = False
