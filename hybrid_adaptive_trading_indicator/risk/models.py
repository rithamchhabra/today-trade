"""
Risk engine models and structured decision types.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RiskDecision:
    status: str  # "APPROVE", "REJECT", "WAIT"
    symbol: str
    timeframe: str
    regime: str
    direction: str
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_reward: float
    setup_reason: str
    candle_timestamp: datetime
    rejection_reason: str
    position_size: float
    position_unit: str  # "lots" (Gold) or "BTC" (Bitcoin)
    risk_amount_usd: float
    estimated_cost_usd: float
    net_risk_reward: float
    max_capital_at_risk_pct: float

    def to_dict(self):
        from dataclasses import asdict
        d = asdict(self)
        if self.candle_timestamp:
            d["candle_timestamp"] = self.candle_timestamp.isoformat()
        return d
