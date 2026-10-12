"""
Paper trading execution models.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

@dataclass
class PaperTrade:
    trade_id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    status: str     # "PENDING", "OPEN", "CLOSED_TP1", "CLOSED_TP2", "CLOSED_SL", "CLOSED_TRAILING", "CANCELLED"
    signal_time: datetime
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    target_entry_price: float = 0.0
    executed_entry_price: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    trailing_stop_active: bool = False
    current_trailing_stop: float = 0.0
    position_size: float = 0.0
    position_unit: str = "lots"
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_gross: float = 0.0
    fees_paid: float = 0.0
    pnl_net: float = 0.0
    r_multiple: float = 0.0
    setup_reason: str = ""

    def to_dict(self):
        d = asdict(self)
        if self.signal_time:
            d["signal_time"] = self.signal_time.isoformat()
        if self.entry_time:
            d["entry_time"] = self.entry_time.isoformat()
        if self.exit_time:
            d["exit_time"] = self.exit_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PaperTrade":
        def parse_dt(v):
            if not v:
                return None
            if isinstance(v, datetime):
                return v
            try:
                return datetime.fromisoformat(str(v))
            except Exception:
                return None

        return cls(
            trade_id=str(d.get("trade_id", "")),
            symbol=str(d.get("symbol", "")),
            direction=str(d.get("direction", "BUY")),
            status=str(d.get("status", "PENDING")),
            signal_time=parse_dt(d.get("signal_time")) or datetime.now(),
            entry_time=parse_dt(d.get("entry_time")),
            exit_time=parse_dt(d.get("exit_time")),
            target_entry_price=float(d.get("target_entry_price", 0.0) or 0.0),
            executed_entry_price=float(d.get("executed_entry_price", 0.0) or 0.0),
            stop_loss=float(d.get("stop_loss", 0.0) or 0.0),
            tp1=float(d.get("tp1", 0.0) or 0.0),
            tp2=float(d.get("tp2", 0.0) or 0.0),
            trailing_stop_active=bool(d.get("trailing_stop_active", False)),
            current_trailing_stop=float(d.get("current_trailing_stop", 0.0) or 0.0),
            position_size=float(d.get("position_size", 0.0) or 0.0),
            position_unit=str(d.get("position_unit", "lots")),
            exit_price=float(d["exit_price"]) if d.get("exit_price") is not None else None,
            exit_reason=str(d.get("exit_reason", "") or ""),
            pnl_gross=float(d.get("pnl_gross", 0.0) or 0.0),
            fees_paid=float(d.get("fees_paid", 0.0) or 0.0),
            pnl_net=float(d.get("pnl_net", 0.0) or 0.0),
            r_multiple=float(d.get("r_multiple", 0.0) or 0.0),
            setup_reason=str(d.get("setup_reason", "") or "")
        )
