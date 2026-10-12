"""
Unit tests for deterministic Risk Approval Engine.
"""
import pytest
from datetime import datetime, timezone
from hybrid_adaptive_trading_indicator.risk.approval_engine import RiskApprovalEngine
from hybrid_adaptive_trading_indicator.risk.position_sizer import calculate_position_size
from hybrid_adaptive_trading_indicator.config import INSTRUMENT_SPECS
from hybrid_adaptive_trading_indicator.strategy.models import TradeCandidate

@pytest.fixture
def base_candidate():
    return TradeCandidate(
        symbol="XAUUSD",
        timeframe="15m",
        candle_timestamp=datetime.now(timezone.utc),
        strategy="TRENDING_BREAKOUT",
        regime="TRENDING",
        direction="BUY",
        entry_price=2350.0,
        stop_loss=2340.0,  # $10 SL
        tp1=2370.0,        # $20 TP (2.0 RR)
        tp2=2380.0,
        risk_distance=10.0,
        reward_tp1_distance=20.0,
        risk_reward_ratio=2.0,
        setup_reason="Resistance breakout",
        volume_confirmed=True
    )

def test_position_sizing_gold():
    spec = INSTRUMENT_SPECS["XAUUSD"]
    # Account: $10,000, Risk: 1% ($100), SL distance: $10.00
    # 1 lot = 100 oz. A $10 drop = $1,000 loss per 1.0 lot.
    # $100 / $1,000 = 0.10 lot
    pos_size, risk_usd, friction, total_capital = calculate_position_size(
        spec=spec,
        account_balance=10000.0,
        risk_pct=1.0,
        entry_price=2350.0,
        stop_loss=2340.0
    )
    assert pos_size == 0.10
    assert risk_usd == 100.0
    assert friction > 0.0

def test_risk_approval_success(base_candidate):
    engine = RiskApprovalEngine(account_balance=10000.0, risk_per_trade_pct=1.0)
    decision = engine.evaluate(base_candidate)
    assert decision.status == "APPROVE"
    assert decision.position_size == 0.10
    assert decision.rejection_reason == ""

def test_risk_rejection_daily_loss_limit(base_candidate):
    engine = RiskApprovalEngine(max_daily_loss=300.0)
    # Today's realized loss already hit $350
    decision = engine.evaluate(base_candidate, current_daily_loss=350.0)
    assert decision.status == "REJECT"
    assert "daily loss limit" in decision.rejection_reason.lower()

def test_risk_rejection_max_concurrent_trades(base_candidate):
    engine = RiskApprovalEngine(max_concurrent_trades=2)
    open_trades = [
        {"symbol": "BTCUSDT", "direction": "BUY"},
        {"symbol": "EURUSD", "direction": "BUY"}
    ]
    decision = engine.evaluate(base_candidate, open_positions=open_trades)
    assert decision.status == "REJECT"
    assert "concurrent trades" in decision.rejection_reason.lower()

def test_risk_rejection_duplicate_trade(base_candidate):
    engine = RiskApprovalEngine()
    open_trades = [{"symbol": "XAUUSD", "direction": "BUY"}]
    decision = engine.evaluate(base_candidate, open_positions=open_trades)
    assert decision.status == "REJECT"
    assert "duplicate trade" in decision.rejection_reason.lower()

def test_risk_rejection_insufficient_rr():
    cand = TradeCandidate(
        symbol="XAUUSD",
        timeframe="15m",
        candle_timestamp=datetime.now(timezone.utc),
        strategy="SIDEWAYS_REVERSION",
        regime="SIDEWAYS",
        direction="BUY",
        entry_price=2350.0,
        stop_loss=2340.0,  # $10 risk
        tp1=2355.0,        # only $5 reward (0.5 RR)
        tp2=2360.0,
        risk_distance=10.0,
        reward_tp1_distance=5.0,
        risk_reward_ratio=0.5,
        setup_reason="Weak range",
        volume_confirmed=True
    )
    engine = RiskApprovalEngine(min_risk_reward=1.5)
    decision = engine.evaluate(cand)
    assert decision.status == "REJECT"
    assert "risk/reward" in decision.rejection_reason.lower()
