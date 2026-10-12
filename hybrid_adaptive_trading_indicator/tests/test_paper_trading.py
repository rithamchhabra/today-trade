"""
Unit tests for Paper Trading execution simulator.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone
from hybrid_adaptive_trading_indicator.execution.paper_trader import PaperTradingSimulator
from hybrid_adaptive_trading_indicator.risk.models import RiskDecision

@pytest.fixture
def sample_approved_decision():
    return RiskDecision(
        status="APPROVE",
        symbol="XAUUSD",
        timeframe="15m",
        regime="TRENDING",
        direction="BUY",
        entry_price=2350.0,
        stop_loss=2340.0,
        tp1=2365.0,
        tp2=2380.0,
        risk_reward=2.0,
        setup_reason="Breakout",
        candle_timestamp=datetime.now(timezone.utc),
        rejection_reason="",
        position_size=0.10,
        position_unit="lots",
        risk_amount_usd=100.0,
        estimated_cost_usd=5.0,
        net_risk_reward=1.9,
        max_capital_at_risk_pct=1.0
    )

def test_paper_trade_lifecycle_fill_and_conservative_rule(sample_approved_decision):
    sim = PaperTradingSimulator(enable_trailing_stop=True)
    trade = sim.create_trade_from_decision(sample_approved_decision)
    assert trade.status == "PENDING"

    # Candle 1: Fills on next candle open
    c1 = pd.Series({
        "datetime": pd.to_datetime("2024-01-01 12:15:00", utc=True),
        "open": 2351.0,
        "high": 2355.0,
        "low": 2349.0,
        "close": 2353.0
    })
    trade = sim.process_candle(trade, c1)
    assert trade.status == "OPEN"
    assert trade.executed_entry_price > 2351.0  # includes spread/slippage

    # Candle 2: High volatility candle where BOTH TP1 (2365) and SL (2340) are touched!
    # Low = 2335, High = 2370.
    # CRITICAL: Conservative rule requires assuming SL HIT FIRST!
    c2 = pd.Series({
        "datetime": pd.to_datetime("2024-01-01 12:30:00", utc=True),
        "open": 2353.0,
        "high": 2370.0,  # TP reached
        "low": 2335.0,   # SL breached
        "close": 2368.0
    })
    trade = sim.process_candle(trade, c2)
    assert trade.status == "CLOSED_SL"
    assert "Conservative fill" in trade.exit_reason
    assert trade.pnl_net < 0

def test_paper_trade_target_hit_and_update_dataframe(sample_approved_decision):
    from hybrid_adaptive_trading_indicator.execution.models import PaperTrade
    sample_approved_decision.candle_timestamp = pd.to_datetime("2024-01-01 12:00:00", utc=True).to_pydatetime()
    sim = PaperTradingSimulator(enable_trailing_stop=False)
    trade = sim.create_trade_from_decision(sample_approved_decision)
    
    # Test from_dict roundtrip
    d = trade.to_dict()
    reconstructed = PaperTrade.from_dict(d)
    assert reconstructed.trade_id == trade.trade_id
    assert reconstructed.status == "PENDING"

    # Create candles dataframe:
    # Bar 1: Fills
    # Bar 2: Hits TP1
    df = pd.DataFrame([
        {
            "datetime": pd.to_datetime("2024-01-01 12:15:00", utc=True),
            "open": 2351.0,
            "high": 2355.0,
            "low": 2350.0,
            "close": 2354.0,
            "atr": 5.0
        },
        {
            "datetime": pd.to_datetime("2024-01-01 12:30:00", utc=True),
            "open": 2355.0,
            "high": 2368.0,  # TP1 is 2365.0
            "low": 2352.0,
            "close": 2366.0,
            "atr": 5.0
        }
    ])

    results = sim.update_trades_against_dataframe([reconstructed], df)
    assert len(results) == 1
    updated_trade, was_closed = results[0]
    assert was_closed is True
    assert updated_trade.status == "CLOSED_TP2" or updated_trade.status.startswith("CLOSED_TP")
    assert updated_trade.pnl_net > 0

