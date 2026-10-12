"""
Unit tests for strategy signal generation: sideways support/resistance rejection and trending breakouts.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from hybrid_adaptive_trading_indicator.strategy.sideways import evaluate_sideways_setup
from hybrid_adaptive_trading_indicator.strategy.trending import evaluate_trending_setup

def create_mock_candle_series(**kwargs):
    default = {
        "datetime": pd.to_datetime("2024-01-01 12:00:00", utc=True),
        "open": 2300.0,
        "high": 2310.0,
        "low": 2295.0,
        "close": 2308.0,
        "volume": 500.0,
        "atr": 10.0,
        "support": 2290.0,
        "resistance": 2320.0,
        "sup_zone_high": 2292.0,
        "res_zone_low": 2318.0,
        "plus_di": 28.0,
        "minus_di": 12.0,
        "adx": 30.0,
        "vol_ma": 400.0,
        "is_vol_elevated": True,
        "regime": "TRENDING"
    }
    default.update(kwargs)
    return default

def test_sideways_support_rejection_buy():
    # Previous candle
    prev = create_mock_candle_series(close=2295.0, regime="SIDEWAYS")
    # Current candle dips to 2291 (inside support zone 2290-2292), but closes bullish at 2298
    curr = create_mock_candle_series(
        open=2292.0,
        low=2291.0,
        high=2300.0,
        close=2298.0,
        regime="SIDEWAYS"
    )
    df = pd.DataFrame([prev, curr])
    
    cand = evaluate_sideways_setup(df, idx=1, symbol="XAUUSD", timeframe="15m")
    assert cand is not None
    assert cand.direction == "BUY"
    assert cand.strategy == "SIDEWAYS_REVERSION"
    assert cand.entry_price == 2298.0
    assert cand.stop_loss < 2291.0  # SL placed below swing low with ATR buffer

def test_trending_breakout_buy():
    prev = create_mock_candle_series(close=2318.0, resistance=2320.0, regime="TRENDING")
    # Current candle closes cleanly above resistance
    curr = create_mock_candle_series(
        open=2319.0,
        close=2325.0,
        high=2328.0,
        low=2318.0,
        resistance=2320.0,
        plus_di=30.0,
        minus_di=15.0,
        regime="TRENDING"
    )
    df = pd.DataFrame([prev, curr])

    cand = evaluate_trending_setup(df, idx=1, symbol="XAUUSD", timeframe="15m")
    assert cand is not None
    assert cand.direction == "BUY"
    assert cand.strategy == "TRENDING_BREAKOUT"
    assert cand.entry_price == 2325.0
    assert cand.tp1 > cand.entry_price
