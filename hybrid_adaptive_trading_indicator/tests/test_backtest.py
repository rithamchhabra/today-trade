"""
Unit tests for Backtesting engine and metrics calculation.
"""
import pytest
import pandas as pd
import numpy as np

from hybrid_adaptive_trading_indicator.backtest.engine import run_backtest
from hybrid_adaptive_trading_indicator.data.generate_samples import generate_sample_dataset

def test_backtest_execution():
    df = generate_sample_dataset("XAUUSD", base_price=2350.0, periods=200)
    summary, trades_df, closed_trades = run_backtest(
        df=df,
        symbol="XAUUSD",
        initial_balance=10000.0,
        risk_pct=1.0,
        adx_sideways=20.0,
        adx_trending=25.0
    )

    assert summary.starting_balance == 10000.0
    assert summary.total_trades == len(closed_trades)
    assert summary.win_rate_pct >= 0.0 and summary.win_rate_pct <= 100.0
    assert summary.max_drawdown_pct >= 0.0
    if not trades_df.empty:
        assert "equity" in trades_df.columns
        assert "pnl_net" in trades_df.columns
