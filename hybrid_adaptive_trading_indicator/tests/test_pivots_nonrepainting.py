"""
Unit tests for non-repainting pivot confirmation and future data leakage prevention.
"""
import pytest
import pandas as pd
import numpy as np

from hybrid_adaptive_trading_indicator.indicators.pivots import calculate_confirmed_pivots
from hybrid_adaptive_trading_indicator.strategy.engine import prepare_indicators, evaluate_all_candidates

def test_pivot_confirmation_timing():
    """
    Verifies that a pivot occurring at bar k is NEVER marked before k + pivot_right.
    """
    # Create synthetic series with a prominent swing high at index 5
    highs = [10.0, 11.0, 12.0, 13.0, 14.0, 25.0, 14.0, 13.0, 12.0, 11.0, 10.0]
    lows = [9.0, 10.0, 11.0, 12.0, 13.0, 20.0, 13.0, 12.0, 11.0, 10.0, 9.0]
    n = len(highs)
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": lows,
        "high": highs,
        "low": lows,
        "close": highs,
        "volume": [100] * n
    })

    pivot_left = 3
    pivot_right = 3
    # Candidate peak is at index 5.
    # It must only be confirmed at index 5 + pivot_right = 8!
    p_high, p_low, res, sup = calculate_confirmed_pivots(df, pivot_left=pivot_left, pivot_right=pivot_right)

    # Before index 8, pivot high must be NaN
    for idx in range(8):
        assert pd.isna(p_high.iloc[idx]), f"Pivot high should be NaN at index {idx} before confirmation"

    # At index 8, it must equal 25.0
    assert p_high.iloc[8] == 25.0
    assert res.iloc[8] == 25.0

def test_no_repainting_when_future_candles_appended():
    """
    CRITICAL NON-REPAINTING TEST:
    Calculate pivots and signals on dataset of length N.
    Then append 30 new future candles and recalculate.
    Verify that all prior values for index 0..N-1 are 100% identical!
    """
    rng = np.random.default_rng(999)
    periods = 120
    dates = pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC")
    close = 2000.0 + np.cumsum(rng.normal(0, 5, periods))
    high = close + rng.uniform(1, 5, periods)
    low = close - rng.uniform(1, 5, periods)
    open_p = close + rng.normal(0, 2, periods)
    vol = rng.integers(100, 1000, periods)

    full_df = pd.DataFrame({
        "datetime": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol
    })

    # Slice at N = 80
    n_slice = 80
    initial_df = full_df.iloc[:n_slice].copy()

    # Calculate indicators on initial subset
    ind_initial = prepare_indicators(initial_df, pivot_left=3, pivot_right=3)
    _, initial_candidates = evaluate_all_candidates(ind_initial, symbol="XAUUSD")

    # Now calculate on full dataset with 40 new future candles
    ind_full = prepare_indicators(full_df, pivot_left=3, pivot_right=3)
    _, full_candidates = evaluate_all_candidates(ind_full, symbol="XAUUSD")

    # Check that past pivots and resistance are unchanged
    np.testing.assert_array_equal(
        ind_initial["pivot_high"].values,
        ind_full["pivot_high"].iloc[:n_slice].values
    )
    np.testing.assert_array_equal(
        ind_initial["resistance"].values,
        ind_full["resistance"].iloc[:n_slice].values
    )
    np.testing.assert_array_equal(
        ind_initial["support"].values,
        ind_full["support"].iloc[:n_slice].values
    )

    # Check that past candidates up to timestamp of initial_df[-1] are completely identical
    cutoff_time = initial_df["datetime"].iloc[-1].to_pydatetime()
    candidates_in_full_before_cutoff = [c for c in full_candidates if c.candle_timestamp <= cutoff_time]

    assert len(initial_candidates) == len(candidates_in_full_before_cutoff)
    for c_init, c_full in zip(initial_candidates, candidates_in_full_before_cutoff):
        assert c_init.candle_timestamp == c_full.candle_timestamp
        assert c_init.direction == c_full.direction
        assert c_init.entry_price == c_full.entry_price
        assert c_init.stop_loss == c_full.stop_loss
        assert c_init.tp1 == c_full.tp1
