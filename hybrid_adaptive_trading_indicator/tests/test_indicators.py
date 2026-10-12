"""
Unit tests for indicators: ADX, +DI, -DI, ATR, and Volume MA.
"""
import pytest
import pandas as pd
import numpy as np

from hybrid_adaptive_trading_indicator.indicators.adx import calculate_adx_di
from hybrid_adaptive_trading_indicator.indicators.atr import calculate_atr
from hybrid_adaptive_trading_indicator.indicators.volume import calculate_volume_ma

@pytest.fixture
def sample_ohlcv():
    periods = 100
    dates = pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, periods))
    high = close + rng.uniform(0.5, 2.0, periods)
    low = close - rng.uniform(0.5, 2.0, periods)
    open_p = close + rng.normal(0, 0.5, periods)
    volume = rng.integers(100, 1000, periods)
    return pd.DataFrame({
        "datetime": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })

def test_adx_calculation_bounds(sample_ohlcv):
    adx, plus_di, minus_di = calculate_adx_di(sample_ohlcv, length=14)
    assert len(adx) == len(sample_ohlcv)
    # Valid values should be between 0 and 100
    valid_adx = adx.dropna()
    assert len(valid_adx) > 0
    assert (valid_adx >= 0).all() and (valid_adx <= 100).all()
    assert (plus_di.dropna() >= 0).all() and (plus_di.dropna() <= 100).all()
    assert (minus_di.dropna() >= 0).all() and (minus_di.dropna() <= 100).all()

def test_atr_positive(sample_ohlcv):
    atr = calculate_atr(sample_ohlcv, length=14)
    valid_atr = atr.dropna()
    assert len(valid_atr) > 0
    assert (valid_atr > 0).all()

def test_volume_ma(sample_ohlcv):
    vol_ma, is_elevated = calculate_volume_ma(sample_ohlcv, length=20)
    assert len(vol_ma) == len(sample_ohlcv)
    assert is_elevated.dtype == bool
