"""
Unit tests for data integrity, candle completeness, and staleness detection.
"""
import pytest
from datetime import datetime, timezone, timedelta
import pandas as pd
from hybrid_adaptive_trading_indicator.data.integrity import validate_and_filter_candles

def test_data_integrity_duplicate_and_unclosed():
    now_utc = datetime.now(timezone.utc)
    # Candle 1: 30 mins ago (closed)
    t1 = (now_utc - timedelta(minutes=30)).replace(second=0, microsecond=0)
    # Candle 2: 15 mins ago (closed)
    t2 = (now_utc - timedelta(minutes=15)).replace(second=0, microsecond=0)
    # Candle 3: Forming candle right now (unclosed)
    t3 = now_utc.replace(second=0, microsecond=0)

    # Introduce duplicate of t1
    df = pd.DataFrame([
        {"datetime": t1, "open": 100, "high": 105, "low": 99, "close": 102, "volume": 10},
        {"datetime": t1, "open": 100, "high": 105, "low": 99, "close": 102, "volume": 10},
        {"datetime": t2, "open": 102, "high": 108, "low": 101, "close": 106, "volume": 20},
        {"datetime": t3, "open": 106, "high": 110, "low": 105, "close": 109, "volume": 5},
    ])

    cleaned_df, report = validate_and_filter_candles(df, drop_unclosed=True)
    assert report.duplicate_count == 1
    assert report.unfinished_candle_dropped is True
    # The forming candle t3 must have been dropped
    assert len(cleaned_df) == 2
    assert cleaned_df["datetime"].iloc[-1] == t2
