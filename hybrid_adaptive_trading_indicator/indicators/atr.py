"""
Average True Range (ATR) calculation using Wilder's smoothing.
"""
import pandas as pd
import numpy as np

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Computes Average True Range (ATR) using Wilder's smoothing.
    """
    if len(df) < length:
        return pd.Series(np.nan, index=df.index)

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / length
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    return atr
