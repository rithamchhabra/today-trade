"""
ADX, +DI, and -DI calculations using classic Wilder's smoothing.
ADX measures trend strength (not direction), while +DI / -DI and price structure determine direction.
"""
from typing import Tuple
import pandas as pd
import numpy as np

def calculate_adx_di(
    df: pd.DataFrame,
    length: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes Wilder's ADX, +DI, and -DI.
    Returns: (adx_series, plus_di_series, minus_di_series)
    """
    if len(df) < length + 1:
        empty = pd.Series(np.nan, index=df.index)
        return empty, empty, empty

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    # True Range (TR)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    # Wilder's Smoothing: ewm with alpha = 1 / length, adjust=False
    alpha = 1.0 / length
    tr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    # Prevent divide by zero
    safe_tr = tr_smooth.replace(0, np.nan)
    plus_di = 100.0 * (plus_dm_smooth / safe_tr)
    minus_di = 100.0 * (minus_dm_smooth / safe_tr)

    # Directional Movement Index (DX)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum)

    # Average Directional Index (ADX)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    return adx, plus_di, minus_di
