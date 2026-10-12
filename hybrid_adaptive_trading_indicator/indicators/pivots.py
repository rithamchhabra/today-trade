"""
Strict non-repainting pivot high and pivot low detection.

Non-repainting invariant:
A swing high or low at candle index k is ONLY officially confirmed at candle index
j = k + pivot_right, after the requisite number of confirmation candles have completed.
The confirmed pivot level is emitted at index j (never backdated to k for decision logic).
"""
from typing import Tuple
import pandas as pd
import numpy as np

def calculate_confirmed_pivots(
    df: pd.DataFrame,
    pivot_left: int = 3,
    pivot_right: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes confirmed pivot highs and pivot lows strictly without repainting.
    
    Returns:
        (pivot_high_at_confirmation, pivot_low_at_confirmation, resistance_series, support_series)
        
    At index j, if candle k (j - pivot_right) was a confirmed swing, the pivot price is stored at index j.
    Resistance and Support are forward-filled from confirmed pivots.
    """
    n = len(df)
    pivot_high = pd.Series(np.nan, index=df.index, dtype=float)
    pivot_low = pd.Series(np.nan, index=df.index, dtype=float)

    if n < pivot_left + pivot_right + 1:
        res = pivot_high.ffill()
        sup = pivot_low.ffill()
        return pivot_high, pivot_low, res, sup

    highs = df["high"].values
    lows = df["low"].values

    for j in range(pivot_left + pivot_right, n):
        k = j - pivot_right  # candidate pivot candle
        start_idx = k - pivot_left
        end_idx = j + 1  # includes bar j (k + pivot_right)
        
        # Check high
        window_highs = highs[start_idx:end_idx]
        candidate_high = highs[k]
        if candidate_high == np.max(window_highs) and np.sum(window_highs == candidate_high) == 1:
            pivot_high.iloc[j] = candidate_high

        # Check low
        window_lows = lows[start_idx:end_idx]
        candidate_low = lows[k]
        if candidate_low == np.min(window_lows) and np.sum(window_lows == candidate_low) == 1:
            pivot_low.iloc[j] = candidate_low

    resistance = pivot_high.ffill()
    support = pivot_low.ffill()

    return pivot_high, pivot_low, resistance, support
