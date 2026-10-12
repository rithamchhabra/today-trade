"""
Volume moving average and volume filter indicators.
"""
from typing import Tuple
import pandas as pd
import numpy as np

def calculate_volume_ma(
    df: pd.DataFrame,
    length: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates simple moving average of volume and volume confirmation flag.
    Returns:
        (volume_ma, is_volume_elevated)
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        empty = pd.Series(0.0, index=df.index)
        flag = pd.Series(True, index=df.index)  # If volume is missing/0, do not block
        return empty, flag

    vol = df["volume"]
    vol_ma = vol.rolling(length, min_periods=max(2, length // 2)).mean().fillna(vol)
    is_elevated = vol >= vol_ma
    return vol_ma, is_elevated
