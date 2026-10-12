"""
Market regime classification with hysteresis to prevent noisy switching.
Regimes:
- SIDEWAYS: Low directional strength (ADX < sideways_threshold)
- TRANSITION: Indecision or developing momentum
- TRENDING: Strong directional movement (ADX >= trending_threshold)
"""
from typing import List
import pandas as pd
import numpy as np

def classify_regimes(
    adx_series: pd.Series,
    sideways_thresh: float = 20.0,
    trending_thresh: float = 25.0,
    hysteresis: float = 1.0
) -> pd.Series:
    """
    Classifies market regime bar-by-bar with hysteresis memory.
    """
    n = len(adx_series)
    regimes = ["TRANSITION"] * n
    current_regime = "TRANSITION"

    for i in range(n):
        val = adx_series.iloc[i]
        if pd.isna(val):
            regimes[i] = "TRANSITION"
            continue

        if current_regime == "TRENDING":
            # Must fall below trending_thresh - hysteresis to leave TRENDING
            if val < (trending_thresh - hysteresis):
                if val < sideways_thresh:
                    current_regime = "SIDEWAYS"
                else:
                    current_regime = "TRANSITION"
        elif current_regime == "SIDEWAYS":
            # Must rise above sideways_thresh + hysteresis to leave SIDEWAYS
            if val > (sideways_thresh + hysteresis):
                if val >= trending_thresh:
                    current_regime = "TRENDING"
                else:
                    current_regime = "TRANSITION"
        else:  # TRANSITION
            if val >= trending_thresh:
                current_regime = "TRENDING"
            elif val < sideways_thresh:
                current_regime = "SIDEWAYS"

        regimes[i] = current_regime

    return pd.Series(regimes, index=adx_series.index)
