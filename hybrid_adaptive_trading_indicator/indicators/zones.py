"""
Support and Resistance zone calculation with ATR-based boundaries.
"""
from typing import Tuple
import pandas as pd
import numpy as np

def calculate_sr_zones(
    resistance: pd.Series,
    support: pd.Series,
    atr: pd.Series,
    zone_atr_mult: float = 0.20
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes S/R zone upper and lower bounds:
    - Resistance zone: [resistance - zone_atr_mult * atr, resistance]
    - Support zone: [support, support + zone_atr_mult * atr]
    
    Returns:
        (res_zone_high, res_zone_low, sup_zone_high, sup_zone_low)
    """
    zone_buffer = (atr * zone_atr_mult).fillna(0.0)
    
    res_zone_high = resistance
    res_zone_low = resistance - zone_buffer

    sup_zone_high = support + zone_buffer
    sup_zone_low = support

    return res_zone_high, res_zone_low, sup_zone_high, sup_zone_low
