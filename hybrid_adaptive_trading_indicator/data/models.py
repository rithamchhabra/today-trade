"""
Data models and normalization routines for OHLCV candle data.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class Candle:
    timestamp: datetime  # UTC timestamp of candle open
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: Optional[datetime] = None  # UTC timestamp of candle close
    is_closed: bool = True

REQUIRED_COLUMNS = ["open", "high", "low", "close"]

def normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame columns and types.
    Ensures UTC datetime column, float OHLCV, sorted by datetime ascending, no duplicates.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    d = df.copy()
    # Normalize column names
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    
    aliases = {
        "time": "datetime",
        "timestamp": "datetime",
        "date": "datetime",
        "open_time": "datetime",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vol": "volume",
    }
    d = d.rename(columns={k: v for k, v in aliases.items() if k in d.columns and v not in d.columns})

    for col in REQUIRED_COLUMNS:
        if col not in d.columns:
            raise ValueError(f"Missing required price column: '{col}'")

    if "volume" not in d.columns:
        d["volume"] = 0.0

    # Handle datetime
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
        d = d.dropna(subset=["datetime"])
    else:
        # Default index
        d["datetime"] = pd.date_range(end=pd.Timestamp.now(tz=timezone.utc).floor("15min"), periods=len(d), freq="15min")

    for col in ["open", "high", "low", "close", "volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=REQUIRED_COLUMNS)
    d = d.drop_duplicates(subset=["datetime"], keep="last")
    d = d.sort_values("datetime").reset_index(drop=True)
    return d

def convert_to_display_timezone(df: pd.DataFrame, tz_name: str = "Asia/Kolkata") -> pd.DataFrame:
    """
    Produce a copy of DataFrame with a display_datetime column converted to the target timezone.
    Internal 'datetime' column remains UTC!
    """
    d = df.copy()
    if "datetime" in d.columns:
        if d["datetime"].dt.tz is None:
            utc_series = d["datetime"].dt.tz_localize("UTC")
        else:
            utc_series = d["datetime"].dt.tz_convert("UTC")
        d["display_datetime"] = utc_series.dt.tz_convert(tz_name)
    return d
