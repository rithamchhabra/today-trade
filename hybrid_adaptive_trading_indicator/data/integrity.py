"""
Data integrity, gap detection, duplicate handling, and closed-candle validation.
"""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class IntegrityReport:
    total_candles: int
    is_valid: bool
    duplicate_count: int
    gaps_detected: List[Dict[str, Any]]
    latest_candle_time: datetime
    is_stale: bool
    stale_duration_minutes: float
    unfinished_candle_dropped: bool
    warnings: List[str]

def validate_and_filter_candles(
    df: pd.DataFrame,
    expected_interval_minutes: int = 15,
    stale_threshold_minutes: int = 45,
    drop_unclosed: bool = True
) -> Tuple[pd.DataFrame, IntegrityReport]:
    """
    Validates candle data:
    1. Removes duplicates
    2. Drops any currently open / uncompleted candle
    3. Detects temporal gaps
    4. Evaluates data freshness/staleness against current UTC time
    """
    warnings: List[str] = []
    if df.empty:
        return df, IntegrityReport(
            total_candles=0,
            is_valid=False,
            duplicate_count=0,
            gaps_detected=[],
            latest_candle_time=datetime.min.replace(tzinfo=timezone.utc),
            is_stale=True,
            stale_duration_minutes=99999.0,
            unfinished_candle_dropped=False,
            warnings=["Dataset is empty."]
        )

    d = df.copy()
    initial_count = len(d)
    
    # 1. Duplicates check
    d = d.drop_duplicates(subset=["datetime"], keep="last")
    duplicate_count = initial_count - len(d)
    if duplicate_count > 0:
        warnings.append(f"Removed {duplicate_count} duplicate candle(s).")

    d = d.sort_values("datetime").reset_index(drop=True)

    # 2. Check for unfinished latest candle
    now_utc = datetime.now(timezone.utc)
    unfinished_dropped = False
    
    if drop_unclosed and len(d) > 0:
        latest_ts = d["datetime"].iloc[-1]
        # In 15m candles, the candle starting at latest_ts closes at latest_ts + 15m
        expected_close_time = latest_ts + timedelta(minutes=expected_interval_minutes)
        if expected_close_time > now_utc:
            # Candle is still forming!
            d = d.iloc[:-1].reset_index(drop=True)
            unfinished_dropped = True
            warnings.append(f"Dropped unfinished forming candle at {latest_ts.isoformat()}. Using closed candles only.")

    if d.empty:
        return d, IntegrityReport(
            total_candles=0,
            is_valid=False,
            duplicate_count=duplicate_count,
            gaps_detected=[],
            latest_candle_time=datetime.min.replace(tzinfo=timezone.utc),
            is_stale=True,
            stale_duration_minutes=99999.0,
            unfinished_candle_dropped=unfinished_dropped,
            warnings=warnings + ["No completed candles available."]
        )

    latest_completed_ts = d["datetime"].iloc[-1]
    
    # 3. Detect gaps
    gaps: List[Dict[str, Any]] = []
    if len(d) > 1:
        time_diffs = d["datetime"].diff()
        expected_delta = pd.Timedelta(minutes=expected_interval_minutes)
        gap_indices = d.index[time_diffs > (expected_delta * 1.5)]
        for idx in gap_indices:
            prev_ts = d.loc[idx - 1, "datetime"]
            curr_ts = d.loc[idx, "datetime"]
            gap_duration_mins = (curr_ts - prev_ts).total_seconds() / 60.0
            gaps.append({
                "from": prev_ts.isoformat(),
                "to": curr_ts.isoformat(),
                "missing_minutes": gap_duration_mins
            })
        if gaps:
            warnings.append(f"Detected {len(gaps)} temporal gap(s) (e.g. weekends or feed outages).")

    # 4. Check staleness
    time_since_last_candle = (now_utc - latest_completed_ts).total_seconds() / 60.0
    is_stale = time_since_last_candle > stale_threshold_minutes
    if is_stale:
        warnings.append(
            f"Data feed may be STALE: latest closed candle is {time_since_last_candle:.1f} minutes old (threshold: {stale_threshold_minutes} min)."
        )

    report = IntegrityReport(
        total_candles=len(d),
        is_valid=len(d) >= 20,
        duplicate_count=duplicate_count,
        gaps_detected=gaps,
        latest_candle_time=latest_completed_ts,
        is_stale=is_stale,
        stale_duration_minutes=time_since_last_candle,
        unfinished_candle_dropped=unfinished_dropped,
        warnings=warnings
    )
    return d, report
