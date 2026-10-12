"""
Sideways Strategy: Confirmed Support & Resistance Zone Rejections (Mean Reversion).
Only active when market regime is SIDEWAYS.
"""
from typing import Optional
import pandas as pd
from hybrid_adaptive_trading_indicator.strategy.models import TradeCandidate

def evaluate_sideways_setup(
    df: pd.DataFrame,
    idx: int,
    symbol: str,
    timeframe: str,
    sl_atr_mult: float = 1.5,
    target_rr: float = 2.0,
    zone_atr_mult: float = 0.20,
    use_volume: bool = False
) -> Optional[TradeCandidate]:
    """
    Evaluates candle at idx for sideways support or resistance rejection.
    Assumes df contains: 'open', 'high', 'low', 'close', 'volume', 'atr', 'support', 'resistance',
    'sup_zone_high', 'res_zone_low', 'vol_ma', 'is_vol_elevated', 'regime'.
    """
    if idx < 1:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    if row["regime"] != "SIDEWAYS":
        return None

    if pd.isna(row["support"]) or pd.isna(row["resistance"]) or pd.isna(row["atr"]) or row["atr"] <= 0:
        return None

    atr = float(row["atr"])
    support = float(row["support"])
    resistance = float(row["resistance"])
    sup_zone_high = float(row["sup_zone_high"])
    res_zone_low = float(row["res_zone_low"])

    # Ensure range has logical height
    if resistance <= support:
        return None

    vol_ok = True
    if use_volume:
        vol_ok = bool(row["is_vol_elevated"])

    c_open = float(row["open"])
    c_high = float(row["high"])
    c_low = float(row["low"])
    c_close = float(row["close"])
    candle_range = max(c_high - c_low, 1e-6)

    # 1. Bullish Support Rejection (BUY)
    # Price tests support zone (low reaches zone, close remains at or above support)
    tested_support = (c_low <= sup_zone_high) and (c_close >= support)
    lower_wick_ratio = (min(c_open, c_close) - c_low) / candle_range
    bullish_candle = (c_close > c_open) and (c_close > float(prev["close"]))
    bullish_rejection = bullish_candle or (lower_wick_ratio >= 0.35 and c_close >= c_open)

    if tested_support and bullish_rejection and vol_ok:
        entry = c_close
        # SL outside swing low / support with ATR buffer
        swing_low = min(support, c_low)
        sl = swing_low - (sl_atr_mult * atr)
        risk_dist = max(entry - sl, 0.01)

        # TP1 at opposite resistance zone lower boundary
        tp1 = max(resistance - (zone_atr_mult * atr), entry + risk_dist)
        tp2 = entry + (target_rr * risk_dist)
        reward_dist = tp1 - entry
        rr_ratio = reward_dist / risk_dist

        return TradeCandidate(
            symbol=symbol,
            timeframe=timeframe,
            candle_timestamp=pd.to_datetime(row["datetime"]).to_pydatetime(),
            strategy="SIDEWAYS_REVERSION",
            regime="SIDEWAYS",
            direction="BUY",
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            tp1=round(tp1, 4),
            tp2=round(tp2, 4),
            risk_distance=round(risk_dist, 4),
            reward_tp1_distance=round(reward_dist, 4),
            risk_reward_ratio=round(rr_ratio, 2),
            setup_reason="Support zone test with confirmed bullish rejection candle",
            volume_confirmed=vol_ok
        )

    # 2. Bearish Resistance Rejection (SELL)
    # Price tests resistance zone (high reaches zone, close remains at or below resistance)
    tested_resistance = (c_high >= res_zone_low) and (c_close <= resistance)
    upper_wick_ratio = (c_high - max(c_open, c_close)) / candle_range
    bearish_candle = (c_close < c_open) and (c_close < float(prev["close"]))
    bearish_rejection = bearish_candle or (upper_wick_ratio >= 0.35 and c_close <= c_open)

    if tested_resistance and bearish_rejection and vol_ok:
        entry = c_close
        # SL outside swing high / resistance with ATR buffer
        swing_high = max(resistance, c_high)
        sl = swing_high + (sl_atr_mult * atr)
        risk_dist = max(sl - entry, 0.01)

        # TP1 at opposite support zone upper boundary
        tp1 = min(support + (zone_atr_mult * atr), entry - risk_dist)
        tp2 = entry - (target_rr * risk_dist)
        reward_dist = entry - tp1
        rr_ratio = reward_dist / risk_dist

        return TradeCandidate(
            symbol=symbol,
            timeframe=timeframe,
            candle_timestamp=pd.to_datetime(row["datetime"]).to_pydatetime(),
            strategy="SIDEWAYS_REVERSION",
            regime="SIDEWAYS",
            direction="SELL",
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            tp1=round(tp1, 4),
            tp2=round(tp2, 4),
            risk_distance=round(risk_dist, 4),
            reward_tp1_distance=round(reward_dist, 4),
            risk_reward_ratio=round(rr_ratio, 2),
            setup_reason="Resistance zone test with confirmed bearish rejection candle",
            volume_confirmed=vol_ok
        )

    return None
