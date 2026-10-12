"""
Trending Strategy: Confirmed Support Breakdown / Resistance Breakout.
Requires:
- Market regime == TRENDING
- Directional indicator confirmation (+DI > -DI for BUY, -DI > +DI for SELL)
- Completed candle breakout close
- Optional volume and retest confirmation
- Duplicate breakout suppression
"""
from typing import Optional
import pandas as pd
from hybrid_adaptive_trading_indicator.strategy.models import TradeCandidate

def evaluate_trending_setup(
    df: pd.DataFrame,
    idx: int,
    symbol: str,
    timeframe: str,
    sl_atr_mult: float = 1.5,
    tp1_rr: float = 1.5,
    tp2_rr: float = 2.5,
    use_volume: bool = False,
    use_retest: bool = False,
    retest_bars: int = 3,
    zone_atr_mult: float = 0.20
) -> Optional[TradeCandidate]:
    """
    Evaluates candle at idx for trending breakout setup.
    """
    if idx < 1:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    if row["regime"] != "TRENDING":
        return None

    if pd.isna(row["support"]) or pd.isna(row["resistance"]) or pd.isna(row["atr"]) or row["atr"] <= 0:
        return None

    atr = float(row["atr"])
    support = float(row["support"])
    resistance = float(row["resistance"])
    zone_buffer = atr * zone_atr_mult

    vol_ok = True
    if use_volume:
        vol_ok = bool(row["is_vol_elevated"])

    c_close = float(row["close"])
    prev_close = float(prev["close"])
    plus_di = float(row["plus_di"])
    minus_di = float(row["minus_di"])

    # 1. Bullish Resistance Breakout (BUY)
    long_dir = plus_di > minus_di
    broke_up = (prev_close <= float(prev["resistance"])) and (c_close > resistance)
    
    # Handle retest if enabled
    retest_up_ok = True
    if use_retest:
        prior_slice = df.iloc[max(0, idx - retest_bars):idx]
        # Retest condition: price dipped near resistance and held above it
        tested_level = (prior_slice["low"] <= resistance + zone_buffer).any()
        held_level = c_close > resistance
        retest_up_ok = tested_level and held_level

    if broke_up and long_dir and vol_ok and retest_up_ok:
        entry = c_close
        # Swing or ATR SL
        sl = entry - (sl_atr_mult * atr)
        risk_dist = max(entry - sl, 0.01)

        tp1 = entry + (tp1_rr * risk_dist)
        tp2 = entry + (tp2_rr * risk_dist)
        reward_dist = tp1 - entry
        rr_ratio = reward_dist / risk_dist

        return TradeCandidate(
            symbol=symbol,
            timeframe=timeframe,
            candle_timestamp=pd.to_datetime(row["datetime"]).to_pydatetime(),
            strategy="TRENDING_BREAKOUT",
            regime="TRENDING",
            direction="BUY",
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            tp1=round(tp1, 4),
            tp2=round(tp2, 4),
            risk_distance=round(risk_dist, 4),
            reward_tp1_distance=round(reward_dist, 4),
            risk_reward_ratio=round(rr_ratio, 2),
            setup_reason=f"Confirmed resistance breakout ({resistance:.2f}) with +DI>{minus_di:.1f}" + (" + Retest" if use_retest else ""),
            retest_confirmed=use_retest,
            volume_confirmed=vol_ok
        )

    # 2. Bearish Support Breakdown (SELL)
    short_dir = minus_di > plus_di
    broke_down = (prev_close >= float(prev["support"])) and (c_close < support)

    retest_down_ok = True
    if use_retest:
        prior_slice = df.iloc[max(0, idx - retest_bars):idx]
        tested_level = (prior_slice["high"] >= support - zone_buffer).any()
        held_level = c_close < support
        retest_down_ok = tested_level and held_level

    if broke_down and short_dir and vol_ok and retest_down_ok:
        entry = c_close
        sl = entry + (sl_atr_mult * atr)
        risk_dist = max(sl - entry, 0.01)

        tp1 = entry - (tp1_rr * risk_dist)
        tp2 = entry - (tp2_rr * risk_dist)
        reward_dist = entry - tp1
        rr_ratio = reward_dist / risk_dist

        return TradeCandidate(
            symbol=symbol,
            timeframe=timeframe,
            candle_timestamp=pd.to_datetime(row["datetime"]).to_pydatetime(),
            strategy="TRENDING_BREAKOUT",
            regime="TRENDING",
            direction="SELL",
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            tp1=round(tp1, 4),
            tp2=round(tp2, 4),
            risk_distance=round(risk_dist, 4),
            reward_tp1_distance=round(reward_dist, 4),
            risk_reward_ratio=round(rr_ratio, 2),
            setup_reason=f"Confirmed support breakdown ({support:.2f}) with -DI>{plus_di:.1f}" + (" + Retest" if use_retest else ""),
            retest_confirmed=use_retest,
            volume_confirmed=vol_ok
        )

    return None
