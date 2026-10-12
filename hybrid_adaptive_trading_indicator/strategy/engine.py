"""
Strategy execution engine.
Computes indicators, classifies regimes, evaluates setups, and prevents duplicate signals.
"""
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from hybrid_adaptive_trading_indicator.indicators.adx import calculate_adx_di
from hybrid_adaptive_trading_indicator.indicators.atr import calculate_atr
from hybrid_adaptive_trading_indicator.indicators.pivots import calculate_confirmed_pivots
from hybrid_adaptive_trading_indicator.indicators.zones import calculate_sr_zones
from hybrid_adaptive_trading_indicator.indicators.volume import calculate_volume_ma
from hybrid_adaptive_trading_indicator.strategy.regime import classify_regimes
from hybrid_adaptive_trading_indicator.strategy.sideways import evaluate_sideways_setup
from hybrid_adaptive_trading_indicator.strategy.trending import evaluate_trending_setup
from hybrid_adaptive_trading_indicator.strategy.models import TradeCandidate

def prepare_indicators(
    df: pd.DataFrame,
    adx_len: int = 14,
    atr_len: int = 14,
    pivot_left: int = 3,
    pivot_right: int = 3,
    zone_atr_mult: float = 0.20,
    volume_len: int = 20,
    adx_sideways: float = 20.0,
    adx_trending: float = 25.0,
    regime_hysteresis: float = 1.0
) -> pd.DataFrame:
    """
    Computes all technical indicators and market regime strictly non-repainting.
    """
    d = df.copy()
    
    # 1. ADX and DIs
    adx, plus_di, minus_di = calculate_adx_di(d, length=adx_len)
    d["adx"] = adx
    d["plus_di"] = plus_di
    d["minus_di"] = minus_di

    # 2. ATR
    d["atr"] = calculate_atr(d, length=atr_len)

    # 3. Confirmed Pivots (Strictly non-repainting)
    pivot_high, pivot_low, resistance, support = calculate_confirmed_pivots(
        d, pivot_left=pivot_left, pivot_right=pivot_right
    )
    d["pivot_high"] = pivot_high
    d["pivot_low"] = pivot_low
    d["resistance"] = resistance
    d["support"] = support

    # 4. S/R Zones
    res_high, res_low, sup_high, sup_low = calculate_sr_zones(
        resistance, support, d["atr"], zone_atr_mult=zone_atr_mult
    )
    d["res_zone_high"] = res_high
    d["res_zone_low"] = res_low
    d["sup_zone_high"] = sup_high
    d["sup_zone_low"] = sup_low

    # 5. Volume MA
    vol_ma, is_elevated = calculate_volume_ma(d, length=volume_len)
    d["vol_ma"] = vol_ma
    d["is_vol_elevated"] = is_elevated

    # 6. Market Regime with Hysteresis
    d["regime"] = classify_regimes(
        d["adx"],
        sideways_thresh=adx_sideways,
        trending_thresh=adx_trending,
        hysteresis=regime_hysteresis
    )

    return d

def evaluate_all_candidates(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "15m",
    sl_atr_mult: float = 1.5,
    target_rr: float = 2.0,
    tp1_rr: float = 1.5,
    tp2_rr: float = 2.5,
    zone_atr_mult: float = 0.20,
    use_volume: bool = False,
    use_retest: bool = False,
    retest_bars: int = 3,
    min_cooldown_bars: int = 4
) -> Tuple[pd.DataFrame, List[TradeCandidate]]:
    """
    Generates trading setups candle-by-candle and suppresses duplicate consecutive triggers.
    """
    d = df.copy()
    d["signal"] = ""
    d["signal_reason"] = ""
    d["entry_price"] = np.nan
    d["stop_loss"] = np.nan
    d["tp1"] = np.nan
    d["tp2"] = np.nan
    d["risk_reward"] = np.nan

    candidates: List[TradeCandidate] = []
    last_signal_idx: Dict[str, int] = {"BUY": -999, "SELL": -999}
    last_breakout_level: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

    for i in range(1, len(d)):
        row = d.iloc[i]
        regime = row["regime"]

        candidate: Optional[TradeCandidate] = None

        if regime == "SIDEWAYS":
            candidate = evaluate_sideways_setup(
                df=d,
                idx=i,
                symbol=symbol,
                timeframe=timeframe,
                sl_atr_mult=sl_atr_mult,
                target_rr=target_rr,
                zone_atr_mult=zone_atr_mult,
                use_volume=use_volume
            )
        elif regime == "TRENDING":
            candidate = evaluate_trending_setup(
                df=d,
                idx=i,
                symbol=symbol,
                timeframe=timeframe,
                sl_atr_mult=sl_atr_mult,
                tp1_rr=tp1_rr,
                tp2_rr=tp2_rr,
                use_volume=use_volume,
                use_retest=use_retest,
                retest_bars=retest_bars,
                zone_atr_mult=zone_atr_mult
            )

        if candidate is not None:
            # Duplicate signal suppression:
            # If same direction fired within cooldown bars, suppress duplicate
            bars_since_last = i - last_signal_idx[candidate.direction]
            if bars_since_last < min_cooldown_bars:
                continue

            # If trending breakout on the exact same resistance/support level, avoid immediate repeat
            if candidate.strategy == "TRENDING_BREAKOUT":
                level = row["resistance"] if candidate.direction == "BUY" else row["support"]
                if level == last_breakout_level[candidate.direction] and bars_since_last < 8:
                    continue
                last_breakout_level[candidate.direction] = level

            last_signal_idx[candidate.direction] = i
            candidates.append(candidate)

            # Record in DataFrame
            d.loc[d.index[i], "signal"] = candidate.direction
            d.loc[d.index[i], "signal_reason"] = candidate.setup_reason
            d.loc[d.index[i], "entry_price"] = candidate.entry_price
            d.loc[d.index[i], "stop_loss"] = candidate.stop_loss
            d.loc[d.index[i], "tp1"] = candidate.tp1
            d.loc[d.index[i], "tp2"] = candidate.tp2
            d.loc[d.index[i], "risk_reward"] = candidate.risk_reward_ratio

    return d, candidates
