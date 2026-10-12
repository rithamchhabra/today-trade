"""
Configuration and instrument specifications for Hybrid Adaptive Trading Indicator.
"""
from dataclasses import dataclass
from typing import Dict, Any

# Timezone settings
INTERNAL_TIMEZONE = "UTC"
DISPLAY_TIMEZONE = "Asia/Kolkata"

# Default indicator parameters
DEFAULT_ADX_LENGTH = 14
DEFAULT_ATR_LENGTH = 14
DEFAULT_ADX_SIDEWAYS = 20.0
DEFAULT_ADX_TRENDING = 25.0
DEFAULT_REGIME_HYSTERESIS = 1.0  # Buffer to prevent rapid regime flip-flopping

DEFAULT_PIVOT_LEFT = 3
DEFAULT_PIVOT_RIGHT = 3  # Confirmation bars (strictly non-repainting)
DEFAULT_ZONE_ATR_MULT = 0.20
DEFAULT_SL_ATR_MULT = 1.50
DEFAULT_TP1_RR = 1.50
DEFAULT_TP2_RR = 2.50
DEFAULT_VOLUME_MA_LEN = 20
DEFAULT_RETEST_BARS = 3

# Account & Risk defaults
DEFAULT_ACCOUNT_BALANCE = 10000.0  # USD
DEFAULT_RISK_PER_TRADE_PCT = 1.0   # 1.0% risk per trade
DEFAULT_MAX_DAILY_LOSS = 300.0     # USD
DEFAULT_MAX_CONCURRENT_TRADES = 2
DEFAULT_MIN_RISK_REWARD = 1.50

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    display_name: str
    asset_type: str  # "commodity" or "crypto"
    contract_size: float  # e.g. 100 troy oz per lot for Gold, 1 BTC for Bitcoin
    tick_size: float      # Minimum price increment, e.g. 0.01 for Gold, 0.01 for BTC
    tick_value: float     # USD value per tick per full contract/lot
    min_order_qty: float  # e.g. 0.01 lot for Gold, 0.001 BTC for Bitcoin
    qty_step: float       # Increment step
    default_spread: float # Expected spread in price points
    default_slippage: float # Expected slippage in price points
    default_fee_pct: float # Percentage fee per roundtrip or per leg

INSTRUMENT_SPECS: Dict[str, InstrumentSpec] = {
    "XAUUSD": InstrumentSpec(
        symbol="XAUUSD",
        display_name="Gold / USD (XAUUSD)",
        asset_type="commodity",
        contract_size=100.0,       # 1 Standard Lot = 100 oz
        tick_size=0.01,
        tick_value=1.0,           # 100 oz * 0.01 = $1.00 per tick per 1.0 lot
        min_order_qty=0.01,       # 0.01 lot (micro-lot)
        qty_step=0.01,
        default_spread=0.30,      # ~$0.30 typical retail spread
        default_slippage=0.15,
        default_fee_pct=0.0002    # ~0.02% or commission per lot
    ),
    "BTCUSDT": InstrumentSpec(
        symbol="BTCUSDT",
        display_name="Bitcoin / USDT (BTCUSDT)",
        asset_type="crypto",
        contract_size=1.0,        # 1 BTC
        tick_size=0.01,
        tick_value=0.01,          # $0.01 per tick per 1 BTC
        min_order_qty=0.001,      # 0.001 BTC
        qty_step=0.001,
        default_spread=5.0,       # ~$5.00 typical spot spread
        default_slippage=2.5,
        default_fee_pct=0.001     # 0.10% Binance spot taker fee
    )
}

# Data source configuration
BINANCE_PUBLIC_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines"
]

YFINANCE_GOLD_SYMBOL = "XAUUSD=X"
BINANCE_BTC_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "15m"
STALE_DATA_THRESHOLD_MINUTES = 45  # Threshold to trigger stale data warning
