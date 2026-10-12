"""
Yahoo Finance data client for Gold (XAUUSD=X).
Fetches 15-minute OHLC candles using yfinance.
Handles unavailable data, rate limits, missing candles, and network issues.
Explicitly treats Yahoo Finance as an unofficial polling source, never fabricating fake candles.
"""
import logging
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import yfinance as yf
from hybrid_adaptive_trading_indicator.config import YFINANCE_GOLD_SYMBOL
from hybrid_adaptive_trading_indicator.data.models import normalize_ohlcv_dataframe

# Suppress yfinance internal delisted/error noise on stdout/stderr
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

class YahooFinanceDataClient:
    def __init__(self, symbol: str = YFINANCE_GOLD_SYMBOL, interval: str = "15m", period: str = "30d"):
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.active_source = symbol

    def fetch_gold_candles(self, period: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch 15-minute candles for Gold (XAUUSD=X or GC=F futures).
        Raises ValueError or ConnectionError with clear diagnostic information if unavailable.
        """
        p = period or self.period
        
        # Priority symbols: configured symbol, then Gold Futures (GC=F) which has high-liquidity intraday data
        candidate_symbols = [self.symbol]
        if self.symbol != "GC=F":
            candidate_symbols.append("GC=F")

        for sym in candidate_symbols:
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(period=p, interval=self.interval, auto_adjust=False)
                
                if df is None or df.empty:
                    # Try fallback period if 30d failed
                    df = ticker.history(period="7d", interval=self.interval, auto_adjust=False)

                if df is not None and not df.empty:
                    df = df.reset_index()
                    col_map = {
                        "Datetime": "datetime",
                        "Date": "datetime",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume"
                    }
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    norm_df = normalize_ohlcv_dataframe(df)
                    if len(norm_df) > 0:
                        self.active_source = sym
                        return norm_df
            except Exception:
                continue

        raise ValueError(
            f"Yahoo Finance returned no intraday data for Gold symbols {candidate_symbols} with interval '{self.interval}'. "
            "Markets may be closed (weekend), symbol format may have changed, or Yahoo Finance rate limit triggered. "
            "No synthetic candles were fabricated."
        )
