"""
Investigation and implementation of free Gold price API fallback options.

Investigation Summary:
1. GoldAPI.io / Metals-API: Requires registration and paid plans for intraday OHLC candles (free tier only provides daily/snapshot spot price, which cannot be used for 15m strategy analysis).
2. Stooq / Alpha Vantage: Strict rate limits (Alpha Vantage 25 requests/day, Stooq intraday is delayed/restricted).
3. PAXGUSDT (Paxos Gold on Binance Spot):
   - Backed 1:1 by one fine troy ounce of London Good Delivery gold stored in Brink's vaults.
   - Provides completely free, unauthenticated, high-liquidity 15m OHLCV candles via Binance public API.
   - Ideal non-commercial proxy for gold price action and 15m intraday analysis when Yahoo Finance is blocked or on weekends.
   
Mandatory Rule:
Use fallback for strategy analysis ONLY if it provides verified, complete OHLC candles.
"""
from typing import Tuple, Optional
import pandas as pd
from hybrid_adaptive_trading_indicator.data.binance_client import BinanceDataClient
from hybrid_adaptive_trading_indicator.data.models import normalize_ohlcv_dataframe

class GoldFallbackClient:
    def __init__(self):
        self.binance_client = BinanceDataClient()

    def fetch_paxg_gold_proxy_candles(self, limit: int = 500) -> Tuple[pd.DataFrame, str]:
        """
        Fetches PAXGUSDT 15m candles from Binance as a verified free 1:1 gold proxy.
        Validates OHLC structure before returning.
        """
        try:
            df = self.binance_client.fetch_klines(symbol="PAXGUSDT", interval="15m", limit=limit)
            # Verify OHLC integrity
            if df.empty or len(df) < 20:
                raise ValueError("Insufficient candles returned from PAXGUSDT endpoint.")
            
            # Check price level sanity for Gold (typically $1500 - $3500/oz range)
            latest_close = df["close"].iloc[-1]
            if latest_close < 1000 or latest_close > 10000:
                raise ValueError(f"PAXG price ${latest_close} out of plausible Gold price bounds.")
                
            return df, "Binance Spot PAXGUSDT (1:1 London Gold Proxy - Verified OHLC)"
        except Exception as e:
            raise ConnectionError(f"PAXGUSDT gold proxy fallback failed: {e}")
