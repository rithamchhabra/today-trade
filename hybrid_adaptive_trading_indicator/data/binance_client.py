"""
Binance Spot public market data client for BTCUSDT.
Queries public Binance endpoints (data-api.binance.vision or api.binance.com).
No API key required. Includes retry with exponential backoff and connection error handling.
"""
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import requests
import pandas as pd
from hybrid_adaptive_trading_indicator.config import BINANCE_PUBLIC_ENDPOINTS
from hybrid_adaptive_trading_indicator.data.models import normalize_ohlcv_dataframe

class BinanceDataClient:
    def __init__(self, timeout: int = 10, max_retries: int = 3, retry_delay: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.endpoints = BINANCE_PUBLIC_ENDPOINTS

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        limit: int = 500,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch public klines (candlesticks) from Binance.
        Klines array response format:
        [
          0: Open time (ms),
          1: Open,
          2: High,
          3: Low,
          4: Close,
          5: Volume,
          6: Close time (ms),
          7: Quote asset volume,
          8: Number of trades,
          9: Taker buy base asset volume,
          10: Taker buy quote asset volume,
          11: Ignore
        ]
        """
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms

        last_exception = None
        for endpoint in self.endpoints:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = requests.get(endpoint, params=params, timeout=self.timeout)
                    if response.status_code == 200:
                        raw_data = response.json()
                        if not isinstance(raw_data, list) or len(raw_data) == 0:
                            raise ValueError(f"Binance returned empty or invalid klines list from {endpoint}")
                        return self._parse_klines(raw_data)
                    elif response.status_code == 429:
                        # Rate limit reached
                        retry_after = int(response.headers.get("Retry-After", self.retry_delay * 2))
                        time.sleep(retry_after)
                    else:
                        response.raise_for_status()
                except Exception as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue

        raise ConnectionError(
            f"Failed to fetch {symbol} 15m klines from Binance after retries. Error: {last_exception}"
        )

    def _parse_klines(self, raw_data: list) -> pd.DataFrame:
        records = []
        for item in raw_data:
            open_time_ms = int(item[0])
            close_time_ms = int(item[6])
            records.append({
                "datetime": pd.to_datetime(open_time_ms, unit="ms", utc=True),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "close_time": pd.to_datetime(close_time_ms, unit="ms", utc=True)
            })
        df = pd.DataFrame(records)
        return normalize_ohlcv_dataframe(df)
