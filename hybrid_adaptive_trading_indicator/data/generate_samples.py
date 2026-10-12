"""
Utility to generate offline realistic sample CSV files for Gold and Bitcoin 15m candles.
Used for offline testing, CI, and fallback demos.
"""
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

def generate_sample_dataset(
    symbol: str,
    base_price: float,
    periods: int = 500,
    volatility: float = 0.002
) -> pd.DataFrame:
    rng = np.random.default_rng(1337 if symbol == "XAUUSD" else 4242)
    end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # round down to 15m
    end_time = end_time - timedelta(minutes=end_time.minute % 15)
    start_time = end_time - timedelta(minutes=periods * 15)
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(periods)]

    prices = [base_price]
    regime = 0
    # Simulate realistic market regime shifts: alternating trending and sideways
    for i in range(1, periods):
        if i % 70 == 0:
            regime = int(rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]))
        
        drift = regime * (base_price * 0.0006)
        shock = rng.normal(0, base_price * volatility * (0.6 if regime == 0 else 1.2))
        next_price = max(base_price * 0.5, prices[-1] + drift + shock)
        prices.append(next_price)

    close = np.array(prices)
    prev = np.r_[close[0], close[:-1]]
    high = np.maximum(prev, close) + rng.uniform(0.1, 1.0, periods) * (base_price * volatility * 0.8)
    low = np.minimum(prev, close) - rng.uniform(0.1, 1.0, periods) * (base_price * volatility * 0.8)
    # Open is close of previous with small gap potential
    open_price = prev + rng.normal(0, base_price * 0.0001, periods)
    volume = rng.integers(500, 8000, periods).astype(float)
    if symbol == "BTCUSDT":
        volume = volume * 2.5

    df = pd.DataFrame({
        "datetime": [ts.isoformat() for ts in timestamps],
        "open": np.round(open_price, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": np.round(volume, 2)
    })
    return df

def create_sample_files(target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Gold (XAUUSD) sample around $2350
    gold_df = generate_sample_dataset("XAUUSD", base_price=2350.0, periods=600, volatility=0.0018)
    gold_path = os.path.join(target_dir, "xauusd_15m_sample.csv")
    gold_df.to_csv(gold_path, index=False)
    
    # 2. Bitcoin (BTCUSDT) sample around $63000
    btc_df = generate_sample_dataset("BTCUSDT", base_price=63500.0, periods=600, volatility=0.0035)
    btc_path = os.path.join(target_dir, "btcusdt_15m_sample.csv")
    btc_df.to_csv(btc_path, index=False)
    
    print(f"Sample files generated:\n- {gold_path}\n- {btc_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(current_dir, "samples")
    create_sample_files(sample_dir)
