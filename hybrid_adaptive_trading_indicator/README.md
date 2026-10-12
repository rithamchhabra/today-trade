# 🛡️ Hybrid Adaptive Trading Dashboard for Gold & Bitcoin

A professional Python-based algorithmic trading analysis dashboard for **Gold (XAUUSD)** and **Bitcoin (BTCUSDT)** built with **Streamlit**, **Plotly**, and **pandas**.

The system dynamically classifies market regimes (**SIDEWAYS**, **TRANSITION**, **TRENDING**) with hysteresis, automatically selects the appropriate trading logic, computes ATR-buffered stop-loss and take-profit targets, and evaluates setups through a **deterministic Risk Approval Engine** (`APPROVE`, `REJECT`, `WAIT`).

---

## ⚡ Key Features

1. **Dual Market Coverage**:
   - **Gold (XAUUSD)**: 15-minute OHLC candles via Yahoo Finance (`XAUUSD=X`) with an audited 1:1 PAX Gold (`PAXGUSDT`) fallback on public exchange endpoints.
   - **Bitcoin (BTCUSDT)**: 15-minute public klines via Binance Spot (`data-api.binance.vision` & `api.binance.com`), requiring no API keys.
2. **Strict Non-Repainting Invariant**:
   - Pivot highs/lows are confirmed only after the configured right-side bars complete (`pivot_right`).
   - Signals are evaluated on **closed candles only**. Past pivots, indicators, and signals never change when future candles arrive.
3. **Adaptive Regime Classification with Hysteresis**:
   - **SIDEWAYS** ($ADX < 20$): Range mean-reversion with confirmed support/resistance rejection candles.
   - **TRANSITION** ($20 \le ADX < 25$): Momentum developing / chop; no forced trades.
   - **TRENDING** ($ADX \ge 25$): Momentum breakout confirmation using $+DI > -DI$ (Long) or $-DI > +DI$ (Short).
   - Hysteresis buffer prevents flip-flopping around threshold boundaries.
4. **Deterministic Python Risk Engine**:
   - Enforces strict rules: Maximum Daily Loss, Maximum Concurrent Trades, Duplicate Trade Prevention, and Minimum Net Risk/Reward.
   - Dynamic position sizing based on instrument specifications (Gold 100 oz standard lots vs Bitcoin spot units) and trading friction (spread, slippage, commission).
5. **Paper Trading & Conservative Execution**:
   - Next-candle executable open fill model.
   - **Conservative Conflict Resolution**: If both Stop-Loss and Target are touched in the same bar, Stop-Loss is assumed hit first!
   - Trailing stop-loss updater and SQLite persistent journal.
6. **Built-in Backtest Engine**:
   - Bar-by-bar historical backtesting with equity curve plotting, win rate, profit factor, max drawdown, and average R-multiple.
7. **Optional Telegram Alerts**:
   - Sends formatted markdown trading alerts via environment variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).
8. **🤖 Google Gemini AI Advisor**:
   - One-click AI deep-dive review on any generated trade setup.
   - Audits whether Gemini agrees or disagrees with the deterministic Risk Engine's **APPROVE** or **REJECT** verdict.
   - Provides 15-minute Market Pulse commentary for Gold and Bitcoin momentum.
9. **📜 Audited Trade & Decision History**:
   - Complete historical log of all detected signals across instruments.
   - Displays whether each trade was **APPROVED** (🟢) or **REJECTED** (🔴) with full rejection reasons (e.g. Min R:R, Max concurrent trades, Daily loss limit).
   - Filterable by Asset and Decision Status with approval rate statistics.

---

## 🏗️ Architecture

```
hybrid_adaptive_trading_indicator/
├── config.py                 # Contract specifications, risk defaults, instrument constants
├── data/
│   ├── models.py             # OHLCV normalization & timezone conversion
│   ├── integrity.py          # Gap detection, duplicate removal, closed candle check
│   ├── binance_client.py     # Binance Spot public klines connector
│   ├── yfinance_client.py    # Yahoo Finance Gold connector
│   ├── gold_api_fallback.py  # Verified free gold proxy investigation & fallback
│   ├── generate_samples.py   # Offline sample generator
│   └── samples/              # Verified 15m offline datasets for Gold and BTC
├── indicators/
│   ├── adx.py                # Wilder's ADX, +DI, -DI
│   ├── atr.py                # Average True Range
│   ├── pivots.py             # Strict non-repainting pivot highs/lows
│   ├── zones.py              # S/R zones with ATR buffers
│   └── volume.py             # Volume MA & elevation filter
├── strategy/
│   ├── regime.py             # Regime classifier with hysteresis
│   ├── sideways.py           # Sideways range mean-reversion
│   ├── trending.py           # Trending breakout & momentum
│   └── engine.py             # Strategy coordinator & duplicate suppression
├── risk/
│   ├── position_sizer.py     # Position sizing for Gold lots & BTC units
│   ├── approval_engine.py    # APPROVE / REJECT / WAIT deterministic decision engine
│   └── models.py             # Risk decision dataclasses
├── execution/
│   ├── paper_trader.py       # Paper trade simulator & conservative fill model
│   └── models.py             # Paper trade models
├── storage/
│   └── database.py           # SQLite database for signals and trade journal
├── ai/
│   └── gemini_advisor.py     # Google Gemini AI trade advisor & market pulse
├── alerts/
│   └── telegram.py           # Telegram notification dispatcher
├── backtest/
│   ├── engine.py             # Historical backtest simulator
│   └── metrics.py            # Performance metrics & drawdown computation
├── tests/                    # 18 unit tests covering all components
├── app.py                    # Streamlit interactive dashboard
├── requirements.txt
└── .env.example
```

---

## 💻 Windows Installation & Startup

### 1. Prerequisites
- Python 3.11+ installed and added to PATH.

### 2. Install Dependencies
Open PowerShell or Command Prompt in the repository folder:

```powershell
pip install -r hybrid_adaptive_trading_indicator/requirements.txt
```

### 3. (Optional) Configure Telegram Alerts
Copy `.env.example` to `.env` and fill in your bot token:

```powershell
cp hybrid_adaptive_trading_indicator/.env.example hybrid_adaptive_trading_indicator/.env
```

### 4. Run the Dashboard
Launch the Streamlit app:

```powershell
python -m streamlit run hybrid_adaptive_trading_indicator/app.py
```
*(Alternatively, if `streamlit` is in your PATH, you can run `streamlit run hybrid_adaptive_trading_indicator/app.py`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the full pytest suite to verify indicators, non-repainting invariants, risk rules, and execution models:

```powershell
python -m pytest hybrid_adaptive_trading_indicator/tests -v
```

All 15 tests verify:
- Non-repainting pivot confirmation timing.
- Resistance and support invariance when future candles are appended.
- Position sizing on Gold (lots) vs Bitcoin (units).
- Deterministic risk approval rules (daily loss, concurrent positions, min R:R, duplicate protection).
- Paper trading lifecycle and conservative same-candle fill rule.
- Unfinished candle dropping and gap detection.

---

## 📡 Free Market Data Sources & Limitations

1. **Gold (`XAUUSD=X`)**:
   - Pulled via `yfinance`. Yahoo Finance is an unofficial public data feed with variable rate limiting and potential delays.
   - Markets are closed on weekends (Friday evening to Sunday evening UTC). When closed, the app alerts the user of stale data.
   - When Yahoo Finance is unreachable, the fallback engine can query Binance `PAXGUSDT` (Paxos Gold), which is backed 1:1 by London physical gold and provides 24/7 liquid 15m OHLC data.
2. **Bitcoin (`BTCUSDT`)**:
   - Pulled directly from Binance Spot's public endpoint (`https://data-api.binance.vision/api/v3/klines`).
   - Free, unauthenticated, and provides 24/7 continuous klines.
3. **Offline / Sample Data**:
   - Select **"Offline Verified Samples"** in the sidebar to run simulations and inspect charts without relying on network APIs.

---

## ⚠️ Research & Paper Trading Disclaimer
*This system is strictly a paper-trading and research tool. It does not place real orders on any brokerage or exchange. Cryptocurrency and commodity trading involve substantial financial risk. Past results or simulated backtests do not guarantee future profitability.*
