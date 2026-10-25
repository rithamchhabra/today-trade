"""
Hybrid Adaptive Trading Dashboard for Gold (XAUUSD) and Bitcoin (BTCUSDT).
Features:
- Automatic market regime classification (Sideways, Transition, Trending)
- Non-repainting pivot S/R zones and multi-indicator analysis (ADX, +DI/-DI, ATR, Volume MA)
- Deterministic Risk Approval Engine (APPROVE / REJECT / WAIT)
- Paper trading lifecycle simulator & SQLite persistent journal
- Historical backtest reporting
- Timezones: UTC internal, Asia/Kolkata display
"""
import os
import sys
from datetime import datetime, timezone
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from hybrid_adaptive_trading_indicator.config import (
    INSTRUMENT_SPECS, DEFAULT_ACCOUNT_BALANCE, DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_MAX_DAILY_LOSS, DEFAULT_MAX_CONCURRENT_TRADES, DEFAULT_MIN_RISK_REWARD,
    DEFAULT_ADX_LENGTH, DEFAULT_ATR_LENGTH, DEFAULT_ADX_SIDEWAYS, DEFAULT_ADX_TRENDING,
    DEFAULT_PIVOT_LEFT, DEFAULT_PIVOT_RIGHT, DEFAULT_ZONE_ATR_MULT, DEFAULT_SL_ATR_MULT,
    DEFAULT_TP1_RR, DEFAULT_TP2_RR, DEFAULT_VOLUME_MA_LEN, DEFAULT_RETEST_BARS,
    DISPLAY_TIMEZONE, STALE_DATA_THRESHOLD_MINUTES
)
from hybrid_adaptive_trading_indicator.data.models import normalize_ohlcv_dataframe, convert_to_display_timezone
from hybrid_adaptive_trading_indicator.data.integrity import validate_and_filter_candles
from hybrid_adaptive_trading_indicator.data.binance_client import BinanceDataClient
from hybrid_adaptive_trading_indicator.data.yfinance_client import YahooFinanceDataClient
from hybrid_adaptive_trading_indicator.data.gold_api_fallback import GoldFallbackClient
from hybrid_adaptive_trading_indicator.strategy.engine import prepare_indicators, evaluate_all_candidates
from hybrid_adaptive_trading_indicator.risk.approval_engine import RiskApprovalEngine
from hybrid_adaptive_trading_indicator.risk.position_sizer import calculate_position_size
from hybrid_adaptive_trading_indicator.execution.paper_trader import PaperTradingSimulator
from hybrid_adaptive_trading_indicator.execution.models import PaperTrade
from hybrid_adaptive_trading_indicator.storage.database import DatabaseManager
from hybrid_adaptive_trading_indicator.alerts.telegram import TelegramAlertDispatcher
from hybrid_adaptive_trading_indicator.backtest.engine import run_backtest
from hybrid_adaptive_trading_indicator.ai.gemini_advisor import GeminiAdvisor
from hybrid_adaptive_trading_indicator.scanner import BackgroundMarketScanner, start_background_scanner

# Streamlit Page Setup
st.set_page_config(
    page_title="Hybrid Adaptive Trading Dashboard | Gold & Bitcoin",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode & Trading Terminal Palette)
st.markdown("""
<style>
    .metric-card {
        background-color: #131722;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .badge-sideways {
        background-color: #2b3990;
        color: #8bb4f8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-transition {
        background-color: #614400;
        color: #ffcc00;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-trending {
        background-color: #0b6623;
        color: #55efc4;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-approve {
        background-color: #00b894;
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-reject {
        background-color: #d63031;
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-wait {
        background-color: #4b6584;
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize singletons in session state
if "db" not in st.session_state:
    st.session_state.db = DatabaseManager()
if "telegram" not in st.session_state or not hasattr(st.session_state.telegram, "start_listener"):
    st.session_state.telegram = TelegramAlertDispatcher()
if "gemini" not in st.session_state:
    st.session_state.gemini = GeminiAdvisor()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.title("⚙️ Terminal Controls")

    st.subheader("⚡ Live Auto-Refresh")
    auto_refresh_rate = st.selectbox(
        "Auto-Refresh Interval",
        ["10s (Fast Live)", "15s", "30s", "60s", "Manual Only"],
        index=0,
        help="Automatically polls the exchange and updates live prices, indicators, and AI decisions without manual refresh."
    )
    if st.button("🔄 Refresh Live Prices Now", key="btn_refresh_now"):
        st.cache_data.clear()
        st.rerun()

    refresh_seconds = None
    if auto_refresh_rate == "10s (Fast Live)":
        refresh_seconds = 10
    elif auto_refresh_rate == "15s":
        refresh_seconds = 15
    elif auto_refresh_rate == "30s":
        refresh_seconds = 30
    elif auto_refresh_rate == "60s":
        refresh_seconds = 60

    st.divider()
    data_source_mode = st.radio(
        "Data Mode",
        ["Live Feeds", "Offline Verified Samples", "Custom CSV Upload"],
        help="Choose between live public exchange feeds, verified offline sample data, or your own CSV."
    )

    uploaded_file = None
    if data_source_mode == "Custom CSV Upload":
        uploaded_file = st.file_uploader("Upload 15m OHLCV CSV", type=["csv"])

    st.subheader("Strategy Parameters")
    adx_len = st.number_input("ADX / DI Length", min_value=5, max_value=50, value=DEFAULT_ADX_LENGTH)
    atr_len = st.number_input("ATR Length", min_value=5, max_value=50, value=DEFAULT_ATR_LENGTH)
    adx_sideways = st.slider("Sideways Threshold (ADX <)", 10, 30, int(DEFAULT_ADX_SIDEWAYS))
    adx_trending = st.slider("Trending Threshold (ADX >=)", 15, 45, int(DEFAULT_ADX_TRENDING))
    regime_hysteresis = st.slider("Regime Hysteresis Buffer", 0.5, 3.0, 1.0, 0.25)
    
    st.divider()
    st.subheader("Pivots & Zone Levels")
    pivot_left = st.slider("Pivot Left Bars", 1, 10, DEFAULT_PIVOT_LEFT)
    pivot_right = st.slider("Pivot Right Bars (Non-Repainting Confirmation)", 1, 10, DEFAULT_PIVOT_RIGHT)
    zone_atr_mult = st.slider("S/R Zone Width (ATR Mult)", 0.05, 0.50, DEFAULT_ZONE_ATR_MULT, 0.05)
    sl_atr_mult = st.slider("Stop-Loss ATR Multiplier", 0.5, 4.0, DEFAULT_SL_ATR_MULT, 0.25)
    target_rr = st.slider("Target Risk/Reward Ratio", 1.0, 5.0, DEFAULT_TP1_RR, 0.25)

    use_volume = st.checkbox("Volume Confirmation Filter", value=False)
    use_retest = st.checkbox("Require Breakout Retest", value=False)
    retest_bars = st.slider("Retest Lookback Bars", 1, 10, DEFAULT_RETEST_BARS, disabled=not use_retest)

    st.divider()
    st.subheader("Deterministic Risk Limits")
    account_balance = st.number_input("Account Balance ($)", min_value=500.0, max_value=1_000_000.0, value=DEFAULT_ACCOUNT_BALANCE, step=500.0)
    risk_pct = st.slider("Risk Per Trade (%)", 0.25, 5.0, DEFAULT_RISK_PER_TRADE_PCT, 0.25)
    max_daily_loss = st.number_input("Max Daily Loss Limit ($)", min_value=50.0, max_value=50000.0, value=DEFAULT_MAX_DAILY_LOSS, step=50.0)
    max_concurrent_trades = st.number_input("Max Concurrent Trades", min_value=1, max_value=10, value=DEFAULT_MAX_CONCURRENT_TRADES)
    min_rr_filter = st.slider("Min Risk/Reward Required", 1.0, 3.0, DEFAULT_MIN_RISK_REWARD, 0.25)

    st.divider()
    st.subheader("🧠 Decision Authority")
    decision_mode = st.selectbox(
        "Trade Approval Engine",
        [
            "🤖 Gemini AI Autonomous (AI Approves/Rejects)",
            "🛡️ Hybrid (Risk Rules + AI Dual Confirmation)",
            "⚙️ Strict Risk Engine Rules Only"
        ],
        index=0,
        help="Select whether Google Gemini AI makes the final trade approval/rejection decision."
    )

    st.divider()
    st.subheader("📱 Telegram Alerts (@Todaytrade_bot)")
    tg_token = st.text_input(
        "Bot Token",
        value=st.session_state.telegram.bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "8859941208:AAFuNovus_6YynDiSlHyuOq68xqcBvlPPMc"),
        type="password"
    )
    st.session_state.telegram.bot_token = tg_token.strip()

    tg_chat = st.text_input(
        "Your Chat ID",
        value=st.session_state.telegram.chat_id or os.getenv("TELEGRAM_CHAT_ID", "995787342"),
        help="Your numeric Telegram Chat ID (discovered: 995787342)."
    )
    st.session_state.telegram.chat_id = str(tg_chat).strip()

    if st.button("🔍 Auto-Detect My Chat ID (@Todaytrade_bot)"):
        with st.spinner("Checking Telegram updates..."):
            res = st.session_state.telegram.auto_discover_chat_id()
            if res["success"]:
                st.success(res["message"])
                st.rerun()
            else:
                st.warning(res["message"])

    enable_telegram = st.checkbox("Alert on Approved Trades", value=True)
    send_telegram_closed = st.checkbox("Alert on Target & SL Hits (P&L + Balance)", value=True, help="Send instant Telegram notification when Take-Profit or Stop-Loss is hit")
    send_telegram_rejections = st.checkbox("Alert on Rejected Trades", value=True)
    auto_paper_trade = st.checkbox("⚡ Auto-Simulate Approved Trades", value=True, help="Automatically open paper trades when signals are approved so Target/SL and P&L are tracked")
    enable_trailing_stop = st.checkbox("Dynamic Trailing Stop-Loss", value=True)

    # Start background interactive Telegram command listener (listens for 'p/l', 'balance', AI chat queries, etc.)
    if st.session_state.telegram.is_configured:
        st.session_state.telegram.start_listener(
            db_manager=st.session_state.db,
            starting_balance_fn=lambda: float(account_balance),
            gemini_advisor=st.session_state.gemini
        )
        st.caption("💬 **Interactive Bot & AI Active:** Send `p/l`, `status` or ask any trading question in Telegram!")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🔔 Test Alert"):
            res = st.session_state.telegram.send_test_message()
            if res["success"]:
                st.success(res["message"])
            else:
                st.error(res["message"] + " (Tip: Open t.me/Todaytrade_bot and press START)")
    with col_t2:
        if st.button("🛑 Test Reject Alert"):
            res = st.session_state.telegram.send_test_rejection_message()
            if res["success"]:
                st.success(res["message"])
            else:
                st.error(res["message"])
    st.divider()
    st.subheader("⚡ 24/7 Background Scanner")
    scanner_instance = BackgroundMarketScanner.get_instance()
    # Synchronize current user configuration to scanner
    scanner_instance.account_balance = float(account_balance)
    scanner_instance.risk_pct = float(risk_pct)
    scanner_instance.max_daily_loss = float(max_daily_loss)
    scanner_instance.max_concurrent_trades = int(max_concurrent_trades)
    scanner_instance.min_rr = float(min_rr_filter)
    scanner_instance.enable_telegram_signals = bool(enable_telegram)
    scanner_instance.enable_telegram_closed = bool(send_telegram_closed)
    scanner_instance.enable_telegram_rejections = bool(send_telegram_rejections)
    scanner_instance.auto_paper_trade = bool(auto_paper_trade)
    scanner_instance.decision_mode = decision_mode

    # Auto-start scanner if not running
    if not scanner_instance.is_running:
        scanner_instance.start(send_startup_alert=False)

    kolkata_tz = pytz.timezone(DISPLAY_TIMEZONE)
    last_sc_str = scanner_instance.last_scan_time.astimezone(kolkata_tz).strftime("%I:%M:%S %p IST") if scanner_instance.last_scan_time else "Starting first scan..."
    st.markdown(
        f"<div style='background: #161b22; padding: 10px 14px; border-radius: 6px; border: 1px solid #238636; margin-bottom: 8px;'>"
        f"<span style='color: #2ea043; font-weight: 700;'>🟢 24/7 Background Scanner: Active</span><br>"
        f"<span style='color: #8b949e; font-size: 0.8rem;'>Scanning Gold & BTC every 60s without browser<br>Last scan: <b style='color: #f0f6fc;'>{last_sc_str}</b></span>"
        f"</div>",
        unsafe_allow_html=True
    )
    if st.button("⚡ Trigger Instant Background Scan", key="btn_trigger_instant_scan"):
        with st.spinner("Scanning Gold & Bitcoin now..."):
            res = scanner_instance.scan_once()
            st.success(f"Scan finished at {res.get('time', 'now')}!")
            st.rerun()

    st.divider()
    st.subheader("🤖 Google Gemini AI")
    gemini_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini.api_key or os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Enter Google Gemini API key to enable AI trade setup reviews and market intelligence."
    )
    gemini_model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro"],
        index=0
    )
    if gemini_key_input:
        st.session_state.gemini.set_api_key(gemini_key_input)
        st.session_state.gemini.model_name = gemini_model_choice
        st.markdown("<span style='color: #00b894; font-weight: 600;'>🟢 Gemini AI Active</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color: #ffcc00; font-weight: 600;'>🟡 Enter API Key or set GEMINI_API_KEY</span>", unsafe_allow_html=True)

    st.divider()
    st.caption("🔒 Strict Non-Repainting: Signals evaluated solely on completed candles. No live orders executed.")

def render_plotly_chart(fig, **kwargs):
    """Safely render Plotly chart across Streamlit versions without deprecation warnings."""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)

def render_dataframe_table(df_data, **kwargs):
    """Safely render DataFrame table across Streamlit versions without deprecation warnings."""
    try:
        st.dataframe(df_data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df_data, use_container_width=True, **kwargs)

# ----------------- DATA LOADER ROUTINE -----------------
@st.cache_data(ttl=10, show_spinner=False)
def load_symbol_data(symbol_key: str, mode: str, uploaded_bytes: bytes = None):
    """
    Loads and normalizes 15-minute candle data for XAUUSD or BTCUSDT.
    Uses 10s cache TTL for real-time live price responsiveness.
    """
    if mode == "Custom CSV Upload" and uploaded_bytes is not None:
        import io
        raw_df = pd.read_csv(io.BytesIO(uploaded_bytes))
        df = normalize_ohlcv_dataframe(raw_df)
        clean_df, report = validate_and_filter_candles(df)
        return clean_df, report, "Custom Uploaded CSV"

    if mode == "Offline Verified Samples":
        sample_filename = "xauusd_15m_sample.csv" if symbol_key == "XAUUSD" else "btcusdt_15m_sample.csv"
        sample_path = os.path.join(BASE_DIR, "data", "samples", sample_filename)
        if not os.path.exists(sample_path):
            from hybrid_adaptive_trading_indicator.data.generate_samples import create_sample_files
            create_sample_files(os.path.join(BASE_DIR, "data", "samples"))
        raw_df = pd.read_csv(sample_path)
        df = normalize_ohlcv_dataframe(raw_df)
        clean_df, report = validate_and_filter_candles(df, drop_unclosed=False)
        return clean_df, report, f"Verified Offline Sample Dataset ({sample_filename})"

    # Live Feeds mode
    if symbol_key == "BTCUSDT":
        client = BinanceDataClient()
        df = client.fetch_klines(symbol="BTCUSDT", interval="15m", limit=500)
        clean_df, report = validate_and_filter_candles(df)
        return clean_df, report, "Binance Spot Public Market Data API (24/7 Live)"

    elif symbol_key == "XAUUSD":
        try:
            client = YahooFinanceDataClient()
            df = client.fetch_gold_candles(period="30d")
            clean_df, report = validate_and_filter_candles(df)

            # If Yahoo Gold data is stale by > 120 minutes (e.g. weekend market closure), switch to 24/7 PAX Gold spot proxy!
            if report.is_stale and report.stale_duration_minutes > 120:
                try:
                    fallback_client = GoldFallbackClient()
                    paxg_df, source_label = fallback_client.fetch_paxg_gold_proxy_candles(limit=500)
                    clean_paxg, paxg_report = validate_and_filter_candles(paxg_df)
                    paxg_report.warnings.append("Traditional Forex Gold market is closed for the weekend. Switched to 24/7 PAX Gold spot live feed.")
                    return clean_paxg, paxg_report, f"{source_label} [Weekend 24/7 Live]"
                except Exception:
                    pass

            return clean_df, report, f"Yahoo Finance ({getattr(client, 'active_source', 'Gold')} 15m) [Unofficial Delayed Feed]"
        except Exception as yf_err:
            # Attempt verified PAX Gold proxy fallback
            try:
                fallback_client = GoldFallbackClient()
                df, source_label = fallback_client.fetch_paxg_gold_proxy_candles(limit=500)
                clean_df, report = validate_and_filter_candles(df)
                report.warnings.append(f"Yahoo Finance unavailable ({yf_err}). Used verified PAXG spot gold proxy.")
                return clean_df, report, f"{source_label} [24/7 Live Proxy]"
            except Exception as fb_err:
                raise ConnectionError(f"Yahoo Finance failed ({yf_err}) and Gold fallback proxy failed ({fb_err}).")

# ----------------- MAIN DASHBOARD & GLOBAL BALANCE BAR -----------------
st.title("🛡️ Hybrid Adaptive Trading Dashboard")
st.caption(f"Intraday 15-Minute Strategy • Regime Hysteresis • Non-Repainting S/R • Deterministic Risk Engine • Display: {DISPLAY_TIMEZONE}")

# Real-time Portfolio Balance & Performance Metrics Bar
all_pt_records = st.session_state.db.get_all_paper_trades(limit=500)
realized_pnl_all = sum(t.get("pnl_net", 0.0) for t in all_pt_records if str(t.get("status", "")).startswith("CLOSED"))
open_pt = [t for t in all_pt_records if t.get("status") == "OPEN"]
closed_pt = [t for t in all_pt_records if str(t.get("status", "")).startswith("CLOSED")]
tp_wins = [t for t in closed_pt if "TP" in str(t.get("status", ""))]
sl_losses = [t for t in closed_pt if "SL" in str(t.get("status", "")) or "TRAILING" in str(t.get("status", ""))]
overall_win_rate = (len(tp_wins) / len(closed_pt) * 100) if closed_pt else 0.0
curr_balance = float(account_balance) + realized_pnl_all

b_c1, b_c2, b_c3, b_c4, b_c5 = st.columns(5)
b_c1.metric("💰 Current Balance", f"${curr_balance:,.2f}", delta=f"{realized_pnl_all:+,.2f} Net P&L" if realized_pnl_all != 0 else None)
b_c2.metric("🏦 Starting Capital", f"${account_balance:,.2f}")
b_c3.metric("🎯 Target Hits (Wins)", f"{len(tp_wins)} trades")
b_c4.metric("🛑 Stop-Loss Hits (Loss)", f"{len(sl_losses)} trades")
b_c5.metric("⚡ Open Positions", f"{len(open_pt)} active", delta=f"{overall_win_rate:.1f}% Win Rate" if closed_pt else "No closed trades")
st.markdown("<hr style='margin: 8px 0 16px 0;'>", unsafe_allow_html=True)

tab_gold, tab_btc, tab_history, tab_journal, tab_backtest = st.tabs([
    "🥇 Gold (XAUUSD)",
    "₿ Bitcoin (BTCUSDT)",
    "📜 Trade & Decision History",
    "📖 Paper Trading Journal",
    "📊 Backtest & Performance"
])

@st.fragment(run_every=refresh_seconds)
def render_instrument_tab(symbol_key: str):
    spec = INSTRUMENT_SPECS[symbol_key]
    uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

    # Load Data
    try:
        candles_df, integrity_report, feed_source = load_symbol_data(symbol_key, data_source_mode, uploaded_bytes)
    except Exception as e:
        st.error(f"❌ Market Data Fetch Error for {symbol_key}: {e}")
        st.info("💡 Tip: Switch to 'Offline Verified Samples' in the sidebar to inspect strategy logic without network dependencies.")
        return

    # Live Status Header
    now_ist = datetime.now(pytz.timezone(DISPLAY_TIMEZONE)).strftime("%I:%M:%S %p")
    refresh_text = f"Auto-refreshing every {refresh_seconds}s" if refresh_seconds else "Manual refresh only"
    st.markdown(
        f"<div style='display: flex; justify-content: space-between; align-items: center; background: #161b22; padding: 6px 14px; border-radius: 6px; border: 1px solid #30363d; margin-bottom: 12px; font-size: 0.85rem; color: #8b949e;'>"
        f"<span>🟢 <b>Live Price Polling Active</b> • Feed: <span style='color: #58a6ff;'>{feed_source}</span></span>"
        f"<span>🕒 Last Polled: <b style='color: #f0f6fc;'>{now_ist} IST</b> ({refresh_text})</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Data Integrity & Warnings
    if integrity_report.is_stale:
        st.warning(
            f"⚠️ STALE DATA WARNING: Latest closed candle is {integrity_report.stale_duration_minutes:.1f} minutes old. "
            "Markets may be closed (weekend) or polling may be delayed. Yahoo Finance is not a real-time streaming feed."
        )

    for w in integrity_report.warnings:
        st.caption(f"ℹ️ {w}")

    # Prepare Indicators
    ind_df = prepare_indicators(
        df=candles_df,
        adx_len=int(adx_len),
        atr_len=int(atr_len),
        pivot_left=int(pivot_left),
        pivot_right=int(pivot_right),
        zone_atr_mult=float(zone_atr_mult),
        volume_len=DEFAULT_VOLUME_MA_LEN,
        adx_sideways=float(adx_sideways),
        adx_trending=float(adx_trending),
        regime_hysteresis=float(regime_hysteresis)
    )

    # Evaluate Setups
    annotated_df, candidates = evaluate_all_candidates(
        df=ind_df,
        symbol=symbol_key,
        timeframe="15m",
        sl_atr_mult=float(sl_atr_mult),
        target_rr=float(target_rr),
        tp1_rr=float(target_rr),
        tp2_rr=float(target_rr * 1.5),
        zone_atr_mult=float(zone_atr_mult),
        use_volume=use_volume,
        use_retest=use_retest,
        retest_bars=int(retest_bars)
    )

    # ----------------- SIMULATE & UPDATE ACTIVE PAPER TRADES -----------------
    raw_pending_open = [
        PaperTrade.from_dict(t) for t in st.session_state.db.get_all_paper_trades(limit=100)
        if t["symbol"] == symbol_key and t["status"] in ["PENDING", "OPEN"]
    ]
    if raw_pending_open:
        sim_tracker = PaperTradingSimulator(
            enable_trailing_stop=enable_trailing_stop,
            trailing_atr_mult=1.5
        )
        updated_pairs = sim_tracker.update_trades_against_dataframe(raw_pending_open, annotated_df)
        for u_trade, was_closed in updated_pairs:
            st.session_state.db.save_or_update_trade(u_trade)
            if was_closed:
                # Recalculate global balance for notification
                all_t_now = st.session_state.db.get_all_paper_trades(limit=500)
                tot_pnl_now = sum(t.get("pnl_net", 0.0) for t in all_t_now if str(t.get("status", "")).startswith("CLOSED"))
                upd_bal = float(account_balance) + tot_pnl_now
                
                # Instant Telegram alert with TP/SL outcome, PnL and Balance
                if enable_telegram and send_telegram_closed:
                    st.session_state.telegram.send_trade_closed_alert(
                        u_trade,
                        starting_balance=float(account_balance),
                        current_balance=upd_bal
                    )
                    st.session_state.db.record_alert(
                        u_trade.symbol,
                        u_trade.direction,
                        f"Trade {u_trade.trade_id} closed: {u_trade.status} (PnL ${u_trade.pnl_net:+,.2f})",
                        True
                    )

    # Check Latest Candle
    last_candle = annotated_df.iloc[-1]
    last_utc = pd.to_datetime(last_candle["datetime"]).to_pydatetime()
    kolkata_tz = pytz.timezone(DISPLAY_TIMEZONE)
    last_ist_str = last_utc.astimezone(kolkata_tz).strftime("%Y-%m-%d %H:%M:%S IST")

    # Evaluate with Risk Engine
    risk_engine = RiskApprovalEngine(
        account_balance=float(account_balance),
        risk_per_trade_pct=float(risk_pct),
        max_daily_loss=float(max_daily_loss),
        max_concurrent_trades=int(max_concurrent_trades),
        min_risk_reward=float(min_rr_filter)
    )

    todays_loss = st.session_state.db.get_todays_realized_loss()
    open_positions = st.session_state.db.get_open_positions()

    # Process all historical candidates in the window through Risk Engine and persist to SQLite
    historical_decisions = {}
    for cand in candidates:
        cand_dec = risk_engine.evaluate(cand, current_daily_loss=todays_loss, open_positions=open_positions)
        st.session_state.db.save_signal(cand_dec)
        ts_k = cand.candle_timestamp.isoformat() if cand.candle_timestamp else ""
        historical_decisions[ts_k] = cand_dec

    latest_candidate = candidates[-1] if candidates else None
    decision = risk_engine.evaluate(latest_candidate, current_daily_loss=todays_loss, open_positions=open_positions)

    # 🤖 AI Decision Authority (Gemini Evaluates & Decides)
    ai_decision_data = None
    if latest_candidate and st.session_state.gemini.is_configured() and ("Gemini AI Autonomous" in decision_mode or "Hybrid" in decision_mode):
        cand_sig = f"{latest_candidate.symbol}_{latest_candidate.direction}_{latest_candidate.candle_timestamp.isoformat()}"
        if f"ai_dec_{cand_sig}" in st.session_state:
            ai_decision_data = st.session_state[f"ai_dec_{cand_sig}"]
        else:
            cand_dict = {
                "symbol": latest_candidate.symbol,
                "direction": latest_candidate.direction,
                "regime": latest_candidate.regime,
                "entry_price": latest_candidate.entry_price,
                "stop_loss": latest_candidate.stop_loss,
                "tp1": latest_candidate.tp1,
                "tp2": latest_candidate.tp2,
                "risk_reward": latest_candidate.risk_reward_ratio,
                "setup_reason": latest_candidate.setup_reason,
                "adx": last_candle.get("adx"),
                "atr": last_candle.get("atr"),
                "support": last_candle.get("support"),
                "resistance": last_candle.get("resistance")
            }
            ai_decision_data = st.session_state.gemini.decide_trade_approval(cand_dict)
            st.session_state[f"ai_dec_{cand_sig}"] = ai_decision_data

        if "Gemini AI Autonomous" in decision_mode:
            if ai_decision_data.get("decision") == "APPROVE":
                decision.status = "APPROVE"
                decision.setup_reason = f"🤖 AI Approved ({ai_decision_data.get('confidence', 'HIGH')}): {ai_decision_data.get('verdict_summary', '')}"
                decision.rejection_reason = ""
            elif ai_decision_data.get("decision") == "REJECT":
                decision.status = "REJECT"
                decision.rejection_reason = f"🤖 AI Rejection: {ai_decision_data.get('reason', '')}"
        elif "Hybrid" in decision_mode:
            if decision.status == "APPROVE" and ai_decision_data.get("decision") != "APPROVE":
                decision.status = "REJECT"
                decision.rejection_reason = f"Risk engine passed but AI Vetoed: {ai_decision_data.get('reason', '')}"

    # Save latest setup and dispatch alerts to Telegram
    if latest_candidate:
        st.session_state.db.save_signal(decision)
        ai_r = ai_decision_data.get("reason") if ai_decision_data else None

        # Auto-create paper trade if approved
        if auto_paper_trade and decision.status == "APPROVE":
            cand_time_str = decision.candle_timestamp.isoformat() if decision.candle_timestamp else "NOW"
            existing_t = st.session_state.db.get_all_paper_trades(limit=50)
            already_created = any(
                t["symbol"] == decision.symbol and t["signal_time"] == cand_time_str 
                for t in existing_t
            )
            if not already_created:
                sim_auto = PaperTradingSimulator(enable_trailing_stop=enable_trailing_stop)
                new_t = sim_auto.create_trade_from_decision(decision)
                if new_t:
                    st.session_state.db.save_or_update_trade(new_t)

        # Dispatch Telegram alert if candidate is recent (within last 4 candles / 60 minutes) to avoid spamming historical signals
        is_recent_candidate = False
        if decision.candle_timestamp:
            try:
                time_diff = abs((last_utc - decision.candle_timestamp).total_seconds())
                is_recent_candidate = (time_diff <= 3600)
            except Exception:
                is_recent_candidate = True

        if is_recent_candidate:
            if enable_telegram and decision.status == "APPROVE":
                st.session_state.telegram.send_trade_signal_alert(decision, ai_reason=ai_r, account_balance=curr_balance)
            elif send_telegram_rejections and decision.status == "REJECT":
                st.session_state.telegram.send_trade_rejection_alert(decision, ai_reason=ai_r)

    # ----------------- STATUS & METRICS ROW -----------------
    m1, m2, m3, m4, m5 = st.columns([1.5, 1.2, 1.2, 1.2, 1.5])
    m1.metric("Latest Close", f"${last_candle['close']:,.2f}")
    m2.metric("ADX (14)", f"{last_candle['adx']:.1f}" if pd.notna(last_candle['adx']) else "—")
    m3.metric("+DI / -DI", f"{last_candle['plus_di']:.1f} / {last_candle['minus_di']:.1f}" if pd.notna(last_candle['plus_di']) else "—")
    m4.metric("ATR (14)", f"${last_candle['atr']:.2f}" if pd.notna(last_candle['atr']) else "—")
    
    regime = str(last_candle["regime"])
    badge_class = "badge-trending" if regime == "TRENDING" else ("badge-sideways" if regime == "SIDEWAYS" else "badge-transition")
    m5.markdown(f"**Market Regime**<br><span class='{badge_class}'>{regime}</span>", unsafe_allow_html=True)

    # Quick AI Market Pulse trigger
    pulse_col1, pulse_col2 = st.columns([1, 4])
    with pulse_col1:
        if st.button(f"🧠 AI Market Pulse", key=f"pulse_btn_{symbol_key}"):
            if not st.session_state.gemini.is_configured():
                st.warning("⚠️ Enter Gemini API Key in the sidebar.")
            else:
                with st.spinner("🤖 Gemini analyzing market momentum..."):
                    pulse_text = st.session_state.gemini.analyze_market_overview(symbol_key, {
                        "close": last_candle["close"],
                        "regime": last_candle["regime"],
                        "adx": last_candle["adx"],
                        "plus_di": last_candle["plus_di"],
                        "minus_di": last_candle["minus_di"],
                        "atr": last_candle["atr"],
                        "resistance": last_candle["resistance"],
                        "support": last_candle["support"]
                    })
                    st.session_state[f"pulse_{symbol_key}"] = pulse_text
    with pulse_col2:
        if f"pulse_{symbol_key}" in st.session_state:
            st.info(st.session_state[f"pulse_{symbol_key}"])

    # ----------------- LATEST RISK DECISION BANNER -----------------
    st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)
    
    d_col1, d_col2 = st.columns([2, 3])
    with d_col1:
        st.subheader("Latest Setup Decision")
        if decision.status == "APPROVE":
            st.markdown(f"<span class='badge-approve'>✅ APPROVE: {decision.direction}</span>", unsafe_allow_html=True)
            st.write(f"**Setup Reason:** {decision.setup_reason}")
        elif decision.status == "REJECT":
            st.markdown(f"<span class='badge-reject'>❌ REJECTED: {decision.direction}</span>", unsafe_allow_html=True)
            st.error(f"**Rejection Reason:** {decision.rejection_reason}")
            st.caption(f"Setup Identified: {decision.setup_reason}")
        else:
            st.markdown("<span class='badge-wait'>⏳ WAIT: No Active Setup</span>", unsafe_allow_html=True)
            st.info("Market is currently scanning for a closed-candle confirmation pattern meeting deterministic criteria.")

        # Gemini Review Button on Latest Decision
        if decision.status in ["APPROVE", "REJECT"]:
            if st.button(f"🤖 Audit Decision with Gemini AI", key=f"gemini_decision_btn_{symbol_key}"):
                if not st.session_state.gemini.is_configured():
                    st.warning("⚠️ Enter Gemini API Key in the sidebar.")
                else:
                    with st.spinner("🤖 Gemini is reviewing the setup..."):
                        t_data = decision.to_dict()
                        rev_out = st.session_state.gemini.analyze_trade_decision(t_data)
                        st.session_state[f"latest_rev_{symbol_key}"] = rev_out

        if f"latest_rev_{symbol_key}" in st.session_state:
            with st.expander("🧠 Gemini AI Decision Analysis", expanded=True):
                st.markdown(st.session_state[f"latest_rev_{symbol_key}"])

    with d_col2:
        if decision.status in ["APPROVE", "REJECT"]:
            c_a, c_b, c_c, c_d = st.columns(4)
            c_a.metric("Entry Price", f"${decision.entry_price:,.2f}")
            c_b.metric("Stop Loss", f"${decision.stop_loss:,.2f}")
            c_c.metric("Target (TP1)", f"${decision.tp1:,.2f}")
            c_d.metric("Net R:R", f"{decision.net_risk_reward:.2f}R")

            r_a, r_b, r_c = st.columns(3)
            r_a.metric("Position Size", f"{decision.position_size:.4f} {decision.position_unit}")
            r_b.metric("Risk Budget", f"${decision.risk_amount_usd:,.2f}")
            r_c.metric("Est. Friction", f"${decision.estimated_cost_usd:,.2f}")

            # Paper Trade Execution Action
            if decision.status == "APPROVE":
                cand_time_str = decision.candle_timestamp.isoformat() if decision.candle_timestamp else "NOW"
                existing_t = st.session_state.db.get_all_paper_trades(limit=50)
                active_match = next((t for t in existing_t if t["symbol"] == decision.symbol and t["signal_time"] == cand_time_str), None)
                
                if active_match:
                    st.success(f"⚡ Trade Active: `{active_match['trade_id']}` [{active_match['status']}] • Tracking Target & SL")
                else:
                    if st.button(f"🚀 Paper-Trade This Setup ({decision.direction} {symbol_key})"):
                        sim = PaperTradingSimulator(enable_trailing_stop=enable_trailing_stop)
                        trade = sim.create_trade_from_decision(decision)
                        if trade:
                            st.session_state.db.save_or_update_trade(trade)
                            st.success(f"Paper trade {trade.trade_id} logged as PENDING! Will simulate fill on next candle.")
                            st.rerun()

    # ----------------- INTERACTIVE PLOTLY CHART -----------------
    st.subheader(f"📊 {spec.display_name} Multi-Panel Technical Chart")
    
    # Convert timestamps to display timezone for the x-axis
    disp_df = convert_to_display_timezone(annotated_df, tz_name=DISPLAY_TIMEZONE)
    x_axis = disp_df["display_datetime"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=["Price Action, S/R Levels & Confirmed Signals", "ADX & Directional Movement", "Average True Range (ATR)", "Volume & Volume MA"]
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=x_axis,
            open=disp_df["open"],
            high=disp_df["high"],
            low=disp_df["low"],
            close=disp_df["close"],
            name="Candles",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # Support and Resistance lines
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=disp_df["resistance"],
            mode="lines",
            name="Confirmed Resistance",
            line=dict(color="#f39c12", width=1.5, dash="dash")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=disp_df["support"],
            mode="lines",
            name="Confirmed Support",
            line=dict(color="#3498db", width=1.5, dash="dash")
        ),
        row=1, col=1
    )

    # Signal Markers
    buys = disp_df[disp_df["signal"] == "BUY"]
    sells = disp_df[disp_df["signal"] == "SELL"]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["display_datetime"],
                y=buys["low"] * 0.999,
                mode="markers",
                name="BUY Signal",
                marker=dict(symbol="triangle-up", size=13, color="#00e676", line=dict(width=1, color="#ffffff")),
                text=[f"BUY @ {p:.2f}<br>SL: {sl:.2f}<br>TP: {tp:.2f}<br>{r}" for p, sl, tp, r in zip(buys["entry_price"], buys["stop_loss"], buys["tp1"], buys["signal_reason"])],
                hoverinfo="text+x"
            ),
            row=1, col=1
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["display_datetime"],
                y=sells["high"] * 1.001,
                mode="markers",
                name="SELL Signal",
                marker=dict(symbol="triangle-down", size=13, color="#ff1744", line=dict(width=1, color="#ffffff")),
                text=[f"SELL @ {p:.2f}<br>SL: {sl:.2f}<br>TP: {tp:.2f}<br>{r}" for p, sl, tp, r in zip(sells["entry_price"], sells["stop_loss"], sells["tp1"], sells["signal_reason"])],
                hoverinfo="text+x"
            ),
            row=1, col=1
        )

    # 2. ADX, +DI, -DI
    fig.add_trace(go.Scatter(x=x_axis, y=disp_df["adx"], name="ADX", line=dict(color="#f1c40f", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=disp_df["plus_di"], name="+DI", line=dict(color="#2ecc71", width=1.2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=disp_df["minus_di"], name="-DI", line=dict(color="#e74c3c", width=1.2, dash="dot")), row=2, col=1)
    fig.add_hline(y=adx_sideways, line_dash="dot", line_color="#7f8c8d", annotation_text=f"Sideways ({adx_sideways})", row=2, col=1)
    fig.add_hline(y=adx_trending, line_dash="dot", line_color="#95a5a6", annotation_text=f"Trending ({adx_trending})", row=2, col=1)

    # 3. ATR
    fig.add_trace(go.Scatter(x=x_axis, y=disp_df["atr"], name="ATR", line=dict(color="#9b59b6", width=1.5)), row=3, col=1)

    # 4. Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(disp_df["close"], disp_df["open"])]
    fig.add_trace(go.Bar(x=x_axis, y=disp_df["volume"], name="Volume", marker_color=colors, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=disp_df["vol_ma"], name="Vol MA(20)", line=dict(color="#3498db", width=1.2)), row=4, col=1)

    fig.update_layout(
        height=850,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    render_plotly_chart(fig)

    # ----------------- SIGNAL HISTORY TABLE -----------------
    st.subheader(f"📜 Detected Signals ({symbol_key})")
    signals_subset = disp_df[disp_df["signal"] != ""].copy().sort_values("datetime", ascending=False)
    if signals_subset.empty:
        st.info("No confirmed signals detected in the current candle window.")
    else:
        statuses = []
        rejections = []
        for dt_val in signals_subset["datetime"]:
            utc_ts = pd.to_datetime(dt_val).to_pydatetime()
            ts_k = utc_ts.isoformat()
            if ts_k in historical_decisions:
                d_obj = historical_decisions[ts_k]
                statuses.append("🟢 APPROVE" if d_obj.status == "APPROVE" else "🔴 REJECT")
                rejections.append(d_obj.rejection_reason or "—")
            else:
                statuses.append("—")
                rejections.append("—")

        signals_subset["status_badge"] = statuses
        signals_subset["rejection_reason"] = rejections

        disp_cols = [
            "display_datetime", "signal", "status_badge", "rejection_reason",
            "entry_price", "stop_loss", "tp1", "risk_reward", "regime", "signal_reason"
        ]
        renamed = signals_subset[disp_cols].rename(columns={
            "display_datetime": f"Time ({DISPLAY_TIMEZONE})",
            "signal": "Direction",
            "status_badge": "Risk Decision",
            "rejection_reason": "Rejection Explanation",
            "entry_price": "Entry",
            "stop_loss": "Stop Loss",
            "tp1": "Target",
            "risk_reward": "R:R",
            "regime": "Regime",
            "signal_reason": "Setup Reason"
        })
        render_dataframe_table(renamed, hide_index=True)

    # Attribution Footer
    st.caption(f"Data Feed: {feed_source} • Closed Candles: {len(candles_df):,} • Last Candle Close: {last_ist_str}")

# Render Tab 1 & 2
with tab_gold:
    render_instrument_tab("XAUUSD")

with tab_btc:
    render_instrument_tab("BTCUSDT")

# ----------------- TAB: TRADE & DECISION HISTORY -----------------
with tab_history:
    st.header("📜 Trade Signal & Approval History")
    st.caption("Comprehensive audit trail of all detected signals, Risk Engine decisions (APPROVED vs REJECTED), rejection reasons, and Gemini AI reviews.")

    h_col1, h_col2, h_col3 = st.columns([1.5, 1.5, 1])
    filter_sym = h_col1.selectbox("Filter Asset", ["ALL", "XAUUSD", "BTCUSDT"], key="hist_filter_sym")
    filter_status = h_col2.selectbox("Filter Decision", ["ALL", "APPROVE", "REJECT"], key="hist_filter_stat")
    limit_count = h_col3.selectbox("Max Records", [50, 100, 200, 500], index=1, key="hist_filter_limit")

    all_signals = st.session_state.db.get_all_signals(symbol=filter_sym, status=filter_status, limit=limit_count)

    if not all_signals:
        st.info("No trade signals found in database matching current filters. Navigate to Gold or Bitcoin tabs to scan candles and detect signals.")
    else:
        df_sig = pd.DataFrame(all_signals)
        
        # Summary metrics
        total_sigs = len(df_sig)
        approved_count = len(df_sig[df_sig["status"] == "APPROVE"])
        rejected_count = len(df_sig[df_sig["status"] == "REJECT"])
        app_rate = (approved_count / total_sigs * 100) if total_sigs > 0 else 0.0

        sh1, sh2, sh3, sh4 = st.columns(4)
        sh1.metric("Total Setups Detected", f"{total_sigs}")
        sh2.metric("Approved Setups", f"{approved_count}", "🟢 Ready to trade")
        sh3.metric("Rejected Setups", f"{rejected_count}", "🔴 Capital protected")
        sh4.metric("Approval Rate", f"{app_rate:.1f}%")

        # Convert timestamps for presentation
        kolkata_tz = pytz.timezone(DISPLAY_TIMEZONE)
        display_times = []
        for ts_val in df_sig["candle_timestamp"]:
            try:
                dt = pd.to_datetime(ts_val)
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt)
                display_times.append(dt.astimezone(kolkata_tz).strftime("%Y-%m-%d %H:%M"))
            except Exception:
                display_times.append(str(ts_val))
        df_sig["display_time"] = display_times

        df_sig["status_icon"] = df_sig["status"].apply(
            lambda s: "🟢 APPROVED" if s == "APPROVE" else ("🔴 REJECTED" if s == "REJECT" else "⏳ WAIT")
        )

        display_cols = [
            "id", "display_time", "symbol", "direction", "status_icon",
            "entry_price", "stop_loss", "tp1", "risk_reward",
            "rejection_reason", "setup_reason", "regime"
        ]

        st.subheader("📋 Signal & Risk Decision Log")
        renamed_df = df_sig[display_cols].rename(columns={
            "id": "ID",
            "display_time": f"Time ({DISPLAY_TIMEZONE})",
            "symbol": "Asset",
            "direction": "Side",
            "status_icon": "Decision",
            "entry_price": "Entry ($)",
            "stop_loss": "SL ($)",
            "tp1": "TP1 ($)",
            "risk_reward": "R:R",
            "rejection_reason": "Rejection Explanation",
            "setup_reason": "Setup Details",
            "regime": "Regime"
        })
        render_dataframe_table(renamed_df, hide_index=True)

        # ----------------- GEMINI AI AUDIT SECTION -----------------
        st.divider()
        st.subheader("🤖 Gemini AI Trade Audit & Advisor")
        st.caption("Select any logged trade above to have Google Gemini evaluate market structure, risk logic, and execution recommendations.")

        trade_lookup = {}
        for _, row in df_sig.iterrows():
            lbl = f"ID #{row['id']} | {row['symbol']} {row['direction']} @ ${row['entry_price']:,.2f} on {row['display_time']} [{row['status']}]"
            trade_lookup[lbl] = row.to_dict()

        sel_trade_label = st.selectbox("Select Trade to Audit", list(trade_lookup.keys()), key="gemini_trade_sel")
        selected_trade_dict = trade_lookup[sel_trade_label]

        if st.button("🚀 Analyze Selected Trade with Gemini AI", key="run_gemini_trade_audit"):
            if not st.session_state.gemini.is_configured():
                st.warning("⚠️ Please provide a Gemini API Key in the left sidebar to generate AI analysis.")
            else:
                with st.spinner("🤖 Gemini is auditing market structure, risk reward, and trade execution..."):
                    ai_review = st.session_state.gemini.analyze_trade_decision(selected_trade_dict)
                    st.session_state["gemini_audit_result"] = ai_review

        if "gemini_audit_result" in st.session_state:
            st.markdown("### 🧠 Gemini AI Review")
            st.markdown(st.session_state["gemini_audit_result"])

# ----------------- TAB: PAPER TRADING JOURNAL -----------------
with tab_journal:
    st.header("📖 Paper Trading Journal & Position Monitor")
    st.caption("All paper trades, targets, stop-losses, and simulated executions persisted in local SQLite database.")

    trades = st.session_state.db.get_all_paper_trades(limit=200)
    if not trades:
        st.info("No paper trades logged yet. Click 'Paper-Trade This Setup' on any approved signal or enable auto-execution to begin simulating.")
    else:
        df_trades = pd.DataFrame(trades)
        
        # Summary metrics
        total_pnl = float(df_trades["pnl_net"].sum())
        closed_trades = df_trades[df_trades["status"].str.startswith("CLOSED")]
        open_trades = df_trades[df_trades["status"] == "OPEN"]
        pending_trades = df_trades[df_trades["status"] == "PENDING"]
        
        tp_wins = closed_trades[closed_trades["status"].isin(["CLOSED_TP1", "CLOSED_TP2"])]
        sl_losses = closed_trades[closed_trades["status"].isin(["CLOSED_SL", "CLOSED_TRAILING"])]
        
        closed_count = len(closed_trades)
        win_count = len(tp_wins)
        wr = (win_count / closed_count * 100) if closed_count > 0 else 0.0
        journal_balance = float(account_balance) + total_pnl

        j1, j2, j3, j4, j5 = st.columns(5)
        j1.metric("💰 Current Balance", f"${journal_balance:,.2f}", delta=f"{total_pnl:+,.2f} Net P&L")
        j2.metric("🎯 Target Hits (Wins)", f"{win_count}")
        j3.metric("🛑 Stop-Loss Hits", f"{len(sl_losses)}")
        j4.metric("⚡ Open / Pending", f"{len(open_trades)} / {len(pending_trades)}")
        j5.metric("🏆 Win Rate", f"{wr:.1f}%")

        # Map status to clear visual labels
        def format_status(s):
            if s in ["CLOSED_TP1", "CLOSED_TP2"]:
                return "🎯 TARGET HIT"
            elif s == "CLOSED_SL":
                return "🛑 STOP-LOSS HIT"
            elif s == "CLOSED_TRAILING":
                return "🛡️ TRAILING STOP HIT"
            elif s == "OPEN":
                return "⚡ OPEN POSITION"
            elif s == "PENDING":
                return "⏳ PENDING FILL"
            return str(s)

        df_display = df_trades.copy()
        df_display["display_status"] = df_display["status"].apply(format_status)
        
        # Format PnL with explicit plus sign
        df_display["formatted_pnl"] = df_display["pnl_net"].apply(lambda v: f"{v:+,.2f}" if pd.notna(v) else "$0.00")
        
        # Reorder and rename columns for display
        cols_to_show = [
            "trade_id", "symbol", "direction", "display_status",
            "executed_entry_price", "exit_price", "tp1", "stop_loss",
            "formatted_pnl", "r_multiple", "exit_reason", "signal_time"
        ]
        available_cols = [c for c in cols_to_show if c in df_display.columns]
        renamed_journal = df_display[available_cols].rename(columns={
            "trade_id": "Trade ID",
            "symbol": "Asset",
            "direction": "Side",
            "display_status": "Execution Status",
            "executed_entry_price": "Entry ($)",
            "exit_price": "Exit ($)",
            "tp1": "Target (TP1)",
            "stop_loss": "Stop-Loss (SL)",
            "formatted_pnl": "Net P&L ($)",
            "r_multiple": "R:R Multiple",
            "exit_reason": "Exit Reason",
            "signal_time": "Signal Time"
        })

        st.subheader("Trade Execution & P&L Records")
        render_dataframe_table(renamed_journal, hide_index=True)

# ----------------- TAB 4: BACKTEST & PERFORMANCE -----------------
with tab_backtest:
    st.header("📊 Strategy Backtest Engine")
    st.caption("Simulate bar-by-bar execution across loaded historical data using exact non-repainting logic.")

    b_col1, b_col2, b_col3 = st.columns(3)
    bt_symbol = b_col1.selectbox("Backtest Instrument", ["XAUUSD", "BTCUSDT"])
    bt_capital = b_col2.number_input("Starting Capital ($)", value=10000.0, step=1000.0)
    bt_risk = b_col3.slider("Backtest Risk / Trade (%)", 0.5, 3.0, 1.0, 0.5)

    if st.button("🚀 Run Historical Backtest"):
        with st.spinner("Simulating closed-candle strategy..."):
            sample_file = "xauusd_15m_sample.csv" if bt_symbol == "XAUUSD" else "btcusdt_15m_sample.csv"
            sample_p = os.path.join(BASE_DIR, "data", "samples", sample_file)
            hist_df = normalize_ohlcv_dataframe(pd.read_csv(sample_p))

            summary, trades_df, closed_trades = run_backtest(
                df=hist_df,
                symbol=bt_symbol,
                initial_balance=bt_capital,
                risk_pct=bt_risk,
                max_daily_loss=max_daily_loss,
                max_concurrent_trades=int(max_concurrent_trades),
                min_risk_reward=min_rr_filter,
                adx_sideways=float(adx_sideways),
                adx_trending=float(adx_trending),
                sl_atr_mult=float(sl_atr_mult),
                target_rr=float(target_rr),
                use_volume=use_volume,
                use_retest=use_retest
            )

            st.success(f"Backtest completed over {len(hist_df)} candles!")

            # Metric Cards
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total Trades", f"{summary.total_trades}")
            bc2.metric("Win Rate", f"{summary.win_rate_pct:.1f}%", f"{summary.winning_trades}W / {summary.losing_trades}L")
            bc3.metric("Profit Factor", f"{summary.profit_factor:.2f}")
            bc4.metric("Average R", f"{summary.average_r_multiple:.2f}R")

            bc5, bc6, bc7, bc8 = st.columns(4)
            bc5.metric("Net P&L", f"${summary.total_net_pnl:,.2f}")
            bc6.metric("Total Fees & Friction", f"${summary.total_fees_paid:,.2f}")
            bc7.metric("Max Drawdown", f"${summary.max_drawdown_usd:,.2f} ({summary.max_drawdown_pct:.1f}%)")
            bc8.metric("Return on Capital", f"{summary.return_on_capital_pct:.1f}%")

            # Equity Curve
            if not trades_df.empty:
                st.subheader("Equity Curve")
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(
                    x=trades_df.index,
                    y=trades_df["equity"],
                    mode="lines+markers",
                    name="Account Equity",
                    line=dict(color="#00b894", width=2)
                ))
                eq_fig.update_layout(
                    template="plotly_dark",
                    title="Account Equity Progression ($)",
                    xaxis_title="Trade Number",
                    yaxis_title="Equity (USD)",
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                render_plotly_chart(eq_fig)

                st.subheader("Simulated Trade Log")
                render_dataframe_table(trades_df, hide_index=True)
            else:
                st.info("No trades triggered under the selected parameters.")

# ----------------- FOOTER DISCLAIMER -----------------
st.divider()
st.caption(
    "⚠️ **Disclaimer & Attribution:** This application is strictly an educational research and paper-trading analytics dashboard. "
    "No real orders are placed. Trading commodities and cryptocurrencies involves significant capital risk; historical results do not guarantee future returns. "
    "Gold data is polled via Yahoo Finance (unofficial, delayed public feed) or verified 1:1 PAXG spot gold proxy. "
    "Bitcoin data is fetched via public Binance Spot market data endpoints."
)
