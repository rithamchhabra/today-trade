"""
24/7 Background Market Scanner for Gold (XAUUSD) and Bitcoin (BTCUSDT).
Runs independently of browser sessions, continuously fetching closed 15m candles,
evaluating setups through the Risk and AI engines, and dispatching instant Telegram alerts.
"""
import os
import sys
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import pytz
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Ensure package is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from hybrid_adaptive_trading_indicator.config import (
    DEFAULT_ACCOUNT_BALANCE, DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_MAX_DAILY_LOSS, DEFAULT_MAX_CONCURRENT_TRADES, DEFAULT_MIN_RISK_REWARD,
    DEFAULT_ADX_LENGTH, DEFAULT_ATR_LENGTH, DEFAULT_ADX_SIDEWAYS, DEFAULT_ADX_TRENDING,
    DEFAULT_PIVOT_LEFT, DEFAULT_PIVOT_RIGHT, DEFAULT_ZONE_ATR_MULT, DEFAULT_SL_ATR_MULT,
    DEFAULT_TP1_RR, DEFAULT_VOLUME_MA_LEN, DISPLAY_TIMEZONE
)
from hybrid_adaptive_trading_indicator.data.integrity import validate_and_filter_candles
from hybrid_adaptive_trading_indicator.data.binance_client import BinanceDataClient
from hybrid_adaptive_trading_indicator.data.yfinance_client import YahooFinanceDataClient
from hybrid_adaptive_trading_indicator.data.gold_api_fallback import GoldFallbackClient
from hybrid_adaptive_trading_indicator.strategy.engine import prepare_indicators, evaluate_all_candidates
from hybrid_adaptive_trading_indicator.risk.approval_engine import RiskApprovalEngine
from hybrid_adaptive_trading_indicator.execution.paper_trader import PaperTradingSimulator
from hybrid_adaptive_trading_indicator.storage.database import DatabaseManager
from hybrid_adaptive_trading_indicator.alerts.telegram import TelegramAlertDispatcher
from hybrid_adaptive_trading_indicator.ai.gemini_advisor import GeminiAdvisor

logger = logging.getLogger("BackgroundScanner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class BackgroundMarketScanner:
    _instance: Optional["BackgroundMarketScanner"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        scan_interval_seconds: int = 60,
        symbols: Optional[List[str]] = None,
        account_balance: float = DEFAULT_ACCOUNT_BALANCE,
        risk_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        max_concurrent_trades: int = DEFAULT_MAX_CONCURRENT_TRADES,
        min_rr: float = DEFAULT_MIN_RISK_REWARD,
        auto_paper_trade: bool = True,
        enable_telegram_signals: bool = True,
        enable_telegram_closed: bool = True,
        enable_telegram_rejections: bool = False,
        decision_mode: str = "Gemini AI Autonomous"
    ):
        load_dotenv()
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", str(scan_interval_seconds)))
        self.symbols = symbols or ["BTCUSDT", "XAUUSD"]
        self.account_balance = float(account_balance)
        self.risk_pct = float(risk_pct)
        self.max_daily_loss = float(max_daily_loss)
        self.max_concurrent_trades = int(max_concurrent_trades)
        self.min_rr = float(min_rr)
        self.auto_paper_trade = auto_paper_trade
        self.enable_telegram_signals = enable_telegram_signals
        self.enable_telegram_closed = enable_telegram_closed
        self.enable_telegram_rejections = enable_telegram_rejections or (os.getenv("ALERT_ON_REJECTED", "false").lower() == "true")
        self.decision_mode = decision_mode

        self.db = DatabaseManager()
        self.telegram = TelegramAlertDispatcher()
        self.gemini = GeminiAdvisor()

        # Wire DB to telegram dispatcher
        TelegramAlertDispatcher._db_manager = self.db
        TelegramAlertDispatcher._starting_balance_fn = lambda: self.account_balance

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.last_scan_time: Optional[datetime] = None
        self.last_scan_status: Dict[str, Any] = {}
        self.scan_count: int = 0

    @classmethod
    def get_instance(cls) -> "BackgroundMarketScanner":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self, send_startup_alert: bool = True):
        """Starts the background 24/7 scanning thread and interactive telegram listener."""
        with self._lock:
            if self.is_running and self._thread and self._thread.is_alive():
                logger.info("Background scanner already running.")
                return

            self.is_running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="24_7_MarketScannerThread"
            )
            self._thread.start()

            # Start interactive Telegram listener for p/l, balance, status, AI chat, etc.
            if self.telegram.is_configured:
                self.telegram.start_listener(self.db, lambda: self.account_balance, gemini_advisor=self.gemini)

            if send_startup_alert and self.telegram.is_configured:
                try:
                    startup_msg = (
                        "🚀 *24/7 Background Market Scanner Online!* 🚀\n\n"
                        "Scanning *Gold (XAUUSD)* & *Bitcoin (BTCUSDT)* every 60s.\n"
                        "• Instant live signals on setup detection\n"
                        "• Real-time TP / SL exit alerts with P&L\n\n"
                        "💬 Send `/status` or `/help` in Telegram anytime!"
                    )
                    self.telegram._send_telegram_raw(startup_msg)
                except Exception as e:
                    logger.warning(f"Failed to send Telegram startup banner: {e}")

            logger.info("24/7 Background Market Scanner started successfully.")

    def stop(self):
        """Stops the scanner loop."""
        with self._lock:
            self.is_running = False
            logger.info("24/7 Background Market Scanner stopped.")

    def _run_loop(self):
        """Continuous background execution loop."""
        logger.info(f"Scanner loop active. Polling interval: {self.scan_interval}s.")
        while self.is_running:
            try:
                self.scan_once()
            except Exception as e:
                logger.error(f"Error in scanner loop: {e}", exc_info=True)

            # Sleep in 1-second chunks to allow graceful termination
            for _ in range(max(1, self.scan_interval)):
                if not self.is_running:
                    break
                time.sleep(1)

    def scan_once(self) -> Dict[str, Any]:
        """Performs a single complete scan across all configured symbols."""
        kolkata_tz = pytz.timezone(DISPLAY_TIMEZONE)
        scan_time_ist = datetime.now(kolkata_tz).strftime("%Y-%m-%d %H:%M:%S IST")
        results = {"time": scan_time_ist, "symbols": {}}

        todays_loss = self.db.get_todays_realized_loss()
        open_positions = self.db.get_open_positions()

        # Recalculate dynamic balance
        all_t = self.db.get_all_paper_trades(limit=500)
        tot_pnl = sum(t.get("pnl_net", 0.0) for t in all_t if str(t.get("status", "")).startswith("CLOSED"))
        curr_balance = self.account_balance + tot_pnl

        risk_engine = RiskApprovalEngine(
            account_balance=curr_balance,
            risk_per_trade_pct=self.risk_pct,
            max_daily_loss=self.max_daily_loss,
            max_concurrent_trades=self.max_concurrent_trades,
            min_risk_reward=self.min_rr
        )

        for symbol in self.symbols:
            try:
                sym_res = self._process_symbol(symbol, risk_engine, curr_balance, todays_loss, open_positions)
                results["symbols"][symbol] = sym_res
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)
                results["symbols"][symbol] = {"error": str(e)}

        self.last_scan_time = datetime.now(timezone.utc)
        self.last_scan_status = results
        self.scan_count += 1
        return results

    def _fetch_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches latest 15m candles with automatic fallback support."""
        if symbol == "BTCUSDT":
            client = BinanceDataClient()
            df = client.fetch_klines(symbol="BTCUSDT", interval="15m", limit=300)
            clean_df, report = validate_and_filter_candles(df)
            return clean_df

        elif symbol == "XAUUSD":
            try:
                client = YahooFinanceDataClient()
                df = client.fetch_gold_candles(period="15d")
                clean_df, report = validate_and_filter_candles(df)
                # If stale (e.g. weekend market closure), fallback to 24/7 PAXG spot proxy
                if report.is_stale and report.stale_duration_minutes > 120:
                    fallback_client = GoldFallbackClient()
                    paxg_df, _ = fallback_client.fetch_paxg_gold_proxy_candles(limit=300)
                    clean_paxg, _ = validate_and_filter_candles(paxg_df)
                    return clean_paxg
                return clean_df
            except Exception:
                try:
                    fallback_client = GoldFallbackClient()
                    paxg_df, _ = fallback_client.fetch_paxg_gold_proxy_candles(limit=300)
                    clean_paxg, _ = validate_and_filter_candles(paxg_df)
                    return clean_paxg
                except Exception as fb_err:
                    logger.error(f"Failed to fetch Gold candles (Yahoo & PAXG fallback): {fb_err}")
                    return None
        return None

    def _process_symbol(
        self,
        symbol: str,
        risk_engine: RiskApprovalEngine,
        curr_balance: float,
        todays_loss: float,
        open_positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Runs the entire pipeline for one instrument: data -> paper trade check -> strategy -> alert."""
        candles_df = self._fetch_candles(symbol)
        if candles_df is None or len(candles_df) < 50:
            return {"status": "insufficient_data"}

        last_candle = candles_df.iloc[-1]
        last_utc = pd.to_datetime(last_candle["datetime"]).to_pydatetime()
        if last_utc.tzinfo is None:
            last_utc = pytz.utc.localize(last_utc)

        # 1. Check and update open paper trades against current price action
        raw_open = [t for t in self.db.get_open_positions(symbol)]
        trades_closed = 0
        if raw_open:
            sim = PaperTradingSimulator(enable_trailing_stop=True, trailing_atr_mult=1.5)
            updated_pairs = sim.update_trades_against_dataframe(raw_open, candles_df)
            for u_trade, was_closed in updated_pairs:
                self.db.save_or_update_trade(u_trade)
                if was_closed:
                    trades_closed += 1
                    all_t_now = self.db.get_all_paper_trades(limit=500)
                    tot_pnl_now = sum(t.get("pnl_net", 0.0) for t in all_t_now if str(t.get("status", "")).startswith("CLOSED"))
                    upd_bal = self.account_balance + tot_pnl_now

                    # Instant Telegram notification when TP or SL is hit
                    if self.enable_telegram_closed and self.telegram.is_configured:
                        self.telegram.send_trade_closed_alert(
                            u_trade,
                            starting_balance=self.account_balance,
                            current_balance=upd_bal
                        )
                        self.db.record_alert(
                            u_trade.symbol,
                            u_trade.direction,
                            f"Trade {u_trade.trade_id} closed: {u_trade.status} (PnL ${u_trade.pnl_net:+,.2f})",
                            True
                        )

        # 2. Compute Indicators & Candidate Setups
        ind_df = prepare_indicators(
            df=candles_df,
            adx_len=DEFAULT_ADX_LENGTH,
            atr_len=DEFAULT_ATR_LENGTH,
            pivot_left=DEFAULT_PIVOT_LEFT,
            pivot_right=DEFAULT_PIVOT_RIGHT,
            zone_atr_mult=DEFAULT_ZONE_ATR_MULT,
            volume_len=DEFAULT_VOLUME_MA_LEN,
            adx_sideways=DEFAULT_ADX_SIDEWAYS,
            adx_trending=DEFAULT_ADX_TRENDING,
            regime_hysteresis=1.0
        )

        annotated_df, candidates = evaluate_all_candidates(
            df=ind_df,
            symbol=symbol,
            timeframe="15m",
            sl_atr_mult=DEFAULT_SL_ATR_MULT,
            target_rr=DEFAULT_TP1_RR,
            tp1_rr=DEFAULT_TP1_RR,
            tp2_rr=DEFAULT_TP1_RR * 1.5,
            use_volume=False,
            use_retest=False
        )

        if not candidates:
            return {"status": "no_candidates", "last_close": float(last_candle["close"]), "trades_closed": trades_closed}

        latest_candidate = candidates[-1]
        cand_ts = latest_candidate.candle_timestamp
        if cand_ts and cand_ts.tzinfo is None:
            cand_ts = pytz.utc.localize(cand_ts)

        # Recency check: is candidate within the last 60 minutes?
        time_diff = abs((last_utc - cand_ts).total_seconds()) if cand_ts else 0
        is_recent = (time_diff <= 3600)

        ts_str = cand_ts.isoformat() if cand_ts else "NOW"
        setup_sig = f"SETUP_{symbol}_{latest_candidate.direction}_{ts_str}"

        # Evaluate through Risk Engine
        decision = risk_engine.evaluate(latest_candidate, current_daily_loss=todays_loss, open_positions=open_positions)

        # AI Decision Evaluation (if Gemini configured)
        ai_reason = None
        if self.gemini.is_configured() and ("AI" in self.decision_mode or "Hybrid" in self.decision_mode):
            try:
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
                ai_data = self.gemini.decide_trade_approval(cand_dict)
                ai_reason = ai_data.get("reason", "")
                if "Autonomous" in self.decision_mode:
                    if ai_data.get("decision") == "APPROVE":
                        decision.status = "APPROVE"
                        decision.setup_reason = f"🤖 AI Approved ({ai_data.get('confidence', 'HIGH')}): {ai_data.get('verdict_summary', '')}"
                        decision.rejection_reason = ""
                    elif ai_data.get("decision") == "REJECT":
                        decision.status = "REJECT"
                        decision.rejection_reason = f"🤖 AI Rejection: {ai_data.get('reason', '')}"
                elif "Hybrid" in self.decision_mode:
                    if decision.status == "APPROVE" and ai_data.get("decision") != "APPROVE":
                        decision.status = "REJECT"
                        decision.rejection_reason = f"Risk engine passed but AI Vetoed: {ai_data.get('reason', '')}"
            except Exception as ai_err:
                logger.warning(f"AI evaluation error: {ai_err}")

        # Persist signal to database
        self.db.save_signal(decision)

        # Dispatch Telegram Alert if recent and not yet sent
        alert_sent = False
        if is_recent and self.telegram.is_configured:
            if decision.status == "APPROVE" and self.enable_telegram_signals:
                approve_sig = f"APPROVE_{decision.symbol}_{decision.direction}_{ts_str}"
                if not self.db.is_alert_sent(approve_sig):
                    sent = self.telegram.send_trade_signal_alert(decision, ai_reason=ai_reason, account_balance=curr_balance)
                    if sent:
                        alert_sent = True
                        logger.info(f"🟢 [TELEGRAM SENT] Approved signal: {symbol} {decision.direction} @ {decision.entry_price}")

                # Auto-create paper trade if enabled
                if self.auto_paper_trade:
                    existing_t = self.db.get_all_paper_trades(limit=50)
                    already_created = any(
                        t["symbol"] == decision.symbol and t["signal_time"] == ts_str 
                        for t in existing_t
                    )
                    if not already_created:
                        sim_auto = PaperTradingSimulator(enable_trailing_stop=True)
                        new_t = sim_auto.create_trade_from_decision(decision)
                        if new_t:
                            self.db.save_or_update_trade(new_t)
                            logger.info(f"📝 Paper trade created: {new_t.trade_id} ({symbol} {new_t.direction})")

            elif decision.status == "REJECT" and self.enable_telegram_rejections:
                reject_sig = f"REJECT_{decision.symbol}_{decision.direction}_{ts_str}"
                if not self.db.is_alert_sent(reject_sig):
                    sent = self.telegram.send_trade_rejection_alert(decision, ai_reason=ai_reason)
                    if sent:
                        alert_sent = True
                        logger.info(f"🔴 [TELEGRAM SENT] Rejection alert: {symbol} {decision.direction}")

        return {
            "status": decision.status,
            "direction": decision.direction,
            "entry_price": decision.entry_price,
            "is_recent": is_recent,
            "alert_sent": alert_sent,
            "trades_closed": trades_closed,
            "last_close": float(last_candle["close"])
        }

def start_background_scanner(scan_interval_seconds: int = 60) -> BackgroundMarketScanner:
    """Convenience helper to get and start the global background scanner daemon."""
    scanner = BackgroundMarketScanner.get_instance()
    scanner.scan_interval = scan_interval_seconds
    if not scanner.is_running:
        scanner.start()
    return scanner
