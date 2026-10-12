"""
Telegram Alert Dispatcher.
Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment variables.
Never hardcodes tokens. Formats clear trading alerts for AI Approved and Rejected signals.
Includes automatic Chat ID discovery via Telegram Bot API getUpdates.
"""
import os
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Set
import requests
from hybrid_adaptive_trading_indicator.risk.models import RiskDecision
from hybrid_adaptive_trading_indicator.config import DISPLAY_TIMEZONE
import pytz
import threading
import time
from dotenv import load_dotenv

load_dotenv()

class TelegramAlertDispatcher:
    _listener_thread: Optional[threading.Thread] = None
    _listener_running: bool = False
    _db_manager = None
    _starting_balance_fn = None
    _last_update_id: int = 0
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        load_dotenv()
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.sent_signatures: Set[str] = set()

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def set_bot_token(self, token: str):
        self.bot_token = token.strip()

    def set_chat_id(self, chat_id: str):
        self.chat_id = str(chat_id).strip()

    def auto_discover_chat_id(self) -> Dict[str, Any]:
        """
        Polls Telegram getUpdates API to find any user who has interacted with the bot.
        Extracts chat_id, saves it to .env, and sends a welcome alert.
        """
        if not self.bot_token:
            return {"success": False, "message": "Bot token is missing. Please configure TELEGRAM_BOT_TOKEN."}

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if not data.get("ok"):
                return {"success": False, "message": f"Telegram API error: {data.get('description')}"}

            results = data.get("result", [])
            if not results:
                return {
                    "success": False,
                    "message": "No messages received by the bot yet. Please open your bot in Telegram (@Todaytrade_bot) and press START or send any message (e.g. 'hi'), then click this button again!"
                }

            # Find the latest chat from updates
            latest_chat = None
            for item in reversed(results):
                msg = item.get("message") or item.get("channel_post") or item.get("my_chat_member")
                if msg and "chat" in msg:
                    latest_chat = msg["chat"]
                    break

            if not latest_chat:
                return {"success": False, "message": "No valid chat found in recent updates."}

            found_chat_id = str(latest_chat["id"])
            username = latest_chat.get("username") or latest_chat.get("first_name", "Trader")

            self.set_chat_id(found_chat_id)

            # Persist to .env files
            self._save_chat_id_to_env(found_chat_id)

            # Send welcome message
            welcome_text = (
                f"🎉 *Telegram Trade Alerts Linked Successfully!*\n\n"
                f"Hello *{username}*! Your bot is now connected to the *Hybrid Adaptive Trading System*.\n"
                f"You will receive instant live notifications for:\n"
                f"• 🟢 *AI Approved Trade Signals* (Entry, SL, TP, Position Size, AI Thesis)\n"
                f"• 🔴 *AI / Risk Rejection Alerts* (Exact explanation of why a trap trade was skipped)\n\n"
                f"⚡ *Live alerts active!*"
            )
            self._send_telegram_raw(welcome_text)

            return {
                "success": True,
                "chat_id": found_chat_id,
                "user": username,
                "message": f"Successfully connected to @{username} (Chat ID: {found_chat_id})! Welcome alert sent to Telegram."
            }
        except Exception as e:
            return {"success": False, "message": f"Connection error checking updates: {e}"}

    def _save_chat_id_to_env(self, chat_id: str):
        """Helper to write discovered chat_id into .env files."""
        env_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        ]
        for p in env_paths:
            try:
                content = ""
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        content = f.read()

                if "TELEGRAM_CHAT_ID=" in content:
                    content = re.sub(r"TELEGRAM_CHAT_ID=.*", f"TELEGRAM_CHAT_ID={chat_id}", content)
                else:
                    content += f"\nTELEGRAM_CHAT_ID={chat_id}\n"

                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                pass

    def send_trade_signal_alert(
        self,
        decision: RiskDecision,
        ai_verdict: Optional[str] = None,
        ai_reason: Optional[str] = None,
        account_balance: Optional[float] = None
    ) -> bool:
        """
        Sends an alert for an APPROVED trade setup once.
        """
        if not self.is_configured:
            return False

        if decision.status != "APPROVE":
            return False

        # Build unique signature to prevent duplicate sending
        ts_str = decision.candle_timestamp.isoformat() if decision.candle_timestamp else "NOW"
        signature = f"APPROVE_{decision.symbol}_{decision.direction}_{ts_str}"
        if signature in self.sent_signatures:
            return False

        local_time_str = self._format_time(decision.candle_timestamp)
        emoji = "🟢" if decision.direction == "BUY" else "🔴"

        ai_section = ""
        if ai_reason:
            ai_section = (
                f"\n🤖 *AI Decision:* `APPROVED`\n"
                f"🧠 *AI Thesis:* _{ai_reason}_\n"
            )

        balance_line = f"💰 *Account Balance:* `${account_balance:,.2f}`\n" if account_balance is not None else ""

        message = (
            f"⚡ *AI-APPROVED TRADING SIGNAL* ⚡\n\n"
            f"{emoji} *{decision.direction} {decision.symbol}* ({decision.timeframe})\n"
            f"📊 *Market Regime:* `{decision.regime}`\n"
            f"🎯 *Setup Trigger:* {decision.setup_reason}\n"
            f"{ai_section}\n"
            f"📍 *Entry Price:* `{decision.entry_price:,.2f}`\n"
            f"🛑 *Stop-Loss:* `{decision.stop_loss:,.2f}`\n"
            f"🎯 *Take-Profit 1:* `{decision.tp1:,.2f}`\n"
            f"🎯 *Take-Profit 2:* `{decision.tp2:,.2f}`\n"
            f"⚖️ *Risk / Reward:* `{decision.risk_reward:.2f}` (Net: `{decision.net_risk_reward:.2f}`)\n\n"
            f"💼 *Position Size:* `{decision.position_size:.4f} {decision.position_unit}`\n"
            f"🛡️ *Monetary Risk:* `${decision.risk_amount_usd:,.2f}` ({decision.max_capital_at_risk_pct:.1f}%)\n"
            f"{balance_line}"
            f"🕒 *Candle Close (IST):* `{local_time_str}`\n\n"
            f"_🛡️ Verified by AI & Risk Engine. Happy Trading!_"
        )

        success = self._send_telegram_raw(message)
        if success:
            self.sent_signatures.add(signature)
        return success

    def send_trade_closed_alert(
        self,
        trade: Any,
        starting_balance: float = 10000.0,
        current_balance: float = 10000.0
    ) -> bool:
        """
        Sends an alert when a trade hits Take-Profit (Target) or Stop-Loss (SL),
        broadcasting exact Profit/Loss, R-Multiple, and updated Account Balance.
        """
        if not self.is_configured:
            return False

        signature = f"CLOSED_{trade.trade_id}_{trade.status}"
        if signature in self.sent_signatures:
            return False

        local_time_str = self._format_time(trade.exit_time or datetime.now(timezone.utc))

        if "TP" in trade.status:
            badge = "🎯 *TARGET REACHED (TAKE-PROFIT)!*"
            header = "🎉🟢"
            status_text = "Target Hit (Profit)"
        elif "SL" in trade.status:
            badge = "🛑 *STOP-LOSS HIT (RISK EXIT)!*"
            header = "⚠️🔴"
            status_text = "Stop-Loss Hit (Loss)"
        elif "TRAILING" in trade.status:
            badge = "🛡️ *TRAILING STOP HIT!*"
            header = "🟡"
            status_text = "Trailing Stop Hit"
        else:
            badge = "🔔 *TRADE POSITION CLOSED*"
            header = "ℹ️"
            status_text = trade.status

        pnl_val = trade.pnl_net
        pnl_prefix = "+" if pnl_val >= 0 else "-"
        pnl_str = f"{pnl_prefix}${abs(pnl_val):,.2f}"
        pnl_pct = (pnl_val / starting_balance) * 100 if starting_balance > 0 else 0.0
        pnl_pct_str = f"{pnl_prefix}{abs(pnl_pct):.2f}%"

        message = (
            f"{header} {badge} {header}\n\n"
            f"📌 *Asset:* `{trade.direction} {trade.symbol}`\n"
            f"🆔 *Trade ID:* `{trade.trade_id}`\n"
            f"🏁 *Result:* `{status_text}`\n"
            f"📝 *Exit Reason:* _{trade.exit_reason}_\n\n"
            f"📍 *Executed Entry:* `${trade.executed_entry_price:,.2f}`\n"
            f"🚪 *Exit Price:* `${trade.exit_price:,.2f}`\n"
            f"🎯 *Target (TP1):* `${trade.tp1:,.2f}`\n"
            f"🛑 *Stop-Loss (SL):* `${trade.stop_loss:,.2f}`\n\n"
            f"💵 *Realized P&L:* *{pnl_str}* ({pnl_pct_str})\n"
            f"⚖️ *R-Multiple:* `{trade.r_multiple:+.2f}R`\n"
            f"💸 *Trading Fees:* `${trade.fees_paid:,.2f}`\n\n"
            f"💰 *Updated Account Balance:* *${current_balance:,.2f}*\n"
            f"🏦 *Starting Capital:* `${starting_balance:,.2f}`\n"
            f"🕒 *Exit Time (IST):* `{local_time_str}`\n\n"
            f"_📈 Tracked live in Paper Trading Journal._"
        )

        success = self._send_telegram_raw(message)
        if success:
            self.sent_signatures.add(signature)
        return success

    def send_trade_rejection_alert(self, decision: RiskDecision, ai_reason: Optional[str] = None) -> bool:
        """
        Sends an alert when a potential trade is REJECTED by AI or Risk Engine.
        """
        if not self.is_configured:
            return False

        if decision.status != "REJECT":
            return False

        ts_str = decision.candle_timestamp.isoformat() if decision.candle_timestamp else "NOW"
        signature = f"REJECT_{decision.symbol}_{decision.direction}_{ts_str}"
        if signature in self.sent_signatures:
            return False

        local_time_str = self._format_time(decision.candle_timestamp)
        reason = ai_reason or decision.rejection_reason or "Risk safety criteria not met."

        message = (
            f"🛡️ *TRADE REJECTED BY AI / RISK ENGINE* 🛡️\n\n"
            f"❌ *{decision.direction} {decision.symbol}* ({decision.timeframe})\n"
            f"📊 *Market Regime:* `{decision.regime}`\n"
            f"⚠️ *Status:* `REJECTED`\n"
            f"🛑 *Rejection Reason:*\n_{reason}_\n\n"
            f"📍 *Proposed Entry:* `${decision.entry_price:,.2f}`\n"
            f"⚖️ *Proposed R:R:* `{decision.risk_reward:.2f}R`\n"
            f"🕒 *Time (IST):* `{local_time_str}`\n\n"
            f"_💡 Capital protected! System filtered out this sub-optimal setup._"
        )

        success = self._send_telegram_raw(message)
        if success:
            self.sent_signatures.add(signature)
        return success

    def _format_time(self, candle_ts: Optional[datetime]) -> str:
        kolkata_tz = pytz.timezone(DISPLAY_TIMEZONE)
        if candle_ts:
            if candle_ts.tzinfo is None:
                dt_utc = pytz.utc.localize(candle_ts)
            else:
                dt_utc = candle_ts.astimezone(pytz.utc)
            return dt_utc.astimezone(kolkata_tz).strftime("%Y-%m-%d %H:%M:%S IST")
        return datetime.now(kolkata_tz).strftime("%Y-%m-%d %H:%M:%S IST")

    def send_test_message(self) -> Dict[str, Any]:
        """
        Sends a test verification message to test Telegram connection.
        """
        if not self.is_configured:
            return {"success": False, "message": "Bot token or Chat ID is missing."}

        text = (
            "✅ *Telegram Alerts Connected Successfully!*\n\n"
            "Hybrid Adaptive Trading Dashboard is linked and ready to broadcast AI Approved and Rejected trade alerts."
        )
        ok = self._send_telegram_raw(text)
        return {
            "success": ok,
            "message": "Message sent successfully to Telegram!" if ok else "Failed to send message via Telegram API. Check Chat ID."
        }

    def _send_telegram_raw(self, text: str, target_chat_id: Optional[str] = None) -> bool:
        cid = target_chat_id or self.chat_id
        if not cid:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": cid,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                return True
            # Fallback without parse_mode if entity markdown failed
            payload_fallback = {
                "chat_id": cid,
                "text": text.replace("*", "").replace("`", "").replace("_", "")
            }
            fb_resp = requests.post(url, json=payload_fallback, timeout=8)
            return fb_resp.status_code == 200
        except Exception:
            return False

    def start_listener(self, db_manager, starting_balance_fn):
        """Starts a background daemon thread that listens for user commands (p/l, balance, etc.)."""
        TelegramAlertDispatcher._db_manager = db_manager
        TelegramAlertDispatcher._starting_balance_fn = starting_balance_fn

        if TelegramAlertDispatcher._listener_running and TelegramAlertDispatcher._listener_thread and TelegramAlertDispatcher._listener_thread.is_alive():
            return

        if not self.bot_token:
            return

        # Fetch latest update_id so ancient unread messages don't spam
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates?offset=-1"
            res = requests.get(url, timeout=5).json()
            if res.get("ok") and res.get("result"):
                TelegramAlertDispatcher._last_update_id = res["result"][-1]["update_id"]
        except Exception:
            pass

        TelegramAlertDispatcher._listener_running = True
        TelegramAlertDispatcher._listener_thread = threading.Thread(
            target=self._listener_worker,
            daemon=True,
            name="TelegramCommandListener"
        )
        TelegramAlertDispatcher._listener_thread.start()

    def _listener_worker(self):
        while TelegramAlertDispatcher._listener_running:
            try:
                if not self.bot_token:
                    time.sleep(3)
                    continue

                url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
                params = {
                    "offset": TelegramAlertDispatcher._last_update_id + 1,
                    "timeout": 2
                }
                resp = requests.get(url, params=params, timeout=5)
                data = resp.json()

                if data.get("ok"):
                    for update in data.get("result", []):
                        TelegramAlertDispatcher._last_update_id = update["update_id"]
                        msg = update.get("message")
                        if not msg:
                            continue

                        chat = msg.get("chat", {})
                        chat_id = str(chat.get("id", ""))
                        text = (msg.get("text") or "").strip().lower()

                        if not text:
                            continue

                        # If user hasn't configured chat_id yet, auto-set it from incoming message
                        if not self.chat_id:
                            self.set_chat_id(chat_id)
                            self._save_chat_id_to_env(chat_id)

                        starting_bal = 10000.0
                        if TelegramAlertDispatcher._starting_balance_fn:
                            try:
                                starting_bal = float(TelegramAlertDispatcher._starting_balance_fn())
                            except Exception:
                                starting_bal = 10000.0

                        if text in ["p/l", "pnl", "/pnl", "pl", "profit", "loss", "p&l"]:
                            reply = self._build_pnl_report(starting_bal)
                            self._send_telegram_raw(reply, target_chat_id=chat_id)
                        elif text in ["balance", "/balance", "bal", "capital"]:
                            reply = self._build_balance_report(starting_bal)
                            self._send_telegram_raw(reply, target_chat_id=chat_id)
                        elif text in ["open", "/open", "positions", "active"]:
                            reply = self._build_open_positions_report()
                            self._send_telegram_raw(reply, target_chat_id=chat_id)
                        elif text in ["trades", "/trades", "history", "log"]:
                            reply = self._build_recent_trades_report()
                            self._send_telegram_raw(reply, target_chat_id=chat_id)
                        elif text in ["help", "/help", "/start", "hi", "hello", "menu"]:
                            reply = self._build_help_message()
                            self._send_telegram_raw(reply, target_chat_id=chat_id)
            except Exception:
                pass

            time.sleep(2)

    def _build_pnl_report(self, starting_balance: float) -> str:
        if not TelegramAlertDispatcher._db_manager:
            return "⚠️ Database manager not connected yet."

        all_trades = TelegramAlertDispatcher._db_manager.get_all_paper_trades(limit=500)
        closed_trades = [t for t in all_trades if str(t.get("status", "")).startswith("CLOSED")]
        open_trades = [t for t in all_trades if t.get("status") == "OPEN"]

        realized_pnl = sum(t.get("pnl_net", 0.0) for t in closed_trades)
        current_balance = starting_balance + realized_pnl

        tp_wins = [t for t in closed_trades if "TP" in str(t.get("status", ""))]
        sl_losses = [t for t in closed_trades if "SL" in str(t.get("status", "")) or "TRAILING" in str(t.get("status", ""))]
        win_rate = (len(tp_wins) / len(closed_trades) * 100) if closed_trades else 0.0

        pnl_prefix = "+" if realized_pnl >= 0 else "-"
        pnl_pct = (realized_pnl / starting_balance * 100) if starting_balance > 0 else 0.0

        kolkata_tz = pytz.timezone(DISPLAY_TIMEZONE)
        time_str = datetime.now(kolkata_tz).strftime("%Y-%m-%d %I:%M:%S %p IST")

        open_info = ""
        if open_trades:
            lines = []
            for ot in open_trades[:3]:
                side_emoji = "🟢" if ot.get("direction") == "BUY" else "🔴"
                lines.append(
                    f"• {side_emoji} *{ot.get('direction')} {ot.get('symbol')}*\n"
                    f"  Entry: `${ot.get('executed_entry_price', 0):,.2f}` | TP: `${ot.get('tp1', 0):,.2f}` | SL: `${ot.get('stop_loss', 0):,.2f}`"
                )
            open_info = "\n⚡ *Open Positions:*\n" + "\n".join(lines) + "\n"
        else:
            open_info = "\n⚡ *Open Positions:* None (All positions closed)\n"

        return (
            f"📊 *LIVE PORTFOLIO & P&L REPORT* 📊\n\n"
            f"💰 *Current Balance:* *${current_balance:,.2f}*\n"
            f"🏦 *Starting Capital:* `${starting_balance:,.2f}`\n"
            f"💵 *Total Realized P&L:* *{pnl_prefix}${abs(realized_pnl):,.2f}* ({pnl_prefix}{abs(pnl_pct):.2f}%)\n\n"
            f"🏆 *Trade Summary:*\n"
            f"• 🎯 *Target Hits (Wins):* `{len(tp_wins)}`\n"
            f"• 🛑 *Stop-Loss Hits (Loss):* `{len(sl_losses)}`\n"
            f"• 🏁 *Total Closed Trades:* `{len(closed_trades)}`\n"
            f"• 📈 *Win Rate:* `{win_rate:.1f}%`\n"
            f"{open_info}\n"
            f"🕒 *Time:* `{time_str}`\n\n"
            f"_💡 Send 'trades' for history or 'balance' for capital summary._"
        )

    def _build_balance_report(self, starting_balance: float) -> str:
        if not TelegramAlertDispatcher._db_manager:
            return f"💰 *Current Account Balance:* `${starting_balance:,.2f}`"

        all_trades = TelegramAlertDispatcher._db_manager.get_all_paper_trades(limit=500)
        closed_trades = [t for t in all_trades if str(t.get("status", "")).startswith("CLOSED")]
        realized_pnl = sum(t.get("pnl_net", 0.0) for t in closed_trades)
        current_balance = starting_balance + realized_pnl
        pnl_prefix = "+" if realized_pnl >= 0 else "-"

        return (
            f"💰 *ACCOUNT BALANCE SUMMARY* 💰\n\n"
            f"💵 *Current Balance:* *${current_balance:,.2f}*\n"
            f"🏦 *Starting Capital:* `${starting_balance:,.2f}`\n"
            f"📈 *Net Realized P&L:* *{pnl_prefix}${abs(realized_pnl):,.2f}*\n\n"
            f"_💡 Send 'p/l' for complete performance report._"
        )

    def _build_open_positions_report(self) -> str:
        if not TelegramAlertDispatcher._db_manager:
            return "⚡ No open positions found."

        all_trades = TelegramAlertDispatcher._db_manager.get_all_paper_trades(limit=100)
        open_trades = [t for t in all_trades if t.get("status") == "OPEN"]
        if not open_trades:
            return "⚡ *Active Positions:* No open trades running currently."

        lines = ["⚡ *ACTIVE OPEN POSITIONS* ⚡\n"]
        for ot in open_trades:
            emoji = "🟢" if ot.get("direction") == "BUY" else "🔴"
            lines.append(
                f"{emoji} *{ot.get('direction')} {ot.get('symbol')}*\n"
                f"• ID: `{ot.get('trade_id')}`\n"
                f"• Entry: `${ot.get('executed_entry_price', 0):,.2f}`\n"
                f"• TP1: `${ot.get('tp1', 0):,.2f}`\n"
                f"• Stop-Loss: `${ot.get('stop_loss', 0):,.2f}`\n"
            )
        return "\n".join(lines)

    def _build_recent_trades_report(self) -> str:
        if not TelegramAlertDispatcher._db_manager:
            return "📜 No trades logged yet."

        all_trades = TelegramAlertDispatcher._db_manager.get_all_paper_trades(limit=10)
        closed = [t for t in all_trades if str(t.get("status", "")).startswith("CLOSED")]
        if not closed:
            return "📜 *Recent Trades:* No completed trades in journal yet."

        lines = ["📜 *RECENT CLOSED TRADES* 📜\n"]
        for t in closed[:5]:
            pnl = t.get("pnl_net", 0.0)
            prefix = "+" if pnl >= 0 else "-"
            icon = "🎯" if "TP" in t.get("status", "") else "🛑"
            lines.append(
                f"{icon} *{t.get('direction')} {t.get('symbol')}* [{t.get('status')}]\n"
                f"  P&L: *{prefix}${abs(pnl):,.2f}* | Exit: `${t.get('exit_price', 0):,.2f}` | R: `{t.get('r_multiple', 0):+.2f}R`\n"
            )
        return "\n".join(lines)

    def _build_help_message(self) -> str:
        return (
            "🤖 *Hybrid Adaptive Trading Bot Commands* 🤖\n\n"
            "Aap Telegram par ye messages bhej sakte hain:\n\n"
            "• *p/l* ya */pnl* 👉 Live Profit/Loss aur Account Balance report\n"
            "• *balance* 👉 Current account balance check karein\n"
            "• *open* 👉 Running active positions dekhein\n"
            "• *trades* 👉 Recent trades aur P&L history dekhein\n"
            "• *help* 👉 Yeh help message dekhein\n\n"
            "⚡ _Bot 24/7 active hai aur turant reply karega!_"
        )

