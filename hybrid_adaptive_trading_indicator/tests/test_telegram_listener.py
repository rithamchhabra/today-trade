"""
Unit tests for interactive Telegram command listeners and report builders.
"""
import pytest
from datetime import datetime, timezone
from hybrid_adaptive_trading_indicator.alerts.telegram import TelegramAlertDispatcher
from hybrid_adaptive_trading_indicator.storage.database import DatabaseManager

class MockDB:
    def get_all_paper_trades(self, limit=500):
        return [
            {
                "trade_id": "TRD-WIN01",
                "symbol": "XAUUSD",
                "direction": "BUY",
                "status": "CLOSED_TP1",
                "executed_entry_price": 2350.0,
                "exit_price": 2365.0,
                "tp1": 2365.0,
                "stop_loss": 2340.0,
                "pnl_net": 150.0,
                "r_multiple": 1.5,
                "exit_reason": "Take-Profit 1 reached",
                "signal_time": datetime.now(timezone.utc).isoformat()
            },
            {
                "trade_id": "TRD-LOSS01",
                "symbol": "BTCUSDT",
                "direction": "SELL",
                "status": "CLOSED_SL",
                "executed_entry_price": 65000.0,
                "exit_price": 65500.0,
                "tp1": 64000.0,
                "stop_loss": 65500.0,
                "pnl_net": -50.0,
                "r_multiple": -1.0,
                "exit_reason": "Stop-loss hit",
                "signal_time": datetime.now(timezone.utc).isoformat()
            },
            {
                "trade_id": "TRD-OPEN01",
                "symbol": "XAUUSD",
                "direction": "BUY",
                "status": "OPEN",
                "executed_entry_price": 2355.0,
                "exit_price": None,
                "tp1": 2370.0,
                "stop_loss": 2345.0,
                "pnl_net": 0.0,
                "r_multiple": 0.0,
                "exit_reason": "",
                "signal_time": datetime.now(timezone.utc).isoformat()
            }
        ]

def test_telegram_pnl_report():
    dispatcher = TelegramAlertDispatcher()
    TelegramAlertDispatcher._db_manager = MockDB()
    
    report = dispatcher._build_pnl_report(starting_balance=10000.0)
    assert "LIVE PORTFOLIO & P&L REPORT" in report
    assert "$10,100.00" in report  # 10000 + 150 - 50 = 10100
    assert "+$100.00" in report
    assert "*Target Hits (Wins):* `1`" in report
    assert "*Stop-Loss Hits (Loss):* `1`" in report
    assert "BUY XAUUSD" in report

def test_telegram_balance_report():
    dispatcher = TelegramAlertDispatcher()
    TelegramAlertDispatcher._db_manager = MockDB()
    
    report = dispatcher._build_balance_report(starting_balance=10000.0)
    assert "ACCOUNT BALANCE SUMMARY" in report
    assert "$10,100.00" in report

def test_telegram_help_message():
    dispatcher = TelegramAlertDispatcher()
    help_msg = dispatcher._build_help_message()
    assert "p/l" in help_msg
    assert "balance" in help_msg
