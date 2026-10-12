"""
SQLite database schema and repository for signals, alerts, and paper trading journal.
"""
import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from hybrid_adaptive_trading_indicator.execution.models import PaperTrade
from hybrid_adaptive_trading_indicator.risk.models import RiskDecision

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trading_system.db"
)

class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Signals table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                candle_timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1 REAL NOT NULL,
                tp2 REAL NOT NULL,
                risk_reward REAL NOT NULL,
                setup_reason TEXT NOT NULL,
                status TEXT NOT NULL,
                rejection_reason TEXT,
                position_size REAL,
                position_unit TEXT,
                risk_amount_usd REAL,
                created_at TEXT NOT NULL
            );
            """)

            # Paper trades table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                status TEXT NOT NULL,
                signal_time TEXT NOT NULL,
                entry_time TEXT,
                exit_time TEXT,
                target_entry_price REAL NOT NULL,
                executed_entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1 REAL NOT NULL,
                tp2 REAL NOT NULL,
                current_trailing_stop REAL,
                position_size REAL NOT NULL,
                position_unit TEXT NOT NULL,
                exit_price REAL,
                exit_reason TEXT,
                pnl_gross REAL DEFAULT 0.0,
                fees_paid REAL DEFAULT 0.0,
                pnl_net REAL DEFAULT 0.0,
                r_multiple REAL DEFAULT 0.0,
                setup_reason TEXT,
                updated_at TEXT NOT NULL
            );
            """)

            # Alerts table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                message TEXT NOT NULL,
                delivered INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """)
            conn.commit()

    def save_signal(self, decision: RiskDecision) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        ts_str = decision.candle_timestamp.isoformat() if decision.candle_timestamp else now_str

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT id FROM signals 
            WHERE symbol = ? AND candle_timestamp = ? AND direction = ?
            LIMIT 1
            """, (decision.symbol, ts_str, decision.direction))
            existing = cursor.fetchone()
            if existing:
                cursor.execute("""
                UPDATE signals SET
                    regime = ?, entry_price = ?, stop_loss = ?, tp1 = ?, tp2 = ?,
                    risk_reward = ?, setup_reason = ?, status = ?, rejection_reason = ?,
                    position_size = ?, position_unit = ?, risk_amount_usd = ?
                WHERE id = ?
                """, (
                    decision.regime, decision.entry_price, decision.stop_loss, decision.tp1, decision.tp2,
                    decision.risk_reward, decision.setup_reason, decision.status, decision.rejection_reason,
                    decision.position_size, decision.position_unit, decision.risk_amount_usd, existing["id"]
                ))
                conn.commit()
                return existing["id"]

            cursor.execute("""
            INSERT INTO signals (
                symbol, timeframe, candle_timestamp, regime, direction,
                entry_price, stop_loss, tp1, tp2, risk_reward,
                setup_reason, status, rejection_reason, position_size,
                position_unit, risk_amount_usd, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.symbol, decision.timeframe, ts_str, decision.regime,
                decision.direction, decision.entry_price, decision.stop_loss,
                decision.tp1, decision.tp2, decision.risk_reward,
                decision.setup_reason, decision.status, decision.rejection_reason,
                decision.position_size, decision.position_unit, decision.risk_amount_usd,
                now_str
            ))
            conn.commit()
            return cursor.lastrowid

    def save_or_update_trade(self, trade: PaperTrade):
        now_str = datetime.now(timezone.utc).isoformat()
        td = trade.to_dict()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO paper_trades (
                trade_id, symbol, direction, status, signal_time, entry_time, exit_time,
                target_entry_price, executed_entry_price, stop_loss, tp1, tp2,
                current_trailing_stop, position_size, position_unit, exit_price,
                exit_reason, pnl_gross, fees_paid, pnl_net, r_multiple, setup_reason, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                status=excluded.status,
                entry_time=excluded.entry_time,
                exit_time=excluded.exit_time,
                executed_entry_price=excluded.executed_entry_price,
                current_trailing_stop=excluded.current_trailing_stop,
                exit_price=excluded.exit_price,
                exit_reason=excluded.exit_reason,
                pnl_gross=excluded.pnl_gross,
                fees_paid=excluded.fees_paid,
                pnl_net=excluded.pnl_net,
                r_multiple=excluded.r_multiple,
                updated_at=excluded.updated_at
            """, (
                td["trade_id"], td["symbol"], td["direction"], td["status"],
                td["signal_time"], td["entry_time"], td["exit_time"],
                td["target_entry_price"], td["executed_entry_price"], td["stop_loss"],
                td["tp1"], td["tp2"], td["current_trailing_stop"], td["position_size"],
                td["position_unit"], td["exit_price"], td["exit_reason"],
                td["pnl_gross"], td["fees_paid"], td["pnl_net"], td["r_multiple"],
                td["setup_reason"], now_str
            ))
            conn.commit()

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute("SELECT * FROM paper_trades WHERE status = 'OPEN' AND symbol = ?", (symbol,))
            else:
                cursor.execute("SELECT * FROM paper_trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_todays_realized_loss(self) -> float:
        """
        Calculates total negative net PnL from trades closed today (UTC).
        """
        today_start = datetime.now(timezone.utc).strftime("%Y-%m-%d 00:00:00")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT SUM(pnl_net) as total_loss FROM paper_trades
            WHERE exit_time >= ? AND pnl_net < 0
            """, (today_start,))
            row = cursor.fetchone()
            if row and row["total_loss"] is not None:
                return abs(float(row["total_loss"]))
            return 0.0

    def get_all_paper_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM paper_trades ORDER BY signal_time DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_signals(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM signals"
            params = []
            conditions = []
            if symbol and symbol != "ALL":
                conditions.append("symbol = ?")
                params.append(symbol)
            if status and status != "ALL":
                conditions.append("status = ?")
                params.append(status)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY candle_timestamp DESC, id DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def record_alert(self, symbol: str, direction: str, message: str, delivered: bool) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO alerts (symbol, direction, message, delivered, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (symbol, direction, message, 1 if delivered else 0, now_str))
            conn.commit()
            return cursor.lastrowid
