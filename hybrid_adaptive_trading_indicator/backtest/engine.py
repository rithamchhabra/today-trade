"""
Backtest engine simulating historical strategy execution.
Runs the exact same non-repainting pipeline and risk approval engine across historical data.
"""
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from hybrid_adaptive_trading_indicator.strategy.engine import prepare_indicators, evaluate_all_candidates
from hybrid_adaptive_trading_indicator.risk.approval_engine import RiskApprovalEngine
from hybrid_adaptive_trading_indicator.execution.paper_trader import PaperTradingSimulator
from hybrid_adaptive_trading_indicator.execution.models import PaperTrade
from hybrid_adaptive_trading_indicator.backtest.metrics import compute_backtest_metrics, BacktestSummary
from hybrid_adaptive_trading_indicator.config import INSTRUMENT_SPECS

def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "15m",
    initial_balance: float = 10000.0,
    risk_pct: float = 1.0,
    max_daily_loss: float = 300.0,
    max_concurrent_trades: int = 2,
    min_risk_reward: float = 1.5,
    adx_len: int = 14,
    atr_len: int = 14,
    pivot_left: int = 3,
    pivot_right: int = 3,
    adx_sideways: float = 20.0,
    adx_trending: float = 25.0,
    sl_atr_mult: float = 1.5,
    target_rr: float = 2.0,
    use_volume: bool = False,
    use_retest: bool = False,
    enable_trailing_stop: bool = True
) -> Tuple[BacktestSummary, pd.DataFrame, List[PaperTrade]]:
    """
    Simulates trades bar-by-bar across historical candles.
    """
    # 1. Prepare indicators non-repainting
    ind_df = prepare_indicators(
        df=df,
        adx_len=adx_len,
        atr_len=atr_len,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        adx_sideways=adx_sideways,
        adx_trending=adx_trending
    )

    # 2. Evaluate candidate setups
    annotated_df, candidates = evaluate_all_candidates(
        df=ind_df,
        symbol=symbol,
        timeframe=timeframe,
        sl_atr_mult=sl_atr_mult,
        target_rr=target_rr,
        use_volume=use_volume,
        use_retest=use_retest
    )

    # 3. Simulate through candles
    risk_engine = RiskApprovalEngine(
        account_balance=initial_balance,
        risk_per_trade_pct=risk_pct,
        max_daily_loss=max_daily_loss,
        max_concurrent_trades=max_concurrent_trades,
        min_risk_reward=min_risk_reward
    )

    simulator = PaperTradingSimulator(
        enable_trailing_stop=enable_trailing_stop
    )

    candidate_by_ts = {c.candle_timestamp: c for c in candidates}
    closed_trades: List[PaperTrade] = []
    active_trades: List[PaperTrade] = []
    current_balance = initial_balance

    for i in range(len(annotated_df)):
        row = annotated_df.iloc[i]
        c_ts = pd.to_datetime(row["datetime"]).to_pydatetime()
        c_atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0

        # Process active trades on this candle
        remaining_active = []
        for trade in active_trades:
            updated_trade = simulator.process_candle(trade, row, atr=c_atr)
            if updated_trade.status in ["CLOSED_TP1", "CLOSED_TP2", "CLOSED_SL", "CLOSED_TRAILING"]:
                closed_trades.append(updated_trade)
                current_balance += updated_trade.pnl_net
                risk_engine.account_balance = max(current_balance, 100.0)
            else:
                remaining_active.append(updated_trade)
        active_trades = remaining_active

        # Check if candidate formed on this candle close
        if c_ts in candidate_by_ts:
            candidate = candidate_by_ts[c_ts]
            open_pos_dicts = [{"symbol": t.symbol, "direction": t.direction} for t in active_trades if t.status == "OPEN"]
            decision = risk_engine.evaluate(
                candidate=candidate,
                current_daily_loss=0.0,  # can track rolling intraday loss
                open_positions=open_pos_dicts
            )

            if decision.status == "APPROVE":
                new_trade = simulator.create_trade_from_decision(decision)
                if new_trade:
                    active_trades.append(new_trade)

    summary, trades_df = compute_backtest_metrics(closed_trades, initial_balance=initial_balance)
    return summary, trades_df, closed_trades
