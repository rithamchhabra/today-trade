"""
Backtesting performance metrics calculation.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from hybrid_adaptive_trading_indicator.execution.models import PaperTrade

@dataclass
class BacktestSummary:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    total_gross_pnl: float
    total_fees_paid: float
    total_net_pnl: float
    return_on_capital_pct: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    average_r_multiple: float
    expectancy_usd: float
    starting_balance: float
    ending_balance: float

def compute_backtest_metrics(
    trades: List[PaperTrade],
    initial_balance: float = 10000.0
) -> Tuple[BacktestSummary, pd.DataFrame]:
    """
    Computes comprehensive backtest metrics and equity curve.
    """
    if not trades:
        empty_summary = BacktestSummary(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            total_gross_pnl=0.0,
            total_fees_paid=0.0,
            total_net_pnl=0.0,
            return_on_capital_pct=0.0,
            max_drawdown_usd=0.0,
            max_drawdown_pct=0.0,
            average_r_multiple=0.0,
            expectancy_usd=0.0,
            starting_balance=initial_balance,
            ending_balance=initial_balance
        )
        return empty_summary, pd.DataFrame(columns=["trade_id", "pnl_net", "equity"])

    pnls = [t.pnl_net for t in trades]
    gross_profits = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    total_net = sum(pnls)
    total_fees = sum(t.fees_paid for t in trades)
    total_gross = sum(t.pnl_gross for t in trades)
    r_multiples = [t.r_multiple for t in trades]

    win_count = sum(1 for p in pnls if p > 0)
    loss_count = sum(1 for p in pnls if p <= 0)
    total_trades = len(trades)
    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (999.0 if gross_profits > 0 else 0.0)

    # Equity curve & Max Drawdown
    equity = initial_balance
    equity_series = [initial_balance]
    peak = initial_balance
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    trade_records = []
    for t in trades:
        equity += t.pnl_net
        equity_series.append(equity)
        if equity > peak:
            peak = equity
        dd_usd = peak - equity
        dd_pct = (dd_usd / peak) * 100.0 if peak > 0 else 0.0
        if dd_usd > max_dd_usd:
            max_dd_usd = dd_usd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        trade_records.append({
            "trade_id": t.trade_id,
            "direction": t.direction,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.executed_entry_price,
            "exit_price": t.exit_price,
            "pnl_gross": t.pnl_gross,
            "fees_paid": t.fees_paid,
            "pnl_net": t.pnl_net,
            "r_multiple": t.r_multiple,
            "equity": equity,
            "exit_reason": t.exit_reason
        })

    avg_r = float(np.mean(r_multiples)) if r_multiples else 0.0
    expectancy = total_net / total_trades if total_trades > 0 else 0.0
    roc_pct = (total_net / initial_balance) * 100.0

    summary = BacktestSummary(
        total_trades=total_trades,
        winning_trades=win_count,
        losing_trades=loss_count,
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(profit_factor, 2),
        total_gross_pnl=round(total_gross, 2),
        total_fees_paid=round(total_fees, 2),
        total_net_pnl=round(total_net, 2),
        return_on_capital_pct=round(roc_pct, 2),
        max_drawdown_usd=round(max_dd_usd, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        average_r_multiple=round(avg_r, 2),
        expectancy_usd=round(expectancy, 2),
        starting_balance=initial_balance,
        ending_balance=round(equity, 2)
    )

    return summary, pd.DataFrame(trade_records)
