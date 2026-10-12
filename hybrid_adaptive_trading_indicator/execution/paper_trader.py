"""
Paper Trading lifecycle and execution simulator.
Features:
- Next-candle executable open fill model with spread & slippage
- Strict conservative conflict resolution (if SL and TP both breached in single candle, SL is assumed hit first!)
- Dynamic trailing stop handling
- Realistic fee and commission deduction
"""
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
from hybrid_adaptive_trading_indicator.config import InstrumentSpec, INSTRUMENT_SPECS
from hybrid_adaptive_trading_indicator.risk.models import RiskDecision
from hybrid_adaptive_trading_indicator.execution.models import PaperTrade

class PaperTradingSimulator:
    def __init__(
        self,
        instrument_specs: Optional[Dict[str, InstrumentSpec]] = None,
        enable_trailing_stop: bool = True,
        trailing_atr_mult: float = 1.5
    ):
        self.specs = instrument_specs or INSTRUMENT_SPECS
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_atr_mult = trailing_atr_mult

    def create_trade_from_decision(self, decision: RiskDecision) -> Optional[PaperTrade]:
        """
        Creates a PENDING paper trade from an APPROVED risk decision.
        """
        if decision.status != "APPROVE":
            return None

        trade_id = f"TRD-{uuid.uuid4().hex[:8].upper()}"
        return PaperTrade(
            trade_id=trade_id,
            symbol=decision.symbol,
            direction=decision.direction,
            status="PENDING",
            signal_time=decision.candle_timestamp,
            target_entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            tp1=decision.tp1,
            tp2=decision.tp2,
            trailing_stop_active=self.enable_trailing_stop,
            current_trailing_stop=decision.stop_loss,
            position_size=decision.position_size,
            position_unit=decision.position_unit,
            setup_reason=decision.setup_reason
        )

    def process_candle(
        self,
        trade: PaperTrade,
        candle: pd.Series,
        atr: float = 0.0
    ) -> PaperTrade:
        """
        Processes an individual candle against an active paper trade.
        """
        if trade.status in ["CLOSED_TP1", "CLOSED_TP2", "CLOSED_SL", "CLOSED_TRAILING", "CANCELLED"]:
            return trade

        spec = self.specs.get(trade.symbol, INSTRUMENT_SPECS["XAUUSD"])
        c_open = float(candle["open"])
        c_high = float(candle["high"])
        c_low = float(candle["low"])
        c_close = float(candle["close"])
        c_time = pd.to_datetime(candle["datetime"]).to_pydatetime()

        # 1. Fill PENDING trade on the next candle open
        if trade.status == "PENDING":
            # Apply half spread + slippage to entry
            half_spread = spec.default_spread / 2.0
            slippage = spec.default_slippage
            if trade.direction == "BUY":
                executed_entry = c_open + half_spread + slippage
            else:
                executed_entry = c_open - half_spread - slippage

            trade.status = "OPEN"
            trade.entry_time = c_time
            trade.executed_entry_price = round(executed_entry, 4)
            trade.current_trailing_stop = trade.stop_loss
            return trade

        # 2. Manage OPEN trade
        if trade.status == "OPEN":
            contract_mult = spec.contract_size * trade.position_size
            initial_risk_dist = abs(trade.executed_entry_price - trade.stop_loss)
            fee_rate = spec.default_fee_pct

            if trade.direction == "BUY":
                # BUY Trade Evaluation
                sl_hit = c_low <= trade.current_trailing_stop
                tp1_hit = c_high >= trade.tp1
                tp2_hit = c_high >= trade.tp2

                # CONSERVATIVE RULE: If both SL and TP breached in same bar, SL assumed hit first!
                if sl_hit and (tp1_hit or tp2_hit):
                    trade.status = "CLOSED_SL"
                    trade.exit_time = c_time
                    trade.exit_price = trade.current_trailing_stop - spec.default_slippage
                    trade.exit_reason = "Conservative fill: Stop-Loss hit during high-volatility bar"
                elif sl_hit:
                    is_trailing = trade.current_trailing_stop > trade.stop_loss
                    trade.status = "CLOSED_TRAILING" if is_trailing else "CLOSED_SL"
                    trade.exit_time = c_time
                    trade.exit_price = trade.current_trailing_stop - spec.default_slippage
                    trade.exit_reason = "Trailing stop hit" if is_trailing else "Initial stop-loss hit"
                elif tp2_hit:
                    trade.status = "CLOSED_TP2"
                    trade.exit_time = c_time
                    trade.exit_price = trade.tp2
                    trade.exit_reason = "Take-Profit 2 reached"
                elif tp1_hit:
                    # Move stop to breakeven or start trailing if trailing stop active; else close at TP1
                    if trade.trailing_stop_active:
                        breakeven = trade.executed_entry_price + (spec.default_spread / 2.0)
                        if atr > 0:
                            trail_level = c_high - (self.trailing_atr_mult * atr)
                            trade.current_trailing_stop = max(trade.current_trailing_stop, breakeven, trail_level)
                        else:
                            trade.current_trailing_stop = max(trade.current_trailing_stop, breakeven)
                    else:
                        trade.status = "CLOSED_TP1"
                        trade.exit_time = c_time
                        trade.exit_price = trade.tp1
                        trade.exit_reason = "Take-Profit 1 reached"
                elif trade.trailing_stop_active and atr > 0:
                    # Trail higher as price progresses
                    trail_level = c_high - (self.trailing_atr_mult * atr)
                    if trail_level > trade.executed_entry_price:
                        trade.current_trailing_stop = max(trade.current_trailing_stop, trail_level)

            else:
                # SELL Trade Evaluation
                sl_hit = c_high >= trade.current_trailing_stop
                tp1_hit = c_low <= trade.tp1
                tp2_hit = c_low <= trade.tp2

                # CONSERVATIVE RULE:
                if sl_hit and (tp1_hit or tp2_hit):
                    trade.status = "CLOSED_SL"
                    trade.exit_time = c_time
                    trade.exit_price = trade.current_trailing_stop + spec.default_slippage
                    trade.exit_reason = "Conservative fill: Stop-Loss hit during high-volatility bar"
                elif sl_hit:
                    is_trailing = trade.current_trailing_stop < trade.stop_loss
                    trade.status = "CLOSED_TRAILING" if is_trailing else "CLOSED_SL"
                    trade.exit_time = c_time
                    trade.exit_price = trade.current_trailing_stop + spec.default_slippage
                    trade.exit_reason = "Trailing stop hit" if is_trailing else "Initial stop-loss hit"
                elif tp2_hit:
                    trade.status = "CLOSED_TP2"
                    trade.exit_time = c_time
                    trade.exit_price = trade.tp2
                    trade.exit_reason = "Take-Profit 2 reached"
                elif tp1_hit:
                    if trade.trailing_stop_active:
                        breakeven = trade.executed_entry_price - (spec.default_spread / 2.0)
                        if atr > 0:
                            trail_level = c_low + (self.trailing_atr_mult * atr)
                            trade.current_trailing_stop = min(trade.current_trailing_stop, breakeven, trail_level)
                        else:
                            trade.current_trailing_stop = min(trade.current_trailing_stop, breakeven)
                    else:
                        trade.status = "CLOSED_TP1"
                        trade.exit_time = c_time
                        trade.exit_price = trade.tp1
                        trade.exit_reason = "Take-Profit 1 reached"
                elif trade.trailing_stop_active and atr > 0:
                    trail_level = c_low + (self.trailing_atr_mult * atr)
                    if trail_level < trade.executed_entry_price:
                        trade.current_trailing_stop = min(trade.current_trailing_stop, trail_level)

            # If closed, calculate realized metrics
            if trade.exit_price is not None:
                trade.exit_price = round(trade.exit_price, 4)
                if trade.direction == "BUY":
                    gross = (trade.exit_price - trade.executed_entry_price) * contract_mult
                else:
                    gross = (trade.executed_entry_price - trade.exit_price) * contract_mult

                entry_val = trade.executed_entry_price * contract_mult
                exit_val = trade.exit_price * contract_mult
                fees = (entry_val + exit_val) * fee_rate
                net = gross - fees

                r_mult = net / max(initial_risk_dist * contract_mult, 0.01)

                trade.pnl_gross = round(gross, 2)
                trade.fees_paid = round(fees, 2)
                trade.pnl_net = round(net, 2)
                trade.r_multiple = round(r_mult, 2)

        return trade

    def update_trades_against_dataframe(
        self,
        trades: List[PaperTrade],
        candles_df: pd.DataFrame
    ) -> List[Tuple[PaperTrade, bool]]:
        """
        Updates open and pending paper trades bar-by-bar against the provided candles dataframe.
        Returns a list of (trade, was_just_closed) where was_just_closed is True if trade just transitioned
        to a closed state (CLOSED_TP1, CLOSED_TP2, CLOSED_SL, CLOSED_TRAILING).
        """
        results = []
        if candles_df is None or candles_df.empty or not trades:
            return [(t, False) for t in trades]

        df_sorted = candles_df.sort_values("datetime").reset_index(drop=True)

        for trade in trades:
            if trade.status in ["CLOSED_TP1", "CLOSED_TP2", "CLOSED_SL", "CLOSED_TRAILING", "CANCELLED"]:
                results.append((trade, False))
                continue

            trade_sig_time = trade.signal_time
            # Ensure safe timezone-aware comparison
            candle_dts = pd.to_datetime(df_sorted["datetime"], utc=True)
            if trade_sig_time is not None:
                if trade_sig_time.tzinfo is None:
                    trade_sig_aware = trade_sig_time.replace(tzinfo=pd.to_datetime("now", utc=True).tzinfo)
                else:
                    trade_sig_aware = trade_sig_time
                mask = candle_dts >= trade_sig_aware
            else:
                mask = pd.Series([True] * len(df_sorted))

            sub_candles = df_sorted[mask]
            was_just_closed = False

            for _, candle in sub_candles.iterrows():
                prev_status = trade.status
                atr_val = float(candle["atr"]) if "atr" in candle and pd.notna(candle["atr"]) else 0.0
                trade = self.process_candle(trade, candle, atr=atr_val)

                if prev_status != trade.status and trade.status.startswith("CLOSED"):
                    was_just_closed = True
                    break

            results.append((trade, was_just_closed))

        return results
