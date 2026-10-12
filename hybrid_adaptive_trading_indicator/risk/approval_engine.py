"""
Deterministic Risk Approval Engine.
Evaluates candidate trade setups against strict risk rules:
- Max daily loss limit
- Max concurrent trades
- Duplicate open position protection
- Minimum Risk/Reward ratio
- Position sizing & contract spec constraints
Returns structured RiskDecision: APPROVE, REJECT, or WAIT.
"""
from typing import Optional, List, Dict, Any
from hybrid_adaptive_trading_indicator.config import InstrumentSpec, INSTRUMENT_SPECS
from hybrid_adaptive_trading_indicator.strategy.models import TradeCandidate
from hybrid_adaptive_trading_indicator.risk.models import RiskDecision
from hybrid_adaptive_trading_indicator.risk.position_sizer import calculate_position_size

class RiskApprovalEngine:
    def __init__(
        self,
        account_balance: float = 10000.0,
        risk_per_trade_pct: float = 1.0,
        max_daily_loss: float = 300.0,
        max_concurrent_trades: int = 2,
        min_risk_reward: float = 1.50,
        instrument_specs: Optional[Dict[str, InstrumentSpec]] = None
    ):
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_daily_loss = max_daily_loss
        self.max_concurrent_trades = max_concurrent_trades
        self.min_risk_reward = min_risk_reward
        self.specs = instrument_specs or INSTRUMENT_SPECS

    def evaluate(
        self,
        candidate: Optional[TradeCandidate],
        current_daily_loss: float = 0.0,
        open_positions: Optional[List[Dict[str, Any]]] = None
    ) -> RiskDecision:
        """
        Runs deterministic checks on a candidate setup.
        Returns a structured RiskDecision.
        """
        positions = open_positions or []

        if candidate is None:
            return RiskDecision(
                status="WAIT",
                symbol="",
                timeframe="",
                regime="",
                direction="",
                entry_price=0.0,
                stop_loss=0.0,
                tp1=0.0,
                tp2=0.0,
                risk_reward=0.0,
                setup_reason="No setup formed on latest candle. Waiting for confirmed closed candle condition.",
                candle_timestamp=None,
                rejection_reason="",
                position_size=0.0,
                position_unit="",
                risk_amount_usd=0.0,
                estimated_cost_usd=0.0,
                net_risk_reward=0.0,
                max_capital_at_risk_pct=0.0
            )

        spec = self.specs.get(candidate.symbol)
        if not spec:
            return self._reject(
                candidate,
                rejection_reason=f"Instrument specification not configured for symbol '{candidate.symbol}'."
            )

        # 1. Check Max Daily Loss
        if current_daily_loss >= self.max_daily_loss:
            return self._reject(
                candidate,
                rejection_reason=f"Max daily loss limit of ${self.max_daily_loss:.2f} already reached (Current realized loss: ${current_daily_loss:.2f})."
            )

        # 2. Check Max Concurrent Open Trades
        if len(positions) >= self.max_concurrent_trades:
            return self._reject(
                candidate,
                rejection_reason=f"Maximum concurrent trades limit ({self.max_concurrent_trades}) reached. Active positions: {len(positions)}."
            )

        # 3. Duplicate Trade Protection
        for pos in positions:
            if pos.get("symbol") == candidate.symbol and pos.get("direction") == candidate.direction:
                return self._reject(
                    candidate,
                    rejection_reason=f"Duplicate trade: An active {candidate.direction} position on {candidate.symbol} is already open."
                )

        # 4. Position Sizing
        try:
            pos_size, actual_risk, friction, total_risk = calculate_position_size(
                spec=spec,
                account_balance=self.account_balance,
                risk_pct=self.risk_per_trade_pct,
                entry_price=candidate.entry_price,
                stop_loss=candidate.stop_loss
            )
        except Exception as e:
            return self._reject(
                candidate,
                rejection_reason=f"Position sizing calculation error: {e}"
            )

        unit_label = "lots" if spec.asset_type == "commodity" else spec.symbol.replace("USDT", "")
        if pos_size < spec.min_order_qty:
            return self._reject(
                candidate,
                rejection_reason=f"Calculated position size ({pos_size:.4f} {unit_label}) is below minimum allowed order size ({spec.min_order_qty} {unit_label}) for account risk budget (${self.account_balance * (self.risk_per_trade_pct / 100.0):.2f})."
            )

        # 5. Check if taking this trade could breach max daily loss
        if (current_daily_loss + total_risk) > (self.max_daily_loss * 1.5):
            return self._reject(
                candidate,
                rejection_reason=f"Risk amount (${total_risk:.2f}) plus current daily loss (${current_daily_loss:.2f}) exceeds risk safety boundary."
            )

        # 6. Minimum Risk/Reward Check (incorporating friction costs)
        gross_reward = abs(candidate.tp1 - candidate.entry_price) * spec.contract_size * pos_size
        net_reward = max(gross_reward - friction, 0.0)
        net_rr = net_reward / max(total_risk, 0.01)

        if net_rr < self.min_risk_reward:
            return self._reject(
                candidate,
                rejection_reason=f"Net Risk/Reward ratio ({net_rr:.2f}) after friction (${friction:.2f}) is below minimum threshold of {self.min_risk_reward:.2f}.",
                position_size=pos_size,
                unit_label=unit_label,
                risk_usd=total_risk,
                friction_usd=friction,
                net_rr=net_rr
            )

        # APPROVED!
        risk_pct_effective = (total_risk / self.account_balance) * 100.0
        return RiskDecision(
            status="APPROVE",
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            regime=candidate.regime,
            direction=candidate.direction,
            entry_price=candidate.entry_price,
            stop_loss=candidate.stop_loss,
            tp1=candidate.tp1,
            tp2=candidate.tp2,
            risk_reward=candidate.risk_reward_ratio,
            setup_reason=candidate.setup_reason,
            candle_timestamp=candidate.candle_timestamp,
            rejection_reason="",
            position_size=pos_size,
            position_unit=unit_label,
            risk_amount_usd=total_risk,
            estimated_cost_usd=friction,
            net_risk_reward=round(net_rr, 2),
            max_capital_at_risk_pct=round(risk_pct_effective, 2)
        )

    def _reject(
        self,
        candidate: TradeCandidate,
        rejection_reason: str,
        position_size: float = 0.0,
        unit_label: str = "",
        risk_usd: float = 0.0,
        friction_usd: float = 0.0,
        net_rr: float = 0.0
    ) -> RiskDecision:
        spec = self.specs.get(candidate.symbol)
        if not unit_label and spec:
            unit_label = "lots" if spec.asset_type == "commodity" else spec.symbol.replace("USDT", "")

        return RiskDecision(
            status="REJECT",
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            regime=candidate.regime,
            direction=candidate.direction,
            entry_price=candidate.entry_price,
            stop_loss=candidate.stop_loss,
            tp1=candidate.tp1,
            tp2=candidate.tp2,
            risk_reward=candidate.risk_reward_ratio,
            setup_reason=candidate.setup_reason,
            candle_timestamp=candidate.candle_timestamp,
            rejection_reason=rejection_reason,
            position_size=position_size,
            position_unit=unit_label,
            risk_amount_usd=risk_usd,
            estimated_cost_usd=friction_usd,
            net_risk_reward=round(net_rr, 2),
            max_capital_at_risk_pct=round((risk_usd / self.account_balance) * 100.0, 2) if self.account_balance > 0 else 0.0
        )
