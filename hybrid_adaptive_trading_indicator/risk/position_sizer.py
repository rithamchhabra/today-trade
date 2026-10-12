"""
Deterministic position sizing and trading friction calculations.
Supports configurable contract specifications for Gold (lots, troy oz) and Bitcoin (units).
"""
import math
from typing import Tuple
from hybrid_adaptive_trading_indicator.config import InstrumentSpec

def calculate_position_size(
    spec: InstrumentSpec,
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float
) -> Tuple[float, float, float, float]:
    """
    Calculates deterministic position size, monetary risk, friction cost, and net risk.
    
    Returns:
        (position_size, actual_risk_usd, estimated_friction_usd, total_capital_at_risk)
    """
    if account_balance <= 0:
        raise ValueError("Account balance must be positive.")
    if risk_pct <= 0 or risk_pct > 100:
        raise ValueError("Risk percentage must be between 0 and 100.")
    if spec.contract_size <= 0 or spec.tick_size <= 0:
        raise ValueError(f"Invalid contract specification for {spec.symbol}.")

    sl_distance = abs(entry_price - stop_loss)
    if sl_distance <= 0:
        raise ValueError("Stop loss cannot equal entry price.")

    target_risk_usd = account_balance * (risk_pct / 100.0)

    # Calculate loss per unit/contract
    # For Gold: 1 lot = contract_size (100 oz). A price move of $1.00 = $100 per lot.
    loss_per_contract = sl_distance * spec.contract_size

    raw_position_size = target_risk_usd / loss_per_contract

    # Quantize to qty_step and min_order_qty
    step = spec.qty_step
    # Floor to nearest step
    qty_steps = math.floor(raw_position_size / step)
    position_size = round(qty_steps * step, 4)

    if position_size < spec.min_order_qty:
        # Minimum position size cannot be satisfied within risk budget
        return 0.0, 0.0, 0.0, 0.0

    actual_risk_usd = round(position_size * loss_per_contract, 2)

    # Estimate trading frictions: spread + slippage + commission
    spread_cost = spec.default_spread * spec.contract_size * position_size
    slippage_cost = spec.default_slippage * spec.contract_size * position_size
    notional_value = entry_price * spec.contract_size * position_size
    fee_cost = notional_value * spec.default_fee_pct * 2.0  # roundtrip
    total_friction_usd = round(spread_cost + slippage_cost + fee_cost, 2)

    total_capital_at_risk = round(actual_risk_usd + total_friction_usd, 2)
    return position_size, actual_risk_usd, total_friction_usd, total_capital_at_risk
