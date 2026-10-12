"""
Strategy package.
"""
from hybrid_adaptive_trading_indicator.strategy.regime import classify_regimes
from hybrid_adaptive_trading_indicator.strategy.models import TradeCandidate
from hybrid_adaptive_trading_indicator.strategy.sideways import evaluate_sideways_setup
from hybrid_adaptive_trading_indicator.strategy.trending import evaluate_trending_setup
from hybrid_adaptive_trading_indicator.strategy.engine import prepare_indicators, evaluate_all_candidates

__all__ = [
    "classify_regimes",
    "TradeCandidate",
    "evaluate_sideways_setup",
    "evaluate_trending_setup",
    "prepare_indicators",
    "evaluate_all_candidates"
]
