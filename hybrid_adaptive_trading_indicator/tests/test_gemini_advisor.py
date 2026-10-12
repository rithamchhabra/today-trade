"""
Unit tests for Gemini AI Advisor.
"""
import pytest
from hybrid_adaptive_trading_indicator.ai.gemini_advisor import GeminiAdvisor

def test_gemini_advisor_initialization(monkeypatch):
    advisor = GeminiAdvisor(api_key="")
    assert not advisor.is_configured()

    monkeypatch.setattr(advisor, "_init_client", lambda: None)
    advisor.set_api_key("test_api_key_12345")
    assert advisor.is_configured()

def test_gemini_advisor_unconfigured_response():
    advisor = GeminiAdvisor(api_key="")
    trade = {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "status": "APPROVE",
        "entry_price": 2350.0,
        "stop_loss": 2340.0,
        "tp1": 2370.0,
        "risk_reward": 2.0,
        "setup_reason": "Resistance breakout"
    }
    resp = advisor.analyze_trade_decision(trade)
    assert "Gemini API Key not configured" in resp

def test_gemini_advisor_decide_approval_unconfigured():
    advisor = GeminiAdvisor(api_key="")
    res = advisor.decide_trade_approval({"symbol": "BTCUSDT", "direction": "BUY"})
    assert res["decision"] == "WAIT"
    assert "GEMINI_API_KEY" in res["reason"]
