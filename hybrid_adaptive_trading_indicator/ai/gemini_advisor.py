"""
Google Gemini AI Trading Advisor Integration.
Provides AI reasoning, trade setup validation, approval/rejection audits,
and market structure intelligence for Gold and Bitcoin trading.
"""
import os
import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GeminiAdvisor:
    """
    Client for interacting with Google Gemini AI models.
    Supports google.genai and legacy google.generativeai with automatic fallback.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        # Robust built-in .env loader if environment variable not yet set
        if not os.getenv("GEMINI_API_KEY"):
            candidate_paths = [
                os.path.abspath(".env"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
            ]
            for env_path in candidate_paths:
                if os.path.exists(env_path):
                    try:
                        with open(env_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#") and "=" in line:
                                    k, v = line.split("=", 1)
                                    os.environ.setdefault(k.strip(), v.strip())
                        break
                    except Exception:
                        pass

        self.api_key = api_key if api_key is not None else (os.getenv("GEMINI_API_KEY") or "")
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
        self._client = None
        self._legacy_configured = False

        if self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str):
        """Update API key dynamically at runtime."""
        self.api_key = api_key.strip()
        self._client = None
        self._legacy_configured = False
        if self.api_key:
            self._init_client()

    def is_configured(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.api_key and len(self.api_key) > 5)

    def _init_client(self):
        """Initialize Google GenAI client."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.warning(f"Failed to init google.genai client: {e}. Falling back to legacy SDK / REST.")
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._legacy_configured = True
            except Exception as legacy_err:
                logger.warning(f"Failed to init legacy GenAI SDK: {legacy_err}")

    def _get_candidate_models(self) -> list:
        """Return ordered list of main model and working fallback models."""
        env_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "")
        fallback_list = [m.strip() for m in env_fallbacks.split(",") if m.strip()] if env_fallbacks else []
        
        default_working = [
            "gemini-3.5-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash"
        ]

        ordered = [self.model_name] + fallback_list + default_working
        unique_models = []
        for m in ordered:
            if m and m not in unique_models:
                unique_models.append(m)
        return unique_models

    def _generate_rest(self, model: str, prompt: str) -> Optional[str]:
        """Direct REST HTTP call using standard urllib as a fail-safe fallback."""
        import urllib.request
        import urllib.error
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "").strip()
                        if text:
                            return text
        except Exception as e:
            logger.debug(f"Direct REST call failed for {model}: {e}")
        return None

    def _generate(self, prompt: str) -> str:
        """Helper to generate content using available SDK, fallbacks, and REST API."""
        if not self.is_configured():
            return "⚠️ Gemini API Key not configured. Please enter your GEMINI_API_KEY in the sidebar or `.env`."

        candidate_models = self._get_candidate_models()

        # 1. Try google.genai client
        if self._client:
            for model in candidate_models:
                try:
                    response = self._client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                    if response and hasattr(response, "text") and response.text:
                        return response.text.strip()
                except Exception as e:
                    logger.debug(f"google.genai call failed with model {model}: {e}")
                    continue

        # 2. Try legacy google.generativeai
        try:
            import google.generativeai as legacy_genai
            if not self._legacy_configured:
                legacy_genai.configure(api_key=self.api_key)
                self._legacy_configured = True

            for model in candidate_models:
                try:
                    m = legacy_genai.GenerativeModel(model)
                    res = m.generate_content(prompt)
                    if res and hasattr(res, "text") and res.text:
                        return res.text.strip()
                except Exception as legacy_err:
                    logger.debug(f"legacy google.generativeai failed with {model}: {legacy_err}")
                    continue
        except Exception as e:
            logger.debug(f"Legacy generativeai execution error: {e}")

        # 3. Try direct REST API fallback
        for model in candidate_models:
            rest_res = self._generate_rest(model, prompt)
            if rest_res:
                return rest_res

        return "❌ Error: Could not generate analysis from Gemini. Please verify your API key and network connection."

    def analyze_trade_decision(self, trade: Dict[str, Any]) -> str:
        """
        Generates comprehensive trading critique and evaluation of an algorithmic trade decision.
        """
        symbol = trade.get("symbol", "N/A")
        direction = trade.get("direction", "N/A")
        status = trade.get("status", "WAIT")
        rejection_reason = trade.get("rejection_reason", "")
        regime = trade.get("regime", "N/A")
        entry = trade.get("entry_price", 0.0)
        sl = trade.get("stop_loss", 0.0)
        tp1 = trade.get("tp1", 0.0)
        tp2 = trade.get("tp2", 0.0)
        rr = trade.get("risk_reward", 0.0)
        setup_reason = trade.get("setup_reason", "")
        pos_size = trade.get("position_size", 0.0)
        pos_unit = trade.get("position_unit", "")
        risk_usd = trade.get("risk_amount_usd", 0.0)
        time_str = trade.get("candle_timestamp", "") or trade.get("display_datetime", "")

        prompt = f"""
You are an expert Quant & Technical Trading Strategist specializing in Gold (XAUUSD) and Bitcoin (BTCUSDT) intraday strategies.

Analyze the following algorithmic trade signal produced by our Hybrid Adaptive Trading System:

### Trade Signal Details:
- **Asset**: {symbol}
- **Timestamp**: {time_str}
- **Direction**: {direction}
- **Market Regime**: {regime}
- **Algorithmic Risk Engine Status**: {status}
{"- **Rejection Reason**: " + rejection_reason if status == "REJECT" else ""}
- **Entry Price**: ${entry:,.2f}
- **Stop Loss**: ${sl:,.2f}
- **Take Profit (TP1)**: ${tp1:,.2f} (TP2: ${tp2:,.2f})
- **Calculated Risk/Reward**: {rr:.2f}R
- **Calculated Position Size**: {pos_size:.4f} {pos_unit} (Risk: ${risk_usd:,.2f})
- **Technical Trigger**: {setup_reason}

### Instructions:
Provide a concise, highly professional review in clean Markdown formatting:
1. **AI Verdict & Decision Alignment**:
   - State clearly whether you agree or disagree with the Risk Engine's decision ({status}).
   - If REJECTED, evaluate whether the rejection protected capital or if a manual trader should consider it under adjusted conditions.
   - If APPROVED, rate the quality of the setup (Low / Medium / High).
2. **Technical & Market Structure Evaluation**:
   - Evaluate the entry, stop-loss placement, and target relative to the {regime} regime.
   - Is the stop loss placed in a logical zone (protected by support/resistance)?
3. **Key Risks & Traps to Watch**:
   - What could go wrong (e.g. false breakout, chop zone, liquidity grab)?
4. **Actionable Trader Recommendation**:
   - Bulleted execution tip (e.g., wait for retest, take partial profit at TP1, move SL to breakeven, or strictly stay out).

Keep the tone objective, quantitative, and directly actionable.
"""
        return self._generate(prompt)

    def analyze_market_overview(self, symbol: str, metrics: Dict[str, Any]) -> str:
        """
        Generates macro/technical market commentary for an asset based on live indicators.
        """
        price = metrics.get("close", 0.0)
        regime = metrics.get("regime", "N/A")
        adx = metrics.get("adx", 0.0)
        plus_di = metrics.get("plus_di", 0.0)
        minus_di = metrics.get("minus_di", 0.0)
        atr = metrics.get("atr", 0.0)
        res = metrics.get("resistance", 0.0)
        sup = metrics.get("support", 0.0)

        prompt = f"""
You are a senior hedge fund market technician.
Provide a quick 3-bullet market pulse update for {symbol} based on current 15-minute technical readings:
- **Last Price**: ${price:,.2f}
- **Regime**: {regime}
- **ADX**: {adx:.1f} (+DI: {plus_di:.1f}, -DI: {minus_di:.1f})
- **ATR**: ${atr:.2f}
- **Confirmed Resistance**: ${res:,.2f}
- **Confirmed Support**: ${sup:,.2f}

Format your response in 3 concise, crisp bullet points:
- **Trend & Volatility State**
- **Critical Key Levels (Support / Resistance)**
- **Strategic Game Plan for Traders**
"""
        return self._generate(prompt)

    def decide_trade_approval(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-Powered Trade Approval Engine.
        Directly evaluates whether to APPROVE or REJECT the setup using Gemini AI.
        Returns structured decision dict:
        {
            "decision": "APPROVE" | "REJECT",
            "confidence": "HIGH" | "MEDIUM" | "LOW",
            "verdict_summary": str,
            "reason": str,
            "key_risk": str,
            "action_plan": str
        }
        """
        if not self.is_configured():
            return {
                "decision": "WAIT",
                "confidence": "LOW",
                "verdict_summary": "Gemini API key not configured.",
                "reason": "Please configure your GEMINI_API_KEY in the sidebar or .env to enable AI-powered trade decisions.",
                "key_risk": "N/A",
                "action_plan": "Add API key"
            }

        symbol = candidate_data.get("symbol", "N/A")
        direction = candidate_data.get("direction", "N/A")
        regime = candidate_data.get("regime", "N/A")
        entry = float(candidate_data.get("entry_price", 0.0) or 0.0)
        sl = float(candidate_data.get("stop_loss", 0.0) or 0.0)
        tp1 = float(candidate_data.get("tp1", 0.0) or 0.0)
        tp2 = float(candidate_data.get("tp2", 0.0) or 0.0)
        rr = float(candidate_data.get("risk_reward", 0.0) or 0.0)
        setup_reason = candidate_data.get("setup_reason", "")
        adx = candidate_data.get("adx", "N/A")
        atr = candidate_data.get("atr", "N/A")
        res = candidate_data.get("resistance", "N/A")
        sup = candidate_data.get("support", "N/A")

        prompt = f"""
You are the Chief Quantitative Risk Officer & AI Decision Engine for Gold (XAUUSD) and Bitcoin (BTCUSDT) intraday trading.
Your mission is to make the definitive decision: APPROVE or REJECT the following trade setup.

### Candidate Trade Setup:
- **Symbol**: {symbol}
- **Direction**: {direction}
- **Market Regime**: {regime}
- **Entry Price**: ${entry:,.2f}
- **Stop Loss**: ${sl:,.2f}
- **Target (TP1)**: ${tp1:,.2f} (TP2: ${tp2:,.2f})
- **Risk / Reward**: {rr:.2f}
- **Setup Trigger**: {setup_reason}
- **ADX**: {adx} | **ATR**: ${atr} | **Support**: ${sup} | **Resistance**: ${res}

### Evaluation Rules:
1. **Regime Alignment**: In TRENDING regimes, favor strong breakouts with directional momentum (+DI/-DI aligned). In SIDEWAYS regimes, favor mean reversion at support/resistance boundaries. In TRANSITION, be very cautious.
2. **Risk / Reward**: A favorable setup should generally provide minimum 1.3+ R:R with logical stop-loss placement beyond recent swing highs/lows.
3. **Capital Preservation**: If there is high risk of false breakout, chop, or chasing an extended move, REJECT it immediately to protect capital.

You MUST respond ONLY with a raw JSON object (no markdown code blocks, no backticks, no text before or after):
{{
    "decision": "APPROVE",
    "confidence": "HIGH",
    "verdict_summary": "Short 1-sentence verdict summary",
    "reason": "Clear explanation of why you APPROVED or REJECTED this trade",
    "key_risk": "Main danger or false-breakout trap to monitor",
    "action_plan": "Clear execution advice for the trader"
}}
Note: "decision" MUST be strictly either "APPROVE" or "REJECT".
"""
        response_text = self._generate(prompt)

        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        try:
            parsed = json.loads(cleaned)
            dec = str(parsed.get("decision", "")).strip().upper()
            if "APPROVE" in dec:
                parsed["decision"] = "APPROVE"
            elif "REJECT" in dec:
                parsed["decision"] = "REJECT"
            else:
                parsed["decision"] = "REJECT"
            return parsed
        except Exception:
            is_approve = "APPROVE" in response_text.upper() and "REJECT" not in response_text.upper()[:60]
            dec_val = "APPROVE" if is_approve else "REJECT"
            return {
                "decision": dec_val,
                "confidence": "MEDIUM",
                "verdict_summary": f"AI decided to {dec_val} this setup based on technical context.",
                "reason": response_text[:300],
                "key_risk": "Market volatility & fakeout risk",
                "action_plan": "Execute with discipline according to risk parameters."
            }

    def chat_with_user(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handles natural language conversation with the trader on Telegram or UI.
        Provides intelligent, context-aware market analysis in Hindi/Hinglish/English.
        """
        if not self.is_configured():
            return "⚠️ Gemini AI is not configured. Please add GEMINI_API_KEY in your .env file."

        ctx_lines = []
        if context:
            if "balance" in context:
                ctx_lines.append(f"- Current Account Balance: ${context['balance']:,.2f}")
            if "realized_pnl" in context:
                ctx_lines.append(f"- Total Realized P&L: ${context['realized_pnl']:+,.2f}")
            if "open_trades" in context:
                ctx_lines.append(f"- Active Open Positions: {context['open_trades']}")
            if "latest_signals" in context:
                ctx_lines.append(f"- Recent Market Signals: {context['latest_signals']}")

        ctx_str = "\n".join(ctx_lines) if ctx_lines else "No active trades currently."

        prompt = f"""
You are the personal AI Trading Companion for the 'Hybrid Adaptive Trading System' (specializing in Gold XAUUSD and Bitcoin BTCUSDT 15m intraday algorithmic trading).
You are chatting directly with your trader (Ritham) via Telegram.

### Live System Status & Context:
{ctx_str}

### Instructions for Response:
1. Tone: Friendly, smart, disciplined, and realistic. Never give blind financial guarantees; emphasize risk management.
2. Language:
   - If user asks in Hindi or Hinglish (e.g., "bhai kaisa market h", "kya lagta h"), reply naturally in clean, friendly Hinglish (Hindi written in Roman English alphabet).
   - If user asks in English, reply in professional, concise English.
3. Length: Keep it crisp and easy to read on a phone screen (Telegram format, max 2-3 short paragraphs or bullet points). Use emojis nicely.
4. If asked about current trades, P&L, or market conditions, use the live system context above.

User's Question:
"{user_message}"

Your reply:
"""
        reply = self._generate(prompt)
        return reply.strip()
