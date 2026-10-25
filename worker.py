"""
Standalone 24/7 Background Trading Worker.
Runs continuously on Render, VPS, or local machine.
Polls Gold (XAUUSD) & Bitcoin (BTCUSDT) 15m candles every 60s,
manages paper trade life-cycle, and dispatches instant Telegram alerts.
"""
import os
import sys
import time
import signal
import logging
from dotenv import load_dotenv

# Ensure root directory and package directory are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "hybrid_adaptive_trading_indicator")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

load_dotenv()

from hybrid_adaptive_trading_indicator.scanner import BackgroundMarketScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("StandaloneWorker")

def main():
    logger.info("==================================================")
    logger.info("🚀 Starting 24/7 Hybrid Adaptive Trading Worker...")
    logger.info("==================================================")

    scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
    alert_rejections = os.getenv("ALERT_ON_REJECTED", "false").lower() == "true"
    decision_mode = os.getenv("DECISION_MODE", "Gemini AI Autonomous")

    scanner = BackgroundMarketScanner(
        scan_interval_seconds=scan_interval,
        enable_telegram_rejections=alert_rejections,
        decision_mode=decision_mode
    )

    def handle_exit(signum, frame):
        logger.info(f"Received exit signal ({signum}). Shutting down worker gracefully...")
        scanner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    scanner.start(send_startup_alert=True)
    logger.info(f"Worker running. Scanning every {scan_interval}s. Press Ctrl+C to terminate.")

    # Keep main thread alive
    while scanner.is_running:
        time.sleep(1)

if __name__ == "__main__":
    main()
