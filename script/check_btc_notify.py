#!/usr/bin/env python3
"""
Fetch Bitcoin price (USD) from CoinGecko and send a Telegram message if price
falls below a configurable threshold.
Read Readme.md for usage instructions.
Not directly connected to the StockMarket project.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def get_btc_price_usd() -> float:
    """
       Return current BTC price in USD using CoinGecko public API.
       Raises RuntimeError on failure.
    """
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    req = Request(url, headers={"User-Agent": "btc-price-checker/1.0"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except HTTPError as e:
        raise RuntimeError(f"HTTP error while fetching price: {e.code} {e.reason}")
    except URLError as e:
        raise RuntimeError(f"Network error while fetching price: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while fetching price: {e}")

    try:
        return float(data["bitcoin"]["usd"])
    except Exception:
        raise RuntimeError("Failed to parse response from CoinGecko")


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    """
    Send a message via Telegram bot API.
    Raises RuntimeError on failure.
    """

    token = bot_token.strip()
    chat = chat_id.strip()
    if not token or not chat:
        raise RuntimeError("Empty bot token or chat id")

    payload = json.dumps({"chat_id": chat, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "btc-price-checker/1.0",
    }
    req = Request(url, data=payload, headers=headers)

    try:
        with urlopen(req, timeout=10) as resp:
            resp_data = json.load(resp)
    except HTTPError as e:
        raise RuntimeError(f"Telegram HTTP error: {e.code} {e.reason}")
    except URLError as e:
        raise RuntimeError(f"Network error when contacting Telegram: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error when contacting Telegram: {e}")

    if not resp_data.get("ok"):
        raise RuntimeError(f"Telegram API error: {resp_data}")


def load_env_from_dotenv(dotenv_path: str = ".env") -> None:
    """
    Set environment variables from a file if present.
    """
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(prog="check_btc_notify.py")
    parser.add_argument("--threshold", "-t", type=float, default=105000.0,
                        help="USD threshold to trigger a Telegram alert (default: 105000)")
    parser.add_argument("--loop", "-l", action="store_true",
                        help="Run continuously, checking at intervals")
    parser.add_argument("--interval", "-i", type=int, default=300,
                        help="Interval in seconds when --loop is used (default: 300)")
    parser.add_argument("--dotenv", default=".env", help="Path to .env file to load")

    args = parser.parse_args()

    load_env_from_dotenv(args.dotenv)

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    interval = float(os.environ.get("INTERVAL", args.interval))
    threshold = int(os.environ.get("THRESHOLD", args.threshold))
    loop_env = os.environ.get("LOOP", str(args.loop))
    loop = loop_env.lower() in ("true", "1", "yes")

    if not bot_token or not chat_id:
        print("Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment or .env", file=sys.stderr)
        return 2


    def check_and_notify() -> bool:
        try:
            price = get_btc_price_usd()
        except Exception as e:
            print(f"Failed to get BTC price: {e}")
            return False

        print(f"Current BTC price: ${price:,.2f}")
        if price < threshold:
            text = f"Alert: BTC price is below ${threshold:,.0f}: current ${price:,.2f}"
            try:
                send_telegram_message(bot_token, chat_id, text)
                print("Telegram alert sent")
            except Exception as e:
                print(f"Failed to send Telegram message: {e}")
                return False
            return True
        return False

    
    if loop:
        print(f"Starting loop: check every {interval}s, alert if below ${threshold:,.0f}")
        try:
            while True:
                sent = check_and_notify()
                if sent:
                    print("Message sent — stopping loop")
                    break
                time.sleep(max(1, interval))
        except KeyboardInterrupt:
            print("Exiting on user request")
            return 0
    else:
        check_and_notify()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
