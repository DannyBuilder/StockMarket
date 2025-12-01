This app is a portfolio tracker with easy to use interface and a build in AI predictor to predict sotck prices.
A simple Stock Predictor model that uses open/close price for prediction. Includes script for telegram bots and sentiment analysis still in progress.
Modern front end and design. Has support for portfolio managment and saves your portfolio every time.

This app is for personal use. The portfolio is a test portfolio. This is not financial advice.


## BTC price alert script

A small helper script `check_btc_notify.py` can check the current Bitcoin price (USD)
and send a Telegram message when the price falls below a threshold.

How to run:
1. Copy `.env.example` to `.env` and fill `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
2. (Windows PowerShell) Run once:
```
python .\script\check_btc_notify.py
```
Run sentiment: "http://localhost:5000/api/news_sentiment?q=bitcoin&limit=5"