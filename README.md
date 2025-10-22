This app is a portfolio tracker with easy to use interface and a build in AI predictor to predict sotck prices.


## BTC price alert script

A small helper script `check_btc_notify.py` can check the current Bitcoin price (USD)
and send a Telegram message when the price falls below a threshold.

How to run:
1. Copy `.env.example` to `.env` and fill `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
2. (Windows PowerShell) Run once:
```
python .\script\check_btc_notify.py
```

