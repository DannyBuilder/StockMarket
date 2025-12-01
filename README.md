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

News & Sentiment
-----------------

You can fetch news headlines and run a simple sentiment analysis using the
new endpoint added to the backend: `/api/news_sentiment?q=QUERY&limit=10`.

1. Get a NewsAPI key from https://newsapi.org and set it in your environment:

```powershell
setx NEWSAPI_KEY "YOUR_NEWSAPI_KEY"
# then open a new terminal so the env var is available
```

2. (Optional) Install Hugging Face transformers & torch to enable model-based
	sentiment analysis:

```powershell
pip install transformers torch
```

3. Call the endpoint (example):

```powershell
curl "http://localhost:5000/api/news_sentiment?q=bitcoin&limit=5"
```

If `transformers` is not installed the endpoint will return an error explaining
how to enable it. The code uses a lazy model initialization so your Flask app
won't attempt to download large models until the endpoint is hit.