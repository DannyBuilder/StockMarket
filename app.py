from flask import Flask, render_template, request, jsonify
from datetime import datetime
import yfinance as yf
import json
import os

app = Flask(__name__)

PORTFOLIO_FILE = "portfolio.json"

# Loads Portfolio from JSON file
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {
            "stocks": {},
            "cash": 5000.0,
            "transactions": []
        }
    return data

# Save portfolio to JSON
def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Convert JSON to a list of stocks
def get_positions_list(portfolio):
    positions = []
    for symbol, info in portfolio["stocks"].items():
        positions.append({
            "symbol": symbol,
            "company_name": f"{symbol} Inc.",
            "shares": info["shares"],
            "avg_price": info["avg_price"],
            "current_price": 0,
            "market_value": 0,
            "gain_loss": 0,
            "gain_loss_percent": 0,
            "ytd_percent": 0
        })
    return positions

# Get the current price of a stock from the API
def get_current_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.history(period="1d")['Close'][0]
        return round(price, 2)
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

# Home page
@app.route("/")
def index():
    return render_template("index.html")

# API to get portfolio summary
@app.route("/portfolio_summary")
@app.route("/portfolio_summary")
def portfolio_api():
    portfolio = load_portfolio()
    positions = get_positions_list(portfolio)

    total_value = portfolio["cash"]
    for p in positions:
        current_price = get_current_price(p["symbol"])
        if current_price:
            p["current_price"] = current_price
            p["market_value"] = p["shares"] * current_price
            p["gain_loss"] = (current_price - p["avg_price"]) * p["shares"]
            p["gain_loss_percent"] = ((current_price - p["avg_price"]) / p["avg_price"]) * 100

            portfolio["stocks"][p["symbol"]]["total_cost"] = round(p["avg_price"] * p["shares"], 2)

    total_value += sum(p["market_value"] for p in positions)

    save_portfolio(portfolio)

    summary = {
        "cash": portfolio["cash"],
        "positions": positions,
        "transactions": portfolio.get("transactions", []),
        "total_value": total_value,
        "total_gain_loss": total_value - (sum(p["avg_price"]*p["shares"] for p in positions) + portfolio["cash"]),
        "total_gain_loss_percent": ((total_value - (sum(p["avg_price"]*p["shares"] for p in positions) + portfolio["cash"])) /
                                   (sum(p["avg_price"]*p["shares"] for p in positions) + portfolio["cash"])) * 100
    }
    return jsonify(summary)

# Add stock
@app.route("/add_stock", methods=["POST"])
def add_stock():
    data = request.get_json()
    symbol = data.get("symbol").upper()
    shares = int(data.get("shares"))

    current_price = get_current_price(symbol)
    if current_price is None:
        return jsonify({"error": f"Could not fetch price for {symbol}"}), 400

    portfolio = load_portfolio()

    cost = current_price * shares
    if cost > portfolio["cash"]:
        return jsonify({"error": f"Not enough cash. You need ${cost}, but only have ${portfolio['cash']}."}), 400

    if symbol in portfolio["stocks"]:
        # Update existing position
        existing = portfolio["stocks"][symbol]
        total_shares = existing["shares"] + shares
        total_cost = existing["avg_price"] * existing["shares"] + current_price * shares
        existing["avg_price"] = round(total_cost / total_shares, 2)
        existing["shares"] += shares
        existing["total_cost"] = round(existing["avg_price"] * existing["shares"], 2)
    else:
        # New stock
        portfolio["stocks"][symbol] = {
            "shares": shares,
            "avg_price": current_price,
            "total_cost": round(current_price * shares, 2)
        }

    portfolio["transactions"].append({
        "date": datetime.now().isoformat(),
        "type": "BUY",
        "symbol": symbol,
        "shares": shares,
        "price": current_price,
        "total": round(current_price * shares, 2)
    })

    portfolio["cash"] -= cost
    save_portfolio(portfolio)

    return jsonify({"success": f"Added {shares} shares of {symbol} at ${current_price}!"})


# Sell stock
@app.route("/sell_stock", methods=["POST"])
def sell_stock():
    data = request.get_json()
    symbol = data.get("symbol").upper()
    shares_to_sell = int(data.get("shares"))

    portfolio = load_portfolio()

    if symbol not in portfolio["stocks"]:
        return jsonify({"error": f"No shares of {symbol} found in portfolio."})

    stock = portfolio["stocks"][symbol]
    if shares_to_sell > stock["shares"]:
        return jsonify({"error": f"Not enough shares to sell. You own {stock['shares']} shares."})

    current_price = get_current_price(symbol)
    stock["shares"] -= shares_to_sell
    stock["total_cost"] = stock["avg_price"] * stock["shares"]
    portfolio["cash"] += shares_to_sell * current_price

    portfolio["transactions"].append({
        "date": datetime.now().isoformat(),
        "type": "SELL",
        "symbol": symbol,
        "shares": shares_to_sell,
        "price": current_price,
        "total": round(current_price * shares_to_sell, 2)
    })

    if stock["shares"] == 0:
        del portfolio["stocks"][symbol]

    save_portfolio(portfolio)
    return jsonify({"success": f"Sold {shares_to_sell} shares of {symbol} at ${current_price}!"})

if __name__ == "__main__":
    app.run(debug=True)
