import requests

def fetch_historical_data(symbol, timeframe, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
    response = requests.get(url)
    return response.json()

def parse_data(raw_data):
    return [{"timestamp": d[0], "open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4])} for d in raw_data]
