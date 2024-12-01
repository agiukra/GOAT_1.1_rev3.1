import requests
import config

def execute_order(symbol, side, quantity, price=None, order_type="MARKET"):
    url = f"{config.BASE_URL}/api/v3/order"
    headers = {"X-MBX-APIKEY": config.API_KEY}
    data = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity
    }
    if price:
        data["price"] = price

    response = requests.post(url, headers=headers, data=data)
    return response.json()
