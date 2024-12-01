import requests
import config

class ExecutorAgent:
    def __init__(self):
        self.trade_history = []

    def execute_order(self, symbol, side, quantity, price=None, order_type="MARKET"):
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
        trade_result = response.json()
        self.trade_history.append(trade_result)
        return trade_result
