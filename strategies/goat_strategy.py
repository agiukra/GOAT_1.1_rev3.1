import numpy as np


class GOATStrategy:
    def __init__(self, risk_reward_ratio, max_loss):
        self.risk_reward_ratio = risk_reward_ratio
        self.max_loss = max_loss

    def generate_signal(self, data):
        last_close = data[-1]['close']
        moving_average_50 = np.mean([d['close'] for d in data[-50:]])

        if last_close > moving_average_50:
            return "buy"
        elif last_close < moving_average_50:
            return "sell"
        else:
            return "hold"

    def calculate_stop_loss(self, entry_price):
        return entry_price * (1 - self.max_loss)

    def calculate_take_profit(self, entry_price):
        return entry_price * (1 + self.risk_reward_ratio * self.max_loss)
