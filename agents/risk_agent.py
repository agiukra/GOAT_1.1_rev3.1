import config

class RiskAgent:
    def __init__(self):
        self.risk_tolerance = config.AGENT_SETTINGS["risk_tolerance"]
        self.trade_history = []

    def calculate_position_size(self, balance, entry_price, stop_loss_price):
        stop_loss_distance = entry_price - stop_loss_price
        risk_amount = balance * self.risk_tolerance
        return risk_amount / stop_loss_distance

    def update_risk_tolerance(self):
        # Самообучение: изменить толерантность к риску на основе истории трейдов
        losses = [trade["loss"] for trade in self.trade_history]
        if len(losses) > 0:
            avg_loss = sum(losses) / len(losses)
            self.risk_tolerance = max(0.01, min(0.02, 1 - avg_loss))  # Корректировка толерантности

    def add_trade_result(self, trade_result):
        self.trade_history.append(trade_result)
        self.update_risk_tolerance()
