from agents.trading_agent import TradingAgent
from agents.risk_agent import RiskAgent
from agents.data_agent import DataAgent
from agents.sentiment_agent import SentimentAgent
from agents.executor_agent import ExecutorAgent


class Hub:
    def __init__(self):
        self.trading_agent = TradingAgent()
        self.risk_agent = RiskAgent()
        self.data_agent = DataAgent()
        self.sentiment_agent = SentimentAgent()
        self.executor_agent = ExecutorAgent()

    def run(self):
        # Шаг 1: Загрузка данных
        data = self.data_agent.fetch_data()

        # Шаг 2: Обучение торгового агента
        self.trading_agent.train_model(data)

        # Шаг 3: Генерация сигнала
        signal = self.trading_agent.generate_signal(data)

        # Шаг 4: Анализ сентимента
        sentiment_score = self.sentiment_agent.analyze_sentiment("Example news")

        # Шаг 5: Управление рисками
        last_close = data[-1]['close']
        stop_loss = self.trading_agent.strategy.calculate_stop_loss(last_close)
        position_size = self.risk_agent.apply_risk_management(1000, last_close, stop_loss)

        # Шаг 6: Исполнение сделки
        if signal == "buy" and sentiment_score > 0.5:
            trade_result = self.executor_agent.execute_order("BTCUSDT", "BUY", position_size)
            self.risk_agent.add_trade_result(trade_result)
        elif signal == "sell" and sentiment_score < -0.5:
            trade_result = self.executor_agent.execute_order("BTCUSDT", "SELL", position_size)
            self.risk_agent.add_trade_result(trade_result)
        else:
            print("Hold")
