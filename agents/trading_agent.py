from strategies.goat_strategy import GOATStrategy
import config
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Пример модели

class TradingAgent:
    def __init__(self):
        self.strategy = GOATStrategy(
            risk_reward_ratio=config.STRATEGY_PARAMETERS["risk_reward_ratio"],
            max_loss=config.STRATEGY_PARAMETERS["max_loss"]
        )
        self.model = RandomForestClassifier()  # Используем RandomForest для прогнозов
        self.is_trained = False

    def train_model(self, data):
        # Обучение модели на данных
        features, labels = self.prepare_data(data)
        self.model.fit(features, labels)
        self.is_trained = True

    def prepare_data(self, data):
        features = []
        labels = []
        for i in range(50, len(data)):
            feature = [d['close'] for d in data[i-50:i]]
            label = 1 if data[i]['close'] > data[i-1]['close'] else 0
            features.append(feature)
            labels.append(label)
        return np.array(features), np.array(labels)

    def generate_signal(self, data):
        # Если модель обучена, используем её для прогнозов
        if self.is_trained:
            features, _ = self.prepare_data(data[-51:])
            prediction = self.model.predict([features[-1]])[0]
            return "buy" if prediction == 1 else "sell"
        else:
            return "hold"  # Или использовать стратегию GOAT
