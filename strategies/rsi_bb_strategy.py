from .base_strategy import SignalStrategy
import logging

class RSIBollingerStrategy(SignalStrategy):
    """Стратегия на основе RSI и полос Боллинджера"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('TradingBot')
    
    def generate_signal(self, data, indicators):
        try:
            if len(data) < 2:
                self.logger.warning("Недостаточно данных для генерации сигнала")
                return 'HOLD'

            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]

            # Проверяем наличие всех необходимых индикаторов
            required_indicators = ['rsi', 'bb_high', 'bb_low', 'sma20', 'sma50', 'close']
            if not all(indicator in last_row.index for indicator in required_indicators):
                self.logger.warning("Отсутствуют необходимые индикаторы")
                return 'HOLD'

            # Условия для сигнала SELL
            sell_conditions = [
                float(last_row['rsi']) > self.config['rsi']['overbought'],
                float(last_row['close']) > float(last_row['bb_high']),
                float(last_row['sma20']) < float(last_row['sma50']),
                float(last_row['close']) < float(prev_row['close'])
            ]

            # Условия для сигнала BUY
            buy_conditions = [
                float(last_row['rsi']) < self.config['rsi']['oversold'],
                float(last_row['close']) < float(last_row['bb_low']),
                float(last_row['sma20']) > float(last_row['sma50']),
                float(last_row['close']) > float(prev_row['close'])
            ]

            # Проверяем условия
            if all(sell_conditions):
                return 'SELL'
            elif all(buy_conditions):
                return 'BUY'
            return 'HOLD'

        except Exception as e:
            self.logger.error(f"Ошибка при генерации сигнала: {str(e)}", exc_info=True)
            return 'HOLD'
    
    def get_name(self):
        return "RSI + Bollinger Bands Strategy"
    
    def get_description(self):
        return """
        Стратегия использует комбинацию RSI и полос Боллинджера для генерации сигналов.
        
        Сигнал на покупку (BUY):
        - RSI < 30 (перепроданность)
        - Цена ниже нижней полосы Боллинджера
        - SMA20 > SMA50 (восходящий тренд)
        - Цена растет
        
        Сигнал на продажу (SELL):
        - RSI > 70 (перекупленность)
        - Цена выше верхней полосы Боллинджера
        - SMA20 < SMA50 (нисходящий тренд)
        - Цена падает
        """
    
    def get_parameters(self):
        return {
            'rsi_period': self.config['rsi']['period'],
            'rsi_overbought': self.config['rsi']['overbought'],
            'rsi_oversold': self.config['rsi']['oversold'],
            'bb_period': self.config['bollinger_bands']['period'],
            'bb_std_dev': self.config['bollinger_bands']['std_dev'],
            'sma_short': self.config['moving_averages']['short_period'],
            'sma_long': self.config['moving_averages']['long_period']
        }

    def validate_data(self, data):
        """Проверка наличия необходимых данных"""
        required_indicators = ['rsi', 'bb_high', 'bb_low', 'sma20', 'sma50', 'close']
        return all(indicator in data.columns for indicator in required_indicators) 