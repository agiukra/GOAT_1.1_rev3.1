from .base_strategy import SignalStrategy
from ta.trend import MACD
import logging

class MACDStrategy(SignalStrategy):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('TradingBot')
    
    def generate_signal(self, data, indicators):
        try:
            if not self.validate_data(data):
                self.logger.warning("Недостаточно данных для MACD стратегии")
                return 'HOLD'

            # Рассчитываем MACD
            macd = MACD(
                close=data['close'],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            
            # Получаем значения MACD и сигнальной линии
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_diff'] = macd.macd_diff()
            
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]
            
            # Сигнал на покупку: MACD пересекает сигнальную линию снизу вверх
            if (prev_row['macd'] < prev_row['macd_signal'] and 
                last_row['macd'] > last_row['macd_signal'] and
                last_row['macd_diff'] > 0):
                return 'BUY'
                
            # Сигнал на продажу: MACD пересекает сигнальную линию сверху вниз
            elif (prev_row['macd'] > prev_row['macd_signal'] and 
                  last_row['macd'] < last_row['macd_signal'] and
                  last_row['macd_diff'] < 0):
                return 'SELL'
                
            return 'HOLD'
            
        except Exception as e:
            self.logger.error(f"Ошибка в MACD стратегии: {str(e)}")
            return 'HOLD'
    
    def get_name(self):
        return "MACD Strategy"
    
    def get_description(self):
        return """
        Стратегия на основе индикатора MACD (Moving Average Convergence Divergence).
        
        Сигналы:
        - BUY: MACD пересекает сигнальную линию снизу вверх при положительной разнице
        - SELL: MACD пересекает сигнальную линию сверху вниз при отрицательной разнице
        - HOLD: Нет пересечений или условия не выполнены
        """
    
    def get_parameters(self):
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'signal_threshold': 0  # Минимальная разница для генерации сигнала
        }

    def validate_data(self, data):
        """Проверка наличия необходимых данных"""
        if data is None or len(data) < 2:
            return False
            
        required_columns = ['close']
        return all(col in data.columns for col in required_columns) 