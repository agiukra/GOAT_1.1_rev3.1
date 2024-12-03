from .macd_strategy import MACDStrategy
from .rsi_bb_strategy import RSIBollingerStrategy
from .goat_strategy import GoatStrategy
import logging

class StrategyManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('TradingBot')
        
        # Инициализируем доступные стратегии
        self.strategies = {
            'MACD': MACDStrategy(config),
            'RSI_BB': RSIBollingerStrategy(config),
            'GOAT': GoatStrategy()
        }
        
        self.current_strategy = None
    
    def select_strategy(self, data):
        """Выбор стратегии на основе рыночных условий"""
        try:
            last_row = data.iloc[-1]
            
            # Анализируем волатильность
            volatility = self._calculate_volatility(data)
            
            # Анализируем тренд
            trend_strength = abs(float(last_row['trend_strength']))
            is_trending = trend_strength > 1.0  # Сильный тренд > 1%
            
            # Анализируем объем
            volume_ratio = float(last_row['volume']) / data['volume'].tail(24).mean()
            high_volume = volume_ratio > 1.5
            
            # Выбор стратегии на основе условий
            if is_trending and high_volume:
                # В трендовом рынке с высоким объемом используем MACD
                strategy = 'MACD'
                reason = "Сильный тренд и высокий объем"
            else:
                # В боковом рынке или при низком объеме используем RSI+BB
                strategy = 'RSI_BB'
                reason = "Боковой рынок или низкий объем"
            
            # Если стратегия изменилась, логируем это
            if self.current_strategy != strategy:
                self.logger.info(
                    f"Смена стратегии: {self.current_strategy} -> {strategy}\n"
                    f"Причина: {reason}\n"
                    f"Параметры рынка:\n"
                    f"- Волатильность: {volatility:.2f}%\n"
                    f"- Сила тренда: {trend_strength:.2f}%\n"
                    f"- Объем относительно среднего: {volume_ratio:.2f}"
                )
                self.current_strategy = strategy
            
            return self.strategies[strategy]
            
        except Exception as e:
            self.logger.error(f"Ошибка при выборе стратегии: {str(e)}")
            # В случае ошибки используем RSI+BB как более консервативную стратегию
            return self.strategies['RSI_BB']
    
    def _calculate_volatility(self, data):
        """Расчет волатильности"""
        try:
            returns = data['close'].pct_change()
            return float(returns.std() * (252 ** 0.5) * 100)
        except Exception as e:
            self.logger.error(f"Ошибка расчета волатильности: {str(e)}")
            return 0 
    
    def initialize_strategy(self, strategy_name):
        try:
            if strategy_name == 'GOAT':
                strategy = GoatStrategy()
                # Проверка необходимых атрибутов
                if not hasattr(strategy, 'timeframe'):
                    raise AttributeError("Strategy missing required 'timeframe' attribute")
                return strategy
        except Exception as e:
            self.logger.error(f"Ошибка при выборе стратегии: {str(e)}")
            return None
    
    def check_rsi_conditions(self, rsi):
        """Проверяет условия RSI для входа в позицию"""
        if rsi > 70:
            self.logger.info(f"RSI > 70, рынок перекуплен")
            return False
        elif rsi < 30:
            self.logger.info(f"RSI < 30, рынок перепродан") 
            return False
        return True