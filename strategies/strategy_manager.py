from .macd_strategy import MACDStrategy
from .rsi_bb_strategy import RSIBollingerStrategy
import logging

class StrategyManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('TradingBot')
        
        # Инициализируем доступные стратегии
        self.strategies = {
            'MACD': MACDStrategy(config),
            'RSI_BB': RSIBollingerStrategy(config)
        }
        
        self.current_strategy = None
    
    def select_strategy(self, data, trading_pairs=None):
        """Выбор стратегии на основе рыночных условий"""
        try:
            last_row = data.iloc[-1]
            
            # Анализируем волатильность
            volatility = self._calculate_volatility(data)
            
            # Анализируем тренд
            trend_strength = abs(float(last_row.get('trend_strength', 0)))
            is_trending = trend_strength > 1.0  # Сильный тренд > 1%
            
            # Анализируем объем
            volume_ratio = float(last_row['volume']) / data['volume'].tail(24).mean()
            high_volume = volume_ratio > 1.5
            
            # Расширенный анализ рыночных условий
            market_conditions = {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_ratio': volume_ratio,
                'rsi': float(last_row.get('rsi', 50)),
                'price_change': float(last_row['close']) / float(data['close'].iloc[-2]) - 1,
                'bb_position': float(last_row.get('bb_position', 0))
            }
            
            self.logger.info(f"\nАнализ рыночных условий:")
            self.logger.info(f"- Волатильность: {volatility:.2f}%")
            self.logger.info(f"- Сила тренда: {trend_strength:.2f}%")
            self.logger.info(f"- Объем/Средний: {volume_ratio:.2f}")
            self.logger.info(f"- RSI: {market_conditions['rsi']:.2f}")
            self.logger.info(f"- Изменение цены: {market_conditions['price_change']*100:.2f}%")
            
            # Выбор стратегии на основе расширенных условий
            if is_trending and high_volume:
                strategy = 'MACD'
                reason = "Сильный тренд и высокий объем"
            elif volatility > 3.0 and abs(market_conditions['bb_position']) > 0.8:
                strategy = 'RSI_BB'
                reason = "Высокая волатильность и экстремальные уровни BB"
            else:
                strategy = 'RSI_BB'
                reason = "Стандартные рыночные условия"
            
            # Если стратегия изменилась, логируем это
            if self.current_strategy != strategy:
                self.logger.info(
                    f"\nСмена стратегии: {self.current_strategy} -> {strategy}"
                    f"\nПричина: {reason}"
                    f"\nДетальные параметры:"
                    f"\n- Волатильность: {volatility:.2f}%"
                    f"\n- Сила тренда: {trend_strength:.2f}%"
                    f"\n- Объем относительно среднего: {volume_ratio:.2f}"
                    f"\n- RSI: {market_conditions['rsi']:.2f}"
                )
                self.current_strategy = strategy
            
            return self.strategies[strategy]
            
        except Exception as e:
            self.logger.error(f"Ошибка при выборе стратегии: {str(e)}", exc_info=True)
            # В случае ошибки используем RSI_BB как более консервативную стратегию
            return self.strategies['RSI_BB']
    
    def _calculate_volatility(self, data):
        """Расчет волатильности"""
        try:
            returns = data['close'].pct_change()
            return float(returns.std() * (252 ** 0.5) * 100)
        except Exception as e:
            self.logger.error(f"Ошибка расчета волатильности: {str(e)}")
            return 0 