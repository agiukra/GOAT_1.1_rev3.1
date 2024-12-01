from .base_strategy import SignalStrategy
from .goat_strategy import GoatStrategy
import logging

class StrategyManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('TradingBot')
        
        # Инициализируем базовую стратегию без привязки к конкретной паре
        self.strategies = {
            'goat': GoatStrategy({
                'interval': '1m',
                'account_balance': 1000,
                'risk_per_trade': 0.01,
                'max_position_size': 0.1,
                'symbol': 'BTCUSDT'
            })
        }
        
        self.current_strategy = None
    
    def select_strategy(self, data, trading_pairs=None):
        """
        Выбор стратегии на основе рыночных условий
        
        Args:
            data (pd.DataFrame): Данные по текущей паре
            trading_pairs (list, optional): Список доступных торговых пар
        """
        try:
            if not self.current_strategy:
                self.current_strategy = 'goat'
                strategy = self.strategies['goat']
                self.logger.info(f"Выбрана стратегия: GOAT (Greatest Of All Time)")
                self.logger.info(f"Базовые параметры стратегии:")
                self.logger.info(f"- Таймфрейм: {strategy.timeframe}")
                self.logger.info(f"- RSI период: {strategy.rsi_period}")
                self.logger.info(f"- EMA периоды: {strategy.ema_short}/{strategy.ema_medium}/{strategy.ema_long}")

            strategy = self.strategies[self.current_strategy]
            
            # Обновляем trading_pairs в стратегии
            if trading_pairs:
                strategy.update_trading_pairs(trading_pairs)
            
            # Проверяем условия стратегии для текущих данных
            volatility = strategy.calculate_volatility(data)
            volume_ratio = strategy.calculate_volume_ratio(data)
            trend_strength = strategy.calculate_trend_strength(data)
            
            # Проверяем ликвидность
            liquidity = strategy._analyze_liquidity(data)
            
            # Проверяем рыночные условия
            market_conditions = strategy._analyze_market_conditions(
                data, 
                strategy.calculate_rsi(data),
                trend_strength > 0,
                trend_strength < 0
            )
            
            # Если условия подходящие, возвращаем стратегию
            if (liquidity['is_liquid'] and 
                market_conditions['is_tradeable'] and
                volatility >= strategy.min_volatility and
                volume_ratio > 0.8):
                return strategy
                    
            return strategy
            
        except Exception as e:
            self.logger.error(f"Ошибка при выборе стратегии: {str(e)}")
            return None
    
    def _calculate_volatility(self, data):
        """Расчет волатильности"""
        try:
            returns = data['close'].pct_change()
            return float(returns.std() * (252 ** 0.5) * 100)
        except Exception as e:
            self.logger.error(f"Ошибка расчета волатильности: {str(e)}")
            return 0 