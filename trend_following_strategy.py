class TrendFollowingStrategy(SignalStrategy):
    """Стратегия следования за трендом"""
    
    def __init__(self, config):
        super().__init__(config)
        # Параметры
        self.sma_short = 20
        self.sma_long = 50
        self.atr_period = 14
        self.min_trend_strength = 0.02  # 2%
        
    def generate_signal(self, data):
        try:
            # Расчет индикаторов
            sma_short = ta.trend.SMAIndicator(data['close'], self.sma_short).sma_indicator()
            sma_long = ta.trend.SMAIndicator(data['close'], self.sma_long).sma_indicator()
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], self.atr_period).average_true_range()
            
            # Определение тренда
            trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            
            # Сигналы
            if trend_strength > self.min_trend_strength and sma_short.iloc[-1] > sma_long.iloc[-1]:
                return "BUY"
            elif trend_strength < -self.min_trend_strength and sma_short.iloc[-1] < sma_long.iloc[-1]:
                return "SELL"
                
            return "HOLD"
            
        except Exception as e:
            logging.error(f"Ошибка в TrendFollowing: {str(e)}")
            return "HOLD" 