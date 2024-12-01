class PairAnalyzer:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger('TradingBot.PairAnalyzer')
        self._exchange_info = None
        self._tickers_cache = {}
        self._cache_time = None

    @property
    def exchange_info(self):
        """Кэшированная информация о бирже"""
        if not self._exchange_info:
            self._exchange_info = self.client.get_exchange_info()
        return self._exchange_info

    def get_tickers(self):
        """Получение всех тикеров с кэшированием"""
        now = datetime.now()
        if (not self._cache_time or 
            (now - self._cache_time).total_seconds() > 60):  # Кэш на 1 минуту
            self._tickers_cache = {
                t['symbol']: t for t in self.client.get_ticker()
            }
            self._cache_time = now
        return self._tickers_cache

    def analyze_pair(self, symbol):
        """Анализ конкретной торговой пары"""
        try:
            # Используем кэшированные данные
            ticker = self.get_tickers().get(symbol)
            if not ticker:
                return None

            # Базовые метрики
            price_change = float(ticker['priceChangePercent'])
            volume = float(ticker['volume']) * float(ticker['lastPrice'])
            
            # Быстрый расчет волатильности
            high = float(ticker['highPrice'])
            low = float(ticker['lowPrice'])
            volatility = ((high - low) / low) * 100

            # Оценка пары
            score = self._calculate_pair_score(volatility, volume, price_change)

            return {
                'symbol': symbol,
                'volatility': volatility,
                'avg_daily_volume': volume,
                'price_change_24h': price_change,
                'score': score
            }

        except Exception as e:
            self.logger.error(f"Ошибка анализа пары {symbol}: {str(e)}")
            return None 