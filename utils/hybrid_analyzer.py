import logging
from concurrent.futures import ThreadPoolExecutor

class HybridPairAnalyzer:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger('TradingBot.HybridAnalyzer')
        self.quick_cache = {}
        self.detailed_cache = {}

    def analyze_pairs(self, pairs):
        """Двухэтапный анализ пар"""
        # Этап 1: Быстрая фильтрация
        candidates = self._quick_filter(pairs)
        
        # Этап 2: Детальный анализ
        return self._detailed_analysis(candidates)

    def _quick_filter(self, pairs):
        """Быстрая фильтрация по базовым метрикам"""
        candidates = []
        tickers = self.client.get_ticker()
        ticker_map = {t['symbol']: t for t in tickers}

        for pair in pairs:
            ticker = ticker_map.get(pair)
            if not ticker:
                continue

            volume = float(ticker['volume']) * float(ticker['lastPrice'])
            if volume < 1000000:  # Минимальный объем
                continue

            volatility = abs(float(ticker['priceChangePercent']))
            if volatility < 0.5:  # Минимальная волатильность
                continue

            candidates.append(pair)

        return candidates[:50]  # Ограничиваем количество кандидатов

    def _detailed_analysis(self, pairs):
        """Детальный анализ отфильтрованных пар"""
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self._analyze_pair_detailed, pairs))
        return [r for r in results if r] 