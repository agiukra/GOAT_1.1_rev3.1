import aiohttp
import asyncio
import logging
from datetime import datetime

class AsyncPairAnalyzer:
    def __init__(self, client, max_concurrent=20):
        self.client = client
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger('TradingBot.AsyncAnalyzer')
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache = {}

    async def analyze_pairs(self, pairs):
        """Асинхронный анализ всех пар"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for pair in pairs:
                task = asyncio.ensure_future(
                    self._analyze_pair_safe(session, pair)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return [r for r in results if r]

    async def _analyze_pair_safe(self, session, pair):
        """Безопасный асинхронный анализ пары"""
        try:
            async with self._semaphore:
                # Проверяем кэш
                if pair in self._cache:
                    cache_age = (datetime.now() - self._cache[pair]['timestamp']).total_seconds()
                    if cache_age < 300:  # 5 минут
                        return self._cache[pair]['data']

                # Получаем данные
                ticker = await self._get_ticker(session, pair)
                klines = await self._get_klines(session, pair)
                
                result = self._analyze_data(pair, ticker, klines)
                
                # Сохраняем в кэш
                self._cache[pair] = {
                    'timestamp': datetime.now(),
                    'data': result
                }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Ошибка анализа пары {pair}: {str(e)}")
            return None 