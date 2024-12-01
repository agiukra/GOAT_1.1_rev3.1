import logging
import asyncio
import aiohttp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache

class OptimizedPairAnalyzer:
    def __init__(self, client, max_workers=20):
        self.client = client
        self.logger = logging.getLogger('TradingBot.OptimizedAnalyzer')
        self.max_workers = max_workers
        # Кэш на 5 минут для тикеров и 1 час для анализа
        self.ticker_cache = TTLCache(maxsize=1000, ttl=300)
        self.analysis_cache = TTLCache(maxsize=1000, ttl=3600)
        
    async def analyze_pairs_async(self, pairs):
        """Асинхронный анализ пар с предварительной фильтрацией"""
        try:
            # Этап 1: Быстрая фильтрация
            filtered_pairs = await self._quick_filter(pairs)
            self.logger.info(f"Отфильтровано {len(filtered_pairs)} пар из {len(pairs)}")
            
            # Этап 2: Параллельный детальный анализ
            async with aiohttp.ClientSession() as session:
                tasks = []
                for pair in filtered_pairs:
                    if pair in self.analysis_cache:
                        tasks.append(asyncio.create_task(
                            asyncio.sleep(0, result=self.analysis_cache[pair])
                        ))
                    else:
                        tasks.append(asyncio.create_task(
                            self._analyze_pair(session, pair)
                        ))
                
                results = await asyncio.gather(*tasks)
                valid_results = [r for r in results if r]
                
                # Сортируем по нескольким критериям
                sorted_results = sorted(
                    valid_results,
                    key=lambda x: (
                        x['score'],
                        x['volatility'],
                        x['avg_daily_volume']
                    ),
                    reverse=True
                )
                
                return sorted_results
                
        except Exception as e:
            self.logger.error(f"Ошибка при анализе пар: {str(e)}")
            return []
            
    async def _quick_filter(self, pairs):
        """Быстрая предварительная фильтрация пар"""
        try:
            # Получаем все тикеры одним запросом
            if 'all_tickers' not in self.ticker_cache:
                tickers = await self._fetch_all_tickers()
                self.ticker_cache['all_tickers'] = {t['symbol']: t for t in tickers}
            
            ticker_map = self.ticker_cache['all_tickers']
            filtered = []
            
            for pair in pairs:
                ticker = ticker_map.get(pair)
                if not ticker:
                    continue
                    
                try:
                    volume = float(ticker['volume']) * float(ticker['lastPrice'])
                    volatility = abs(float(ticker['priceChangePercent']))
                    
                    # Базовые фильтры
                    if (volume >= 1_000_000 and  # Минимальный объем
                        volatility >= 0.5 and    # Минимальная волатильность
                        volatility <= 20):       # Максимальная волатильность
                        filtered.append(pair)
                        
                except (ValueError, KeyError):
                    continue
                    
            return filtered[:50]  # Ограничиваем количество пар для детального анализа
            
        except Exception as e:
            self.logger.error(f"Ошибка при фильтрации пар: {str(e)}")
            return pairs[:50]
            
    async def _analyze_pair(self, session, pair):
        """Детальный анализ пары"""
        try:
            # Получаем необходимые данные
            ticker = self.ticker_cache['all_tickers'].get(pair)
            if not ticker:
                return None
                
            klines = await self._fetch_klines(session, pair)
            if not klines:
                return None
                
            # Рассчитываем метрики
            volume = float(ticker['volume']) * float(ticker['lastPrice'])
            volatility = abs(float(ticker['priceChangePercent']))
            
            # Дополнительные метрики из клайнов
            price_changes = []
            volumes = []
            for k in klines[-24:]:  # Последние 24 свечи
                open_price = float(k[1])
                close_price = float(k[4])
                price_changes.append(abs(close_price - open_price) / open_price * 100)
                volumes.append(float(k[5]))
                
            avg_volatility = sum(price_changes) / len(price_changes)
            volume_stability = sum(volumes) / len(volumes)
            
            # Рассчитываем итоговую оценку
            score = (
                (volume / 1_000_000) * 0.4 +     # Объем (40%)
                volatility * 0.3 +                # Волатильность (30%)
                (volume_stability / 1_000_000) * 0.2 +  # Стабильность объема (20%)
                (avg_volatility / 10) * 0.1       # Средняя волатильность (10%)
            )
            
            result = {
                'symbol': pair,
                'score': score,
                'volatility': volatility,
                'avg_daily_volume': volume,
                'volume_stability': volume_stability,
                'avg_volatility': avg_volatility
            }
            
            # Сохраняем в кэш
            self.analysis_cache[pair] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа пары {pair}: {str(e)}")
            return None 