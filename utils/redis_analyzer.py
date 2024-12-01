import redis
import json
import logging
from datetime import datetime

class RedisPairAnalyzer:
    def __init__(self, client, redis_url='redis://localhost:6379/0'):
        self.client = client
        self.redis = redis.from_url(redis_url)
        self.logger = logging.getLogger('TradingBot.RedisAnalyzer')

    def analyze_pairs_batch(self, pairs):
        """Анализ пар с использованием Redis для кэширования"""
        results = []
        for pair in pairs:
            # Проверяем кэш
            cached = self.redis.get(f"pair_analysis:{pair}")
            if cached:
                data = json.loads(cached)
                if (datetime.now() - datetime.fromisoformat(data['timestamp'])).total_seconds() < 3600:
                    results.append(data['analysis'])
                    continue

            # Анализируем пару
            analysis = self._analyze_pair(pair)
            if analysis:
                # Сохраняем в Redis
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis
                }
                self.redis.setex(
                    f"pair_analysis:{pair}",
                    3600,  # TTL 1 час
                    json.dumps(cache_data)
                )
                results.append(analysis)

        return results 